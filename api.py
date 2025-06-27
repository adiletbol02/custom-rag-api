import uuid
import time
import json
import logging
import os
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List, Optional
from openai import AsyncAzureOpenAI  # Import Azure OpenAI client
from config import DEFAULT_INSTRUCTIONS
from models import ChatCompletionRequest, ChatCompletionResponse, Model, Message, Choice, Usage, StreamChunk, StreamChoice
from vectorstore import update_vectorstore
from langchain_core.documents import Document
from tiktoken import get_encoding
from redisvl.query import VectorQuery, FilterQuery
from redisvl.query.filter import Tag, Text
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

tokenizer = get_encoding("cl100k_base")

async def try_get_from_cache(query: str, embeddings, cache_index):
    query_embedding = np.array(embeddings.embed_query(query), dtype=np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    logger.info(f"Query: '{query}', Embedding norm: {np.linalg.norm(query_embedding)}")
    vector_query = VectorQuery(
        vector=query_embedding.tolist(),
        vector_field_name="query_embedding",
        num_results=1,
        return_fields=["response", "query_text"],
        return_score=True
    )
    cache_results = cache_index.query(vector_query)
    distance_threshold = float(os.getenv("SEMANTIC_CACHE_DISTANCE_THRESHOLD", 0.1))
    logger.debug(f"Cache results: '{cache_results}'")
    if cache_results and len(cache_results) > 0 and 'vector_distance' in cache_results[0]:
        score = float(cache_results[0]['vector_distance'])
        cached_query = cache_results[0]['query_text']
        logger.info(f"Closest cached query: '{cached_query}', score: {score}")
        if score <= distance_threshold:
            logger.info("Cache hit")
            return cache_results[0]['response']
        else:
            logger.info(f"No cache hit, score {score} > threshold {distance_threshold}")
    else:
        logger.info("No cache results found")
    return None

async def generate_response(messages: List[dict], deployment_name: str, openai_client: AsyncAzureOpenAI):
    """Generate a response using the Azure OpenAI model with retry logic for rate limits."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await openai_client.chat.completions.create(
                model=deployment_name,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, retrying after 60 seconds (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(60)
            else:
                logger.error(f"Azure OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {e}")
    logger.error(f"Failed to generate response after {max_retries} attempts")
    raise HTTPException(status_code=429, detail="Rate limit exceeded after retries")

def construct_prompt(query: str, vectorstore, custom_instructions: Optional[str], chat_history: Optional[List[Message]], embeddings):
    """Construct a list of messages for the Azure OpenAI model, including context and history."""
    instructions = custom_instructions or DEFAULT_INSTRUCTIONS
    messages = [{"role": "system", "content": instructions}]

    # Add chat history
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})

    # Add context from vectorstore if available
    if vectorstore:
        try:
            k = int(os.getenv("SIMILARITY_K", 5))
            content_weight = float(os.getenv("CONTENT_WEIGHT", 0.7))
            questions_weight = float(os.getenv("QUESTIONS_WEIGHT", 0.3))
            
            query_embedding = embeddings.embed_query(query)
            vector_query = VectorQuery(
                vector=query_embedding,
                vector_field_name="embedding",
                num_results=k,
                return_fields=["text", "chunk_id", "content_type", "source", "page_numbers", "total_char_count", "questions"],
                return_score=True
            )
            
            content_filter = Tag("content_type") == "content"
            vector_query.set_filter(content_filter)
            content_results = vectorstore.query(vector_query)
            
            questions_filter = Tag("content_type") == "questions"
            vector_query.set_filter(questions_filter)
            questions_results = vectorstore.query(vector_query)
            
            chunk_scores = {}
            for doc in content_results or []:
                chunk_id = doc.get('chunk_id')
                try:
                    score = float(doc.get('vector_score', 0))
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (1 - score) * content_weight
                except (TypeError, ValueError):
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0)
            for doc in questions_results or []:
                chunk_id = doc.get('chunk_id')
                try:
                    score = float(doc.get('vector_score', 0))
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (1 - score) * questions_weight
                except (TypeError, ValueError):
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0)
            
            top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:k]
            
            docs = []
            for chunk_id in top_chunk_ids:
                filter_expr = (Text("chunk_id") == chunk_id) & (Tag("content_type") == "content")
                query1 = FilterQuery(
                    return_fields=["text", "chunk_id", "content_type", "source", "page_numbers", "total_char_count", "questions"]
                )
                query1.set_filter(filter_expr)
                results = vectorstore.query(query1)
                if results:
                    doc_data = results[0]
                    doc = Document(
                        page_content=doc_data['text'],
                        metadata={
                            'chunk_id': doc_data['chunk_id'],
                            'content_type': doc_data['content_type'],
                            'source': doc_data['source'],
                            'page_numbers': doc_data['page_numbers'],
                            'total_char_count': int(doc_data['total_char_count']),
                            'questions': doc_data['questions']
                        }
                    )
                    docs.append(doc)
            
            context = ""
            for doc in docs:
                context += f"Chunk from {doc.metadata.get('source', 'unknown')} (Pages {doc.metadata.get('page_numbers', 'unknown')}):\n{doc.page_content}\n\n"
            messages.append({"role": "user", "content": f"Context:\n{context}\nQuery:\n{query}"})
        except Exception as e:
            logger.error(f"Error constructing prompt: {e}")
            messages.append({"role": "user", "content": query})
    else:
        messages.append({"role": "user", "content": query})

    return messages

async def stream_response(answer: str, model_name: str) -> AsyncGenerator[str, None]:
    """Stream the response in chunks for the client."""
    id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    chunk_size = 50
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i:i+chunk_size]
        chunk_data = StreamChunk(
            id=id,
            created=created,
            model=model_name,
            choices=[StreamChoice(
                index=0,
                delta=Message(role="assistant", content=chunk),
                finish_reason=None
            )]
        )
        yield f"data: {json.dumps(chunk_data.dict())}\n\n"
    final_chunk = StreamChunk(
        id=id,
        created=created,
        model=model_name,
        choices=[StreamChoice(
            index=0,
            delta=Message(role="assistant", content=""),
            finish_reason="stop"
        )]
    )
    yield f"data: {json.dumps(final_chunk.dict())}\n\n"
    yield "data: [DONE]\n\n"

async def chat_completions_v1(request: ChatCompletionRequest, vectorstore, embeddings, cache_index, openai_client: AsyncAzureOpenAI):
    """Handle chat completion requests with Redis semantic caching."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    query = request.messages[-1].content
    chat_history = request.messages[:-1] if len(request.messages) > 1 else []
    
    logger.info(f"Received chat query: {query}")
    
    # Check Redis semantic cache
    cached_answer = await try_get_from_cache(query, embeddings, cache_index)
    if cached_answer is not None:
        answer = cached_answer
    else:
        # Check if query is for generating follow-up questions
        is_followup_query = query.startswith('### Task:\nSuggest')
        is_gen_title = query.startswith('### Task:\nGenerate')
        messages = construct_prompt(query, vectorstore, request.custom_instructions, chat_history, embeddings)
        answer = await generate_response(messages, os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-deployment"), openai_client)
        
        # Only cache if not a follow-up question generation query
        if not is_followup_query or is_gen_title:
            query_embedding = np.array(embeddings.embed_query(query), dtype=np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            cache_data = {
                "query_text": query,
                "response": answer,
                "query_embedding": query_embedding.tolist()
            }
            try:
                cache_index.load([cache_data])
                logger.info("Response cached in Redis")
            except Exception as e:
                logger.error(f"Error caching response: {e}")
        else:
            logger.info("Skipping caching for follow-up question generation query")
    
    if request.stream:
        return StreamingResponse(
            stream_response(answer, request.model),
            media_type="text/event-stream"
        )
    else:
        # Reconstruct prompt for token calculation
        messages = construct_prompt(query, vectorstore, request.custom_instructions, chat_history, embeddings)
        prompt_text = "".join([msg["content"] for msg in messages])
        prompt_tokens = len(tokenizer.encode(prompt_text))
        completion_tokens = len(tokenizer.encode(answer))
        total_tokens = prompt_tokens + completion_tokens
        choice = Choice(index=0, message=Message(role="assistant", content=answer), finish_reason="stop")
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            ),
            system_fingerprint="fp_custom_rag"
        )
        return chat_response

def register_api_routes(app: FastAPI):
    """Register API routes with the FastAPI app."""
    @app.get("/v1/models")
    async def list_models_v1():
        return {"data": [Model(id="custom-rag-azure", created=int(time.time()), owned_by="user")]}

    @app.get("/models")
    async def list_models():
        return await list_models_v1()
    
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        return await chat_completions_v1(request, app.state.vectorstore, app.state.embeddings, app.state.cache_index, app.state.openai_client)

    @app.post("/v1/chat/completions")
    async def chat_completions_v1_endpoint(request: ChatCompletionRequest):
        return await chat_completions_v1(request, app.state.vectorstore, app.state.embeddings, app.state.cache_index, app.state.openai_client)

    @app.post("/update_documents")
    async def update_documents(background_tasks: BackgroundTasks):
        logger.info("Received request to update vectorstore")
        background_tasks.add_task(update_vectorstore, app.state.vectorstore)
        try:
            app.state.cache_index.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        return {"status": "Update started in background, cache cleared"}