import os
import time
import json
import logging
import re
import uuid
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
from document_processing import load_document
import hashlib
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery
from redisvl.query.filter import Tag, Text

logger = logging.getLogger(__name__)

# Cache for storing generated questions
question_cache = {}

async def generate_questions_async(text, model_name="gemini-2.0-flash", max_retries=4):
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    if cache_key in question_cache:
        logger.debug("Returning cached questions")
        return question_cache[cache_key]

    prompt = f"Сгенерируй 10 вопросов на которые отвечает следующий текст:\n\n{text}\n\n Определи язык и отвечай только на этом языке. Прономеруй все вопросы (например, 1. Вопрос):"
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
            questions = response.text.split('\n')
            questions = [q.strip() for q in questions if q.strip() and q[0].isdigit()]
            questions = questions[:10]
            question_cache[cache_key] = questions
            return questions
        except Exception as e:
            if "429 you exceeded your current quota" in str(e).lower():
                wait_time = 2 ** attempt * 15
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt+1}/{max_retries}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Error generating questions: {e}")
                await asyncio.sleep(5)
    logger.error(f"Failed to generate questions after {max_retries} attempts")
    return []

async def load_and_process_documents(docs_dir, redis_client):
    processed_files_key = "processed_files"
    processed_files = {}
    try:
        processed_files_data = redis_client.get(processed_files_key)
        if processed_files_data:
            processed_files = json.loads(processed_files_data)
            logger.info("Loaded processed files metadata from Redis")
    except Exception as e:
        logger.error(f"Error loading processed files metadata from Redis: {e}")

    current_files = {}
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in {'.txt', '.pdf', '.docx', '.csv'}:
                    mtime = os.path.getmtime(file_path)
                    current_files[file] = mtime
    else:
        logger.warning("Docs folder does not exist. Creating it.")
        os.makedirs(docs_dir)
        return [], processed_files

    new_files = [file for file, mtime in current_files.items() if file not in processed_files or processed_files[file] < mtime]

    if not new_files:
        logger.info("No new or modified files detected")
        return [], processed_files

    logger.info(f"Detected {len(new_files)} new or modified files: {new_files}")
    new_documents = []
    for file in new_files:
        file_path = os.path.join(docs_dir, file)
        try:
            loaded_docs = load_document(file_path)
            new_documents.extend(loaded_docs)
            logger.info(f"Loaded new/modified {file}: {len(loaded_docs)} documents")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not new_documents:
        logger.warning("No new documents loaded")
        return [], processed_files

    logger.info(f"Total new/modified documents: {len(new_documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1500)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 350))
    )
    splits = text_splitter.split_documents(new_documents)

    if not splits:
        logger.warning("No new document splits created")
        return [], processed_files

    logger.info(f"Created {len(splits)} new document splits")
    enhanced_splits = []

    async def process_split(i, split):
        logger.info(f"Processing chunk {i+1}/{len(splits)}")
        chunk_id = str(uuid.uuid4())
        page_matches = re.findall(r'\[Page (\d+)\]', split.page_content)
        if page_matches:
            split.metadata['page_numbers'] = ','.join(sorted(set(page_matches)))
        else:
            split.metadata['page_numbers'] = split.metadata.get('page_numbers', '')
        split.metadata['total_char_count'] = len(split.page_content)
        split.metadata['chunk_id'] = chunk_id
        split.metadata['content_type'] = 'content'
        if len(split.page_content.strip()) > 100:
            logger.info(f"Generating questions for chunk {i+1}")
            questions = await generate_questions_async(split.page_content)
            split.metadata['questions'] = '\n'.join(questions)
        else:
            logger.info(f"Skipping question generation for chunk {i+1} due to insufficient content")
            split.metadata['questions'] = ''

        enhanced_splits.append(split)

        if split.metadata['questions']:
            questions_doc = Document(
                page_content=split.metadata['questions'],
                metadata={
                    'chunk_id': chunk_id,
                    'content_type': 'questions',
                    'source': split.metadata.get('source', 'unknown'),
                    'page_numbers': split.metadata.get('page_numbers', ''),
                    'total_char_count': len(split.metadata['questions']),
                    'questions': split.metadata['questions']
                }
            )
            enhanced_splits.append(questions_doc)

    async def process_all_splits():
        batch_size = 3
        tasks = [process_split(i, split) for i, split in enumerate(splits)]
        for i in range(0, len(tasks), batch_size):
            await asyncio.gather(*tasks[i:i + batch_size])

    await process_all_splits()
    for file in new_files:
        processed_files[file] = current_files[file]

    # Save processed files metadata back to Redis
    try:
        redis_client.set(processed_files_key, json.dumps(processed_files))
        logger.info("Updated processed files metadata in Redis")
    except Exception as e:
        logger.error(f"Error saving processed files metadata to Redis: {e}")

    return enhanced_splits, processed_files

def add_to_vectorstore(index, embeddings, enhanced_splits):
    if not enhanced_splits:
        logger.info("No enhanced splits to add")
        return

    data = []
    for doc in enhanced_splits:
        try:
            embedding = embeddings.embed_documents([doc.page_content])[0]
            data.append({
                "text": doc.page_content,
                "chunk_id": doc.metadata['chunk_id'],
                "content_type": doc.metadata['content_type'],
                "source": doc.metadata['source'],
                "page_numbers": doc.metadata['page_numbers'],
                "total_char_count": doc.metadata['total_char_count'],
                "questions": doc.metadata['questions'],
                "embedding": embedding
            })
        except Exception as e:
            logger.error(f"Error processing document: {e}")

    try:
        index.load(data)
        logger.info(f"Added {len(data)} documents to Redis index")
    except Exception as e:
        logger.error(f"Error loading documents to Redis: {e}")

async def initialize_vectorstore():
    logger.info("Initializing vectorstore")
    docs_dir = os.getenv("DOCS_DIR", "/data/docs")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        logger.info("Embeddings initialized")

        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True)
        schema = IndexSchema.from_dict({
            "index": {
                "name": "documents",
                "prefix": "doc",
                "storage_type": "json",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "chunk_id", "type": "text"},
                {"name": "content_type", "type": "tag"},
                {"name": "source", "type": "text"},
                {"name": "page_numbers", "type": "text"},
                {"name": "total_char_count", "type": "numeric"},
                {"name": "questions", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": 1024,
                        "distance_metric": "cosine",
                        "datatype": "float32"
                    }
                }
            ]
        })
        index = SearchIndex(schema, redis_client)
        if not index.exists():
            index.create()
            logger.info("Created new Redis index")
        else:
            logger.info("Connected to existing Redis index")

        enhanced_splits, processed_files = await load_and_process_documents(docs_dir, redis_client)
        add_to_vectorstore(index, embeddings, enhanced_splits)

        test_text = "This is a test."
        test_embedding = embeddings.embed_documents([test_text])
        logger.info(f"Test embedding length: {len(test_embedding[0])}")

        return index
    except Exception as e:
        logger.error(f"Vectorstore initialization error: {e}", exc_info=True)
        return None

async def update_vectorstore(index):
    logger.info("Starting vectorstore update")
    docs_dir = os.getenv("DOCS_DIR", "./docs")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True)
        enhanced_splits, processed_files = await load_and_process_documents(docs_dir, redis_client)
        add_to_vectorstore(index, embeddings, enhanced_splits)
    except Exception as e:
        logger.error(f"Vectorstore update error: {e}", exc_info=True)