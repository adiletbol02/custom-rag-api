# custom-rag-api

This repository contains a FastAPI application that provides a Retrieval-Augmented Generation (RAG) pipeline using the Gemini model and Redis for caching. It includes a frontend provided by OpenWebUI.

**Prerequisites**

-Docker
-Docker Compose

# Setup

Clone this repository:
```
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

Create a .env file based on .env.example. Example:

GEMINI_API_KEY=your_gemini_api_key

AZURE_OPENAI_API_KEY=''

AZURE_OPENAI_ENDPOINT=''
AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4'

DOCS_DIR=/data/docs

EMBEDDING_MODEL=intfloat/multilingual-e5-large

CHUNK_SIZE=1500

CHUNK_OVERLAP=350


Build and start the application:

`docker-compose up -d`

Copy documents to directory:

`docker cp "Directory of your documents" python-app:/data/docs/`



# Usage

Frontend (OpenWebUI): http://localhost:3000

In OpenWebUI go to ->settings ->connections and add a new connection http://localhost:8000, leave APY Key empty

In the chat choose model "custom-rag"

API: http://localhost:8000 

API Endpoints

GET /v1/models: List available models.

POST /v1/chat/completions: Get chat completions.

POST /update_documents: Update the vectorstore (background task).


Updating the Vectorstore

If you add or modify documents in the data/docs directory, update the vectorstore by calling:

curl -X POST http://localhost:8000/update_documents

This triggers a background task to reprocess the documents.

Data

Documents: Place in data/docs (mounted to /data/docs).
