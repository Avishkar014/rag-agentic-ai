# Agentic AI RAG Chatbot

A production-oriented Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph, Pinecone, Hugging Face Sentence Transformers, Gemini, and FastAPI**.

The system ingests a PDF knowledge source, splits it into searchable chunks, generates vector embeddings locally using Hugging Face, stores them in Pinecone, retrieves relevant context for a user query, and uses Gemini to generate a grounded response.

---

## 1. Project Overview

### Problem

Large language models can generate useful answers but may produce information that is not present in a specific knowledge base.

This project solves that problem by implementing a RAG pipeline that:

1. Loads information from a PDF.
2. Splits the PDF into smaller chunks.
3. Generates vector embeddings using Hugging Face.
4. Stores embeddings in Pinecone.
5. Retrieves relevant document chunks for a query.
6. Passes the retrieved context to Gemini.
7. Generates a grounded answer.
8. Exposes the complete pipeline through a FastAPI API.

### Target Use Case

The chatbot is designed for question answering over a specific document or knowledge base.

The current knowledge source is an Agentic AI ebook.

---

## 2. Key Features

* PDF document ingestion
* Automatic document chunking
* Local Hugging Face embeddings
* Pinecone vector database
* Semantic similarity search
* LangGraph-based RAG workflow
* Gemini-powered answer generation
* Context-grounded responses
* Out-of-context question handling
* Confidence score
* FastAPI REST API
* Swagger/OpenAPI documentation
* Health-check endpoint
* Environment-based configuration
* API response validation with Pydantic

---

## 3. Architecture

```text
                    ┌─────────────────────┐
                    │       PDF File      │
                    │ Ebook-Agentic-AI    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PDF Loader       │
                    │   PyPDFLoader       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Splitter     │
                    │  Chunk Generation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hugging Face Model  │
                    │ all-MiniLM-L6-v2    │
                    │    384 dimensions   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Pinecone       │
                    │   Vector Database   │
                    └──────────┬──────────┘
                               │
                               │ Semantic Search
                               ▼
┌──────────────┐     ┌─────────────────────┐
│ User Query   │────►│    Retriever        │
└──────────────┘     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     LangGraph       │
                     │    RAG Workflow     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │       Gemini        │
                     │   Answer Generator  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │    Final Answer     │
                     │ Context + Confidence│
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │      FastAPI        │
                     │      /chat          │
                     └─────────────────────┘
```

---

## 4. RAG Workflow

The application follows this workflow:

```text
User Query
    ↓
FastAPI /chat
    ↓
LangGraph
    ↓
Query Embedding
    ↓
Pinecone Similarity Search
    ↓
Top Relevant Chunks
    ↓
Context Construction
    ↓
Gemini
    ↓
Grounded Answer
    ↓
Confidence Score
    ↓
FastAPI Response
```

The system uses the uploaded document as the primary knowledge source.

If the retrieved context does not contain enough information to answer the question, the system is designed to return an insufficient-information response instead of relying on unsupported information.

---

## 5. Technology Stack

| Component              | Technology                               |
| ---------------------- | ---------------------------------------- |
| Language               | Python                                   |
| API                    | FastAPI                                  |
| API Server             | Uvicorn                                  |
| RAG Orchestration      | LangGraph                                |
| LLM                    | Google Gemini                            |
| Embeddings             | Hugging Face Sentence Transformers       |
| Embedding Model        | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database        | Pinecone                                 |
| PDF Processing         | PyPDFLoader                              |
| Validation             | Pydantic                                 |
| Environment Management | python-dotenv                            |

---

## 6. Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── ingestion.py
│   ├── retriever.py
│   └── graph.py
│
├── check_index.py
├── test_connections.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Important Files

#### `src/ingestion.py`

Responsible for:

* Loading the PDF
* Splitting documents into chunks
* Generating embeddings
* Uploading vectors to Pinecone

#### `src/retriever.py`

Responsible for:

* Creating the Hugging Face embedding model
* Connecting to Pinecone
* Performing semantic similarity search
* Returning relevant document chunks

#### `src/graph.py`

Responsible for:

* Defining the LangGraph RAG workflow
* Retrieving context
* Calling Gemini
* Generating the final answer
* Producing confidence information

#### `src/api.py`

Responsible for:

* FastAPI application
* `/health` endpoint
* `/chat` endpoint
* Request validation
* Response validation
* Error handling

#### `src/config.py`

Responsible for:

* Loading environment variables
* Pinecone configuration
* Google Gemini configuration

---

# 7. Requirements

Before running the project, install:

* Python 3.10+
* Pinecone account
* Google AI Studio / Gemini API key
* Hugging Face account/token is optional for this embedding model

---

# 8. Installation

Clone the repository:

```bash
git clone https://github.com/Avishkar014/rag-agentic-ai.git
```

Move into the project:

```bash
cd rag-agentic-ai
```

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 9. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# 10. Environment Variables

Create a `.env` file in the project root.

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
GOOGLE_API_KEY=your_google_api_key
```

Optional Hugging Face authentication:

```env
HF_TOKEN=your_huggingface_token
```

Do not commit `.env` to GitHub.

The repository contains `.env.example` for reference.

---

# 11. Pinecone Configuration

The embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

produces vectors with:

```text
Dimension: 384
```

Therefore the Pinecone index must also use:

```text
Dimension: 384
Metric: cosine
```

Example:

```text
Index Name: agentic-ai-index
Dimension: 384
Metric: cosine
```

The embedding dimension and Pinecone index dimension must match.

---

# 12. Document Ingestion

The current PDF contains:

```text
60 pages
```

The ingestion process creates:

```text
119 chunks
```

Run:

```powershell
python .\src\ingestion.py
```

Expected output includes:

```text
Loaded 60 pages from PDF.
Created 119 chunks.
Uploading chunks to Pinecone...
Successfully uploaded chunks to Pinecone.
```

After successful ingestion, the document vectors are available in Pinecone.

---

# 13. Test the Retriever

Run:

```powershell
python .\src\retriever.py
```

Example query:

```text
What is Agentic AI?
```

The retriever returns relevant chunks from the uploaded document.

Example retrieved content includes:

```text
Agentic AI refers to systems capable of autonomous decision-making
and action in pursuit of specific objectives.
```

---

# 14. Test the LangGraph RAG Pipeline

Run:

```powershell
python .\src\graph.py
```

Example:

```text
Query:
What is Agentic AI?

Final Answer:
Agentic AI refers to systems capable of autonomous decision-making
and action in pursuit of specific objectives.

Grounded:
True
```

The pipeline also returns:

```text
Confidence Score
Grading Reason
Retrieved Context
```

---

# 15. Run the FastAPI Server

Start the API:

```powershell
python -m uvicorn src.api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

# 16. Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive interface for testing the API.

---

# 17. Health API

### Endpoint

```http
GET /health
```

Example:

```bash
curl -X GET http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "healthy",
  "service": "Agentic AI RAG Chatbot"
}
```

---

# 18. Chat API

### Endpoint

```http
POST /chat
```

Request:

```json
{
  "query": "What is Agentic AI?"
}
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is Agentic AI?\"}"
```

---

# 19. Response Structure

The `/chat` endpoint returns:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "Relevant document chunk 1",
    "Relevant document chunk 2",
    "Relevant document chunk 3"
  ],
  "confidence_score": 1.0
}
```

### Required Response Fields

| Field                      | Type         | Description                             |
| -------------------------- | ------------ | --------------------------------------- |
| `query`                    | string       | User's question                         |
| `final_answer`             | string       | Generated answer                        |
| `retrieved_context_chunks` | list[string] | Retrieved document context              |
| `confidence_score`         | float        | Confidence returned by the RAG pipeline |

---

# 20. Out-of-Context Question Handling

The system is designed to avoid answering questions that cannot be supported by the uploaded document.

For example:

```text
Question:
Who is the current Prime Minister of India?
```

The system can return:

```text
I don't have enough information in the provided document to answer this question.
```

This behavior helps reduce unsupported or hallucinated answers.

---

# 21. API Validation

Empty queries are rejected.

Example:

```json
{
  "query": ""
}
```

The API returns:

```text
400 Bad Request
```

Invalid request structures are handled through FastAPI/Pydantic validation and return:

```text
422 Unprocessable Entity
```

---

# 22. Connection Testing

The project includes:

```text
test_connections.py
```

Run:

```powershell
python .\test_connections.py
```

This verifies the Pinecone connection and index availability.

You can also verify the Pinecone index using:

```powershell
python .\check_index.py
```

Expected configuration:

```text
Index name: agentic-ai-index
Dimension: 384
Metric: cosine
Status: Ready
```

---

# 23. Submission Requirement Checklist

### Repository

* [x] GitHub repository
* [x] Project source code
* [x] `.gitignore`
* [x] `.env.example`

### RAG Pipeline

* [x] PDF ingestion
* [x] Document chunking
* [x] Hugging Face embeddings
* [x] Pinecone vector storage
* [x] Semantic retrieval
* [x] LangGraph workflow
* [x] Gemini generation
* [x] Grounded response generation

### API

* [x] FastAPI application
* [x] `GET /health`
* [x] `POST /chat`
* [x] Request validation
* [x] Response validation
* [x] Swagger/OpenAPI documentation

### Response Requirements

* [x] `final_answer`
* [x] `retrieved_context_chunks`
* [x] `confidence_score`

### Testing

* [x] Pinecone connection tested
* [x] Embedding model tested
* [x] Retriever tested
* [x] LangGraph pipeline tested
* [x] FastAPI health endpoint tested
* [x] Chat endpoint tested
* [x] Empty query validation tested
* [x] Out-of-context query tested

---

# 24. Security

API keys are stored in environment variables.

The following file should never be committed:

```text
.env
```

The `.gitignore` contains:

```text
.env
venv/
.venv/
__pycache__/
*.pyc
.cache/
.pytest_cache/
```

Only `.env.example` should be committed with empty placeholder values.

---

# 25. Current Limitations

### Gemini API Quota

The Gemini free tier has request limits.

If the configured Gemini project reaches its quota, `/chat` can return:

```text
429 Too Many Requests
```

This is an external API quota limitation and does not indicate a failure of the retrieval or Pinecone components.

The RAG pipeline can otherwise be tested through the local LangGraph pipeline when Gemini quota is available.

### Hugging Face Authentication

The embedding model can be downloaded without authentication, but Hugging Face may display an unauthenticated-request warning.

A Hugging Face token can be configured through:

```env
HF_TOKEN=your_huggingface_token
```

---

# 26. Design Decisions

### Why Hugging Face Embeddings?

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for local embedding generation.

This avoids depending on OpenAI's embedding API and produces 384-dimensional vectors suitable for the configured Pinecone index.

### Why Pinecone?

Pinecone provides vector storage and similarity search for retrieving semantically relevant document chunks.

### Why LangGraph?

LangGraph provides a structured workflow for coordinating retrieval, generation, and response processing.

### Why FastAPI?

FastAPI provides:

* REST API support
* Pydantic validation
* Automatic OpenAPI documentation
* Swagger UI
* Easy local deployment

### Why Gemini?

Gemini is used as the generation model to produce natural-language responses from the retrieved document context.

---

# 27. End-to-End Execution

For a fresh setup, use the following sequence:

```powershell
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

Configure:

```text
.env
```

Then ingest the document:

```powershell
python .\src\ingestion.py
```

Test retrieval:

```powershell
python .\src\retriever.py
```

Test the RAG pipeline:

```powershell
python .\src\graph.py
```

Start the API:

```powershell
python -m uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Then test:

```http
POST /chat
```

with:

```json
{
  "query": "What is Agentic AI?"
}
```

---

# 28. Example RAG Flow

For the query:

```text
What is Agentic AI?
```

the system performs:

```text
1. Receive user query
       ↓
2. Convert query into embedding
       ↓
3. Search Pinecone
       ↓
4. Retrieve relevant document chunks
       ↓
5. Construct context
       ↓
6. Send context + query to Gemini
       ↓
7. Generate grounded answer
       ↓
8. Calculate/return confidence
       ↓
9. Return API response
```

---

# 29. Future Improvements

Potential improvements include:

* Streaming responses
* Conversation history
* Authentication
* Persistent chat sessions
* Better document parsing
* Reranking retrieved chunks
* Hybrid keyword + vector search
* Citation/page references in answers
* Automated evaluation datasets
* Docker deployment
* Cloud deployment
* Observability and tracing
* Rate limiting
* Automated test suite

---

# 30. Repository

GitHub Repository:

https://github.com/Avishkar014/rag-agentic-ai

---

## Assignment Deliverables

This repository provides:

* Functional document ingestion
* Vector embedding and storage
* Semantic retrieval
* LangGraph RAG pipeline
* Gemini-based generation
* FastAPI API
* Health endpoint
* Chat endpoint
* Structured API responses
* Confidence score
* Retrieved context
* Setup instructions
* Architecture documentation
* Testing instructions
* Environment configuration
