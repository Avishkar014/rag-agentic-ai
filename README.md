# Agentic AI RAG Chatbot

Question-answering service over the Agentic AI eBook (`data/Ebook-Agentic-AI.pdf`).
Relevant passages are retrieved from Pinecone, Gemini generates the answer, and a
grading step checks that the answer is supported by the retrieved passages before
it is returned through a FastAPI API.

## 1. Project Objective

Answer questions about the contents of the Agentic AI eBook using retrieval
augmented generation. Answers must be derived only from the retrieved document
chunks. When the eBook does not cover a question, the system states that the
document does not contain enough information instead of using outside knowledge.

## 2. Architecture

The workflow is a cyclic LangGraph state machine (`src/graph.py`):

```text
START -> retrieve -> generate -> grade -> END
              ^                     |
              |------ retry --------+
```

- `retrieve` embeds the query with `sentence-transformers/all-MiniLM-L6-v2` and
  fetches the top 5 chunks from the Pinecone index.
- `generate` calls `gemini-2.5-flash` and answers using only the retrieved chunks.
- `grade` asks the same model whether the answer is supported by those chunks and
  returns JSON with `grounded`, `confidence_score` and `reason`.
- Routing returns a grounded answer directly. An ungrounded answer triggers one
  more retrieval and generation attempt, then the run ends.

`src/api.py` exposes the compiled graph over HTTP.

## 3. Technologies

| Layer | Technology |
| --- | --- |
| Orchestration | LangGraph, LangChain |
| Vector store | Pinecone serverless index (cosine, 384 dimensions) |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Generation and grading | Google Gemini `gemini-2.5-flash` via `langchain-google-genai` |
| API | FastAPI with Uvicorn |
| PDF loading | `pypdf` via `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| Configuration | `python-dotenv` |

## 4. Installation

Python 3.10 or newer is required.

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 5. Environment Variables

Copy `.env.example` to `.env` and fill in your own values. `.env` is git-ignored
and must never be committed.

```text
PINECONE_API_KEY=<your Pinecone API key>
PINECONE_INDEX_NAME=agentic-ai-index
GOOGLE_API_KEY=<your Google AI Studio API key>
```

## 6. PDF Ingestion

`src/ingestion.py` loads the eBook, splits it into 1000 character chunks with 200
characters of overlap, embeds them with `all-MiniLM-L6-v2` and uploads them to
Pinecone.

```bash
python src/ingestion.py
```

Run it from the project root so the relative path `data/Ebook-Agentic-AI.pdf`
resolves. Re-running it uploads the chunks again, so it is only needed when the
index is empty or the document changes.

## 7. Pinecone Setup

`test_connections.py` creates the serverless index (`aws`, `us-east-1`) when it
does not exist yet and waits until it is ready. `check_index.py` prints the index
name, dimension, metric and host.

```bash
python test_connections.py
python check_index.py
```

The index must use dimension `384` and metric `cosine` because that is what the
embedding model and `langchain-pinecone` expect.

## 8. Running the API

```bash
python -m uvicorn src.api:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## 9. API Request and Response

`GET /health` reports the service status.

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "healthy",
  "service": "Agentic AI RAG Chatbot"
}
```

`POST /chat` accepts a JSON body with a single `query` field.

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"What is Agentic AI?\"}"
```

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": ["chunk 1", "chunk 2"],
  "confidence_score": 1.0
}
```

| Status | Meaning |
| --- | --- |
| 200 | Answer generated |
| 400 | The query is empty or whitespace only |
| 422 | The request body does not contain a `query` field |
| 500 | Pinecone or Gemini failed while answering |

## 10. Sample Queries

```text
What is the core definition of Agentic AI as outlined in the eBook?
What are the main architectural components required to build agentic systems?
What real-world industry use cases for Agentic AI are discussed in the eBook?
Who is the current Prime Minister of India?
```

## 11. Groundedness and Confidence

Every answer is graded before it is returned. The grading node sends the question,
the answer and the retrieved chunks to `gemini-2.5-flash` and expects JSON:

```json
{
  "grounded": true,
  "confidence_score": 0.95,
  "reason": "The answer is directly supported by the retrieved context."
}
```

`grounded` decides whether the answer is accepted, and `confidence_score` is
reported in the API response as a value between 0 and 1 that expresses how
strongly the retrieved chunks support the answer. A grading response that cannot
be parsed is treated as ungrounded with a score of `0.0`, and the run is retried
once before it ends.

## 12. Out-of-Context Behaviour

The generation prompt only allows the retrieved chunks as a knowledge source. For
a question outside the eBook, such as:

```text
Who is the current Prime Minister of India?
```

the returned answer is:

```text
I don't have enough information in the provided document to answer this question.
```

## 13. Project Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   └── retriever.py
├── test_connections.py
├── check_index.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## 14. Limitations

- The knowledge source is the single ingested eBook.
- Chunking is fixed at 1000 characters with 200 characters of overlap, and
  retrieval always returns 5 chunks.
- The embedding model and the Gemini model are hardcoded in `src/ingestion.py`,
  `src/retriever.py` and `src/graph.py`.
- Google and Pinecone failures, expired keys and exhausted Gemini quotas are
  reported as HTTP 500 responses.
- `confidence_score` is produced by the model and is not a deterministic metric.
- `all-MiniLM-L6-v2` is small and fast, so retrieval quality is not domain
  specific.
- The API has no authentication, rate limiting or caching, and answers are not
  streamed.

