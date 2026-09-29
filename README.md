# RAG Agentic AI

Agentic RAG chatbot using LangGraph, LangChain, OpenAI, and Pinecone.

## Requirements

- Python 3.10+

## Project Structure

```
rag-agentic-ai/
├── data/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── retriever.py
│   ├── graph.py
│   ├── prompts.py
│   └── schemas.py
├── tests/
│   └── test_queries.py
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```cmd
venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your own values:

```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=agentic-ai-index
```

Never commit the `.env` file; it is listed in `.gitignore`.

## Status

Project skeleton only. No functionality is implemented yet.
