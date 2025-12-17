# AI CA2 - Skincare Product Safety Analyzer

A RAG-powered (Retrieval-Augmented Generation) application that analyzes skincare products for safety based on user allergies and skin conditions.

## Overview

This application combines product ingredient extraction, vector database retrieval, and LLM analysis to provide personalized skincare safety recommendations.

## Features

- **Product Analysis**: Analyzes skincare products from known brands
- **Ingredient Extraction**: Fetches product ingredients
- **RAG Integration**: Retrieves relevant research from Pinecone vector database
- **LLM Analysis**: Uses GPT-4o-mini to generate safety assessments
- **Personalized**: Considers user allergies and skin conditions

## Project Structure

```
src/
├── app.py                          # Main orchestration file
├── services/
│   ├── ingredientfetcher.py       # Fetches product ingredients
│   ├── n8n_client.py              # N8N workflow integration
│   └── openai_service.py          # LLM prompt & execution
└── rag_builder/
    ├── rag_builder.py             # Vector database indexing
    ├── rag_retriever.py           # Research retrieval
    └── .env                       # API keys (not committed)
```

## Installation

1. **Clone the repository**
   ```bash
   cd "C:\Users\laura\Dropbox\PC\Desktop\College Work\AI\CA2"
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install langchain langchain-openai langchain-core openai pinecone python-dotenv pymupdf tiktoken
   ```

4. **Set up environment variables**
   
   Create `src/rag_builder/.env` with:
   ```
   OPENAI_API_KEY=your_openai_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   PINECONE_INDEX_NAME=skincare-rag
   PINECONE_HOST=your_pinecone_host_here
   ```

## Usage

### Run the main application

```powershell
.venv\Scripts\python.exe src\app.py
```

Or use VS Code's Run button (▶️) in the top-right corner.

### Example Analysis

The app analyzes products like:
```python
result = run_analysis(
    product_name="Pantene Anti-Dandruff Shampoo",
    allergies=["fragrance"],
    conditions=["eczema"]
)
```

Returns JSON with:
- Safety summary
- Suitability score (0-100)
- Recommendation
- Allergy/condition-specific notes
- Key ingredients analysis

## Dependencies

### Core Dependencies
- `langchain` (1.2.0) - LangChain framework
- `langchain-core` (1.2.2) - Core LangChain components
- `langchain-openai` (1.1.4) - OpenAI integration for LangChain
- `openai` (2.13.0) - OpenAI API client
- `pinecone` (8.0.0) - Vector database client
- `pymupdf` (1.26.6) - PDF processing
- `python-dotenv` (1.2.1) - Environment variable management
- `tiktoken` (0.12.0) - Token counting

## Technical Details

### Pipeline Steps

1. **User Input**: Product name, allergies, conditions
2. **Brand Validation**: Checks against known brands
3. **Ingredient Retrieval**: Fetches ingredients via service
4. **RAG Retrieval**: Queries Pinecone for relevant research
5. **Prompt Construction**: Builds structured prompt with context
6. **LLM Execution**: Analyzes with GPT-4o-mini
7. **JSON Response**: Returns structured safety assessment

### Vector Database

- **Provider**: Pinecone
- **Index**: skincare-rag
- **Embedding Model**: text-embedding-3-large (3072 dimensions)
- **Content**: Research papers and safety data

## Supported Brands

- Pantene
- CeraVe
- La Roche-Posay
- Nivea

## Notes

- Ensure virtual environment is activated before running
- API keys must be configured in `.env` file
- Pinecone index must exist and contain embeddings
- Uses GPT-4o-mini for cost-effective analysis