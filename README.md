# Multi-Industry RAG Document Intelligence System

AI-powered document Q&A system with industry-specific filtering for service businesses

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## PROBLEM STATEMENT

Service businesses (legal firms, healthcare providers, real estate agencies, HVAC companies) need AI assistants that can answer questions from their industry-specific documents without mixing information across different business verticals.

_Example challenge:_ A healthcare clinic and a law firm both use the same AI system, but queries about "patient records" should never return information from legal contracts, and vice versa.

## SOLUTION

A production-ready RAG (Retrieval-Augmented Generation) system that:
1. Accepts PDF uploads tagged by industry
2. Stores documents in separate filtered knowledge bases
3. Answers questions using ONLY relevant industry documents
4. Prevents information leakage across business verticals

## ARCHITECTURE

```
User Question → FastAPI Endpoint → FAISS Vector Search (filtered by industry) 
→ Retrieve Top 3 Relevant Chunks → LLM Generates Answer → Return JSON Response
```

_Key Components:_
1. _Document Ingestion:_ PDFs converted to text chunks, embedded, and stored with industry metadata
2. _Semantic Search:_ FAISS vector database performs similarity search filtered by industry tag
3. _Answer Generation:_ Groq LLaMA 3.3 generates responses using only retrieved context

## TECH STACK

1. _Backend:_ FastAPI (async Python web framework)
2. _Vector Database:_ FAISS (Facebook AI Similarity Search)
3. _Embeddings:_ HuggingFace all-MiniLM-L6-v2 (384-dimensional)
4. _LLM:_ Groq LLaMA 3.3 70B (fast inference)
5. _PDF Processing:_ PyPDFLoader (Langchain)
6. _Deployment:_ Render (cloud platform)

## FEATURES

1. Multi-industry support (healthcare, legal, real-estate, hvac, education, finance)
2. Industry-filtered semantic search (prevents cross-contamination)
3. Incremental document indexing (add new docs without rebuilding database)
4. Comprehensive error handling (file validation, API failures, empty queries)
5. Structured logging (production debugging)
6. Environment-based configuration (secure API key management)
7. RESTful API design (JSON responses, HTTP status codes)

## INSTALLATION

_Prerequisites:_
- Python 3.11+
- pip package manager
- Groq API key (get one at console.groq.com)

_Setup Steps:_

1. Clone repository:
```bash
git clone https://github.com/ifechukwu-commit/rag-document.git
cd rag-document
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables - Create .env file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the server:
```bash
python main.py
```

Server starts at http://localhost:8000

## API USAGE

### 1. Upload Document

_Endpoint:_ POST /ingest

_Parameters:_
- industry (form field): Industry tag (healthcare, legal, real-estate, hvac, education, finance)
- file (file upload): PDF document

_Example (cURL):_
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "industry=healthcare" \
  -F "file=@patient_handbook.pdf"
```

_Response:_
```json
{
  "message": "Successfully added patient_handbook.pdf to healthcare storage",
  "pages": 45,
  "industry": "healthcare"
}
```

### 2. Ask Question

_Endpoint:_ POST /ask

_Parameters:_
- question (form field): User question
- industry (form field): Industry to search within

_Example (cURL):_
```bash
curl -X POST "http://localhost:8000/ask" \
  -F "question=What are the HIPAA compliance requirements?" \
  -F "industry=healthcare"
```

_Response:_
```json
{
  "answer": "HIPAA compliance requires healthcare providers to implement administrative, physical, and technical safeguards...",
  "industry_used": "healthcare",
  "sources_found": 3
}
```

### 3. Health Check

_Endpoint:_ GET /

_Response:_
```json
{
  "status": "running",
  "message": "Multi-Industry RAG API",
  "endpoints": {
    "/ingest": "Upload PDFs with industry tags",
    "/ask": "Ask questions filtered by industry"
  }
}
```

## HOW IT WORKS

### Document Ingestion Flow

1. _File Upload:_ User sends PDF + industry tag via /ingest endpoint
2. _Validation:_ System checks file type (PDF only) and industry tag validity
3. _Text Extraction:_ PyPDFLoader converts PDF pages to text documents
4. _Metadata Tagging:_ Each document chunk tagged with industry label
5. _Embedding Generation:_ HuggingFace model converts text to 384-dimensional vectors
6. _Vector Storage:_ FAISS stores embeddings with metadata in persistent index
7. _Cleanup:_ Temporary files removed

### Query Processing Flow

1. _Question Received:_ User sends question + industry tag via /ask endpoint
2. _Input Validation:_ System checks question length and industry validity
3. _Database Check:_ Verifies FAISS index exists
4. _Embedding Query:_ Question converted to vector using same embedding model
5. _Filtered Search:_ FAISS retrieves top 3 similar chunks WHERE industry = specified tag
6. _Context Building:_ Retrieved chunks concatenated into context string
7. _LLM Prompt:_ Context + question sent to Groq LLaMA 3.3
8. _Answer Generation:_ LLM generates response using ONLY provided context
9. _Response:_ Structured JSON returned to user

## PRODUCTION CONSIDERATIONS

_Error Handling:_
1. File type validation (PDF only)
2. Industry tag whitelist
3. Empty query rejection
4. Question length limits (500 char max)
5. Try/except blocks around all external operations

_Logging:_
1. All ingestion events tracked
2. Error details captured with timestamps
3. Query processing logged for debugging
4. File operations monitored

_Security:_
1. API keys stored in environment variables (never hardcoded)
2. Input validation on all endpoints
3. HTTP exception handling with appropriate status codes

## DEPLOYMENT

### Deploy to Render

1. Push code to GitHub
2. Connect repository to Render
3. Configure environment variables (GROQ_API_KEY)
4. Deploy automatically

_Build Command:_ pip install -r requirements.txt  
_Start Command:_ uvicorn main:app --host 0.0.0.0 --port $PORT

## USE CASES

_Healthcare Clinics:_
- Upload patient handbooks, HIPAA policies, treatment protocols
- Staff queries get clinic-specific answers only

_Law Firms:_
- Upload contract templates, legal precedents, firm policies
- Queries return firm-specific information only

_Real Estate Agencies:_
- Upload property listings, contract templates, local regulations
- Agents get relevant procedures for their market

_HVAC Companies:_
- Upload service manuals, safety procedures, technical specifications
- Technicians get equipment-specific guidance

## ROADMAP

1. Add file size limits (10MB max)
2. Implement response time tracking
3. Add /list-industries endpoint
4. Write unit tests (pytest)
5. Support .docx and .txt files
6. Add conversation history
7. Implement caching for common queries

## CONTRIBUTING

This is a personal learning project, but feedback is welcome. Open an issue or submit a pull request.

## LICENSE

MIT License - feel free to use this for learning or commercial projects.

## AUTHOR

_Ifechukwu Chiwetalu_  
Applied AI Engineer  

Email: ifechukwudarlington.dev@gmail.com  
LinkedIn: https://www.linkedin.com/in/ifechukwu-chiwetalu-001929271  
GitHub: https://github.com/ifechukwu-commit

---

_Built with AI assistance (Claude, ChatGPT) - modern development practice for faster iteration while maintaining deep system understanding._
