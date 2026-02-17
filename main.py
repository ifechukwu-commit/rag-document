"""
Multi-Industry RAG Document Q&A System
Handles PDF uploads, vector search, and industry-filtered question answering
"""

import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

# --- RAG & AI Stack Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- CONFIGURATION SETUP ---
# Load environment variables from .env file (API keys, etc.)
load_dotenv()

# --- LOGGING SETUP ---
# Configure logging to track system events, errors, and debugging info
logging.basicConfig(
    level=logging.INFO,  # Log level: INFO, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'  # Timestamp + level + message
)
logger = logging.getLogger(__name__)

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Multi-Industry RAG API")

# --- CONSTANTS ---
FAISS_INDEX_PATH = "faiss_index"  # Where we save the vector database
ALLOWED_INDUSTRIES = ["healthcare", "real-estate", "hvac", "legal", "finance"]  # Valid industry tags

# =============================================================================
# ENDPOINT 1: DOCUMENT INGESTION
# Purpose: Upload PDFs and store them in FAISS with industry tags
# =============================================================================

@app.post("/ingest")
async def ingest_document(industry: str = Form(...), file: UploadFile = File(...)):
    """
    Uploads a PDF, breaks it into chunks, converts to embeddings, 
    and stores in FAISS database with industry metadata.
    
    Args:
        industry: Industry tag (e.g., 'healthcare', 'real-estate')
        file: PDF file to upload
        
    Returns:
        Success message with filename and industry
    """
    
    # --- INPUT VALIDATION ---
    # Check if industry tag is valid
    if industry.lower() not in ALLOWED_INDUSTRIES:
        logger.error(f"Invalid industry: {industry}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid industry. Allowed: {ALLOWED_INDUSTRIES}"
        )
    
    # Check if file is actually a PDF
    if not file.filename.endswith('.pdf'):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    logger.info(f"Starting ingestion: {file.filename} for {industry}")
    
    # --- TEMPORARY FILE HANDLING ---
    # Save uploaded file temporarily so we can process it
    temp_path = f"temp_{file.filename}"
    
    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved temporarily: {temp_path}")
        
        # --- PDF LOADING ---
        # Extract text from PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
        except Exception as e:
            logger.error(f"PDF loading failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to read PDF file")
        
        # --- METADATA TAGGING ---
        # Add industry tag to each document chunk
        for doc in documents:
            doc.metadata["industry"] = industry.lower()
        logger.info(f"Tagged all documents with industry: {industry}")
        
        # --- EMBEDDINGS SETUP ---
        # Initialize the model that converts text to numbers
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Embeddings model failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load embeddings model")
        
        # --- VECTOR DATABASE OPERATIONS ---
        # Either add to existing database or create new one
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                # Database exists - load it and add new documents
                logger.info("Loading existing FAISS index")
                vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                vector_store.add_documents(documents)
                logger.info("Added documents to existing index")
            else:
                # No database - create new one from documents
                logger.info("Creating new FAISS index")
                vector_store = FAISS.from_documents(documents, embeddings)
                logger.info("New FAISS index created")
            
            # Save database to disk
            vector_store.save_local(FAISS_INDEX_PATH)
            logger.info("FAISS index saved to disk")
            
        except Exception as e:
            logger.error(f"FAISS operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to store documents in database")
        
        # --- CLEANUP ---
        # Delete temporary file
        os.remove(temp_path)
        logger.info(f"Temporary file removed: {temp_path}")
        
        return {
            "message": f"Successfully added {file.filename} to {industry} storage",
            "pages": len(documents),
            "industry": industry
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (these are expected errors)
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during ingestion: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# =============================================================================
# ENDPOINT 2: QUESTION ANSWERING
# Purpose: Search documents and generate answers filtered by industry
# =============================================================================

@app.post("/ask")
async def ask_agent(question: str = Form(...), industry: str = Form(...)):
    """
    Searches FAISS for relevant documents filtered by industry,
    then uses AI to generate an answer.
    
    Args:
        question: The question to answer
        industry: Industry to filter documents by
        
    Returns:
        AI-generated answer and the industry used
    """
    
    # --- INPUT VALIDATION ---
    # Check if industry is valid
    if industry.lower() not in ALLOWED_INDUSTRIES:
        logger.error(f"Invalid industry in query: {industry}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid industry. Allowed: {ALLOWED_INDUSTRIES}"
        )
    
    # Check if question is not empty
    if not question.strip():
        logger.error("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check if question is too long (prevent abuse)
    if len(question) > 500:
        logger.error("Question too long")
        raise HTTPException(status_code=400, detail="Question must be under 500 characters")
    
    logger.info(f"Processing question for {industry}: {question[:50]}...")
    
    # --- DATABASE CHECK ---
    # Make sure FAISS database exists
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error("FAISS index not found")
        raise HTTPException(
            status_code=404, 
            detail="No documents in database. Please upload documents first using /ingest"
        )
    
    try:
        # --- EMBEDDINGS SETUP ---
        # Load the same embeddings model used during ingestion
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embeddings model loaded for query")
        except Exception as e:
            logger.error(f"Embeddings model failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load embeddings model")
        
        # --- LOAD VECTOR DATABASE ---
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"FAISS load failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load document database")
        
        # --- SIMILARITY SEARCH ---
        # Find top 3 most relevant document chunks filtered by industry
        try:
            relevant_docs = vector_store.similarity_search(
                question, 
                k=3,  # Return top 3 results
                filter={"industry": industry.lower()}  # Only search this industry
            )
            logger.info(f"Found {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Search failed")
        
        # Check if any documents were found
        if not relevant_docs:
            logger.warning(f"No documents found for industry: {industry}")
            return {
                "answer": f"No documents found for {industry}. Please upload documents first.",
                "industry_used": industry,
                "sources_found": 0
            }
        
        # --- CONTEXT BUILDING ---
        # Combine retrieved document chunks into one context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"Built context from {len(relevant_docs)} documents")
        
        # --- LLM SETUP ---
        # Initialize Groq AI model
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            logger.info("LLM initialized")
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail="AI model unavailable")
        
        # --- PROMPT CONSTRUCTION ---
        # Create the instruction for the AI
        prompt = f"""You are a specialist in {industry}. 
        Use ONLY the context below to answer the question accurately.
        If the answer is not in the context, say "I don't have enough information to answer that."
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        # --- AI ANSWER GENERATION ---
        # Send prompt to AI and get response
        try:
            response = llm.invoke(prompt)
            logger.info("AI response generated successfully")
        except Exception as e:
            logger.error(f"LLM invocation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        return {
            "answer": response.content,
            "industry_used": industry,
            "sources_found": len(relevant_docs)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# =============================================================================
# HEALTH CHECK ENDPOINT
# Purpose: Simple endpoint to verify the API is running
# =============================================================================

@app.get("/")
async def root():
    """
    Health check endpoint - returns API status
    """
    logger.info("Health check requested")
    return {
        "status": "running",
        "message": "Multi-Industry RAG API",
        "endpoints": {
            "/ingest": "Upload PDFs with industry tags",
            "/ask": "Ask questions filtered by industry"
        }
    }


# =============================================================================
# RUN SERVER
# Purpose: Start the FastAPI server when script is run directly
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)