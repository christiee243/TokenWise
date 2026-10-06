"""
TokenWise - FastAPI Backend Application
Handles PDF analysis, extraction, markdown conversion, chunking, and prompt optimization.
Exposes endpoints for the Next.js frontend and Member 2/4 integration.
M2 addition: RAG retrieval endpoints for document indexing and semantic search.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_processor import process_pdf
from prompt_optimizer import optimize_prompt
from rag import RAGRetriever

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tokenwise")

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TokenWise API",
    description="Token-optimization backend for PDF analysis, prompt refinement, and RAG retrieval",
    version="1.1.0",
)

# Allow Next.js frontend (local dev or production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton RAG Retriever ──────────────────────────────────────────────────
# Shared instance so the embedding model is loaded once and indexes persist
# across requests for the lifetime of the server process.
rag_retriever = RAGRetriever()


# ── Request / Response Models ────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str


class ChunkInput(BaseModel):
    chunk_id: int = 0
    text: str
    section: str = "General"
    tokens: int = 0


class IndexDocumentRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    chunks: List[ChunkInput] = Field(..., description="Chunks from M1's process_pdf")


class QueryRequest(BaseModel):
    document_id: str = Field(..., description="Which indexed document to search")
    query: str = Field(..., description="Natural language question")
    top_k: Optional[int] = Field(None, description="Override number of results")


# ═════════════════════════════════════════════════════════════════════════════
# Existing M1 Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend service availability."""
    indexed_docs = rag_retriever.list_documents()
    return {
        "status": "ok",
        "service": "TokenWise Backend",
        "version": "1.1.0",
        "indexed_documents": len(indexed_docs),
    }


@app.post("/api/analyze-pdf")
async def analyze_pdf_endpoint(
    file: UploadFile = File(...),
    compression: str = Form("Balanced"),
):
    """
    Accepts a PDF file upload and compression level ('Light', 'Balanced', 'Maximum').
    Extracts text, strips repeating headers/footers, extracts tables,
    converts to Markdown, chunks data, and returns real token metrics.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF exceeds maximum allowed size of 30 MB.")

    try:
        result = process_pdf(content, compression_level=compression)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/optimize-prompt")
def optimize_prompt_endpoint(req: PromptRequest):
    """
    Refactors user prompt for higher clarity and reduced token count.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        result = optimize_prompt(req.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing prompt: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# M2 RAG Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/index-document")
def index_document_endpoint(req: IndexDocumentRequest):
    """
    Index document chunks into the RAG vector store for semantic search.
    Called after PDF analysis with the chunks returned by process_pdf.

    The chunks are embedded using sentence-transformers and stored in a
    FAISS index keyed by document_id. Re-indexing the same document_id
    replaces the previous index.
    """
    if not req.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided for indexing.")

    if not req.document_id or not req.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required.")

    # Convert Pydantic models to dicts for the retriever
    chunk_dicts: List[Dict[str, Any]] = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "section": c.section,
            "tokens": c.tokens,
        }
        for c in req.chunks
    ]

    try:
        result = rag_retriever.index_document(
            chunks=chunk_dicts,
            document_id=req.document_id.strip(),
        )
        return result
    except Exception as e:
        logger.exception("Error indexing document '%s'", req.document_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing document: {str(e)}",
        )


@app.post("/api/query")
def query_endpoint(req: QueryRequest):
    """
    Retrieve the most relevant document chunks for a natural language query.

    Performs query analysis (intent, complexity, keyword extraction), then
    runs semantic similarity search against the indexed document's FAISS store.
    Returns ranked chunks with relevance scores and metadata.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not req.document_id or not req.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required.")

    try:
        result = rag_retriever.retrieve(
            query=req.query.strip(),
            document_id=req.document_id.strip(),
            top_k=req.top_k,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing query on document '%s'", req.document_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@app.get("/api/documents")
def list_documents_endpoint():
    """List all currently indexed documents and their chunk counts."""
    return {
        "documents": rag_retriever.list_documents(),
        "total": len(rag_retriever.list_documents()),
    }


@app.delete("/api/documents/{document_id}")
def delete_document_endpoint(document_id: str):
    """Remove a document's index from the vector store."""
    if rag_retriever.remove_document(document_id):
        return {"status": "removed", "document_id": document_id}
    raise HTTPException(
        status_code=404,
        detail=f"Document '{document_id}' not found.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

