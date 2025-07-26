from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

import shutil
import os

from backend.pdf_utils import extract_text_from_pdf, validate_pdf_file
from backend.vector_store import VectorStore, chunk_text
from backend.embedder import get_embedding
from backend.chat_engine import get_answer

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="RAG PDF Chat API",
    description="Upload PDFs and chat with their content"
)

# Mount the frontend from /app
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

# Redirect root to frontend
@app.get("/")
async def root():
    return RedirectResponse(url="/app")

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory vector store
vector_store = VectorStore()

# Request & response models
class QueryRequest(BaseModel):
    question: str
    k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: int

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    backend_dir = "backend"
    os.makedirs(backend_dir, exist_ok=True)
    temp_pdf_path = os.path.abspath(os.path.join(backend_dir, "temp.pdf"))

    try:
        await file.seek(0)
        with open(temp_pdf_path, "wb") as buffer:
            content = await file.read()
            if not content:
                raise Exception("Uploaded file is empty")
            buffer.write(content)

        is_valid, msg = validate_pdf_file(temp_pdf_path)
        if not is_valid:
            raise Exception(f"Invalid PDF: {msg}")

        full_text = extract_text_from_pdf(temp_pdf_path)
        chunks = chunk_text(full_text, chunk_size=800, overlap=100)

        embeddings = []
        metadata = []
        for i, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({
                    "chunk_id": i,
                    "source": file.filename,
                    "chunk_length": len(chunk)
                })
            except Exception as e:
                print(f"Embedding failed on chunk {i}: {e}")

        if not embeddings:
            raise Exception("No embeddings generated")

        vector_store.add(chunks, embeddings, metadata)

        try:
            os.remove(temp_pdf_path)
        except:
            pass

        return {
            "success": True,
            "message": "PDF uploaded and processed",
            "file_info": {
                "filename": file.filename,
                "total_characters": len(full_text),
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings)
            },
            "vector_store_stats": vector_store.get_stats()
        }

    except Exception as e:
        try:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: QueryRequest):
    stats = vector_store.get_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded.")

    try:
        query_embedding = get_embedding(request.question)
        search_results = vector_store.search(query_embedding, k=request.k)

        if not search_results:
            return ChatResponse(
                answer="No relevant info found.",
                sources=[],
                retrieved_chunks=0
            )

        relevant_chunks = []
        sources = []

        for i, (text, score, metadata) in enumerate(search_results):
            relevant_chunks.append(text)
            sources.append({
                "chunk_id": metadata.get("chunk_id", i),
                "score": round(score, 4),
                "source": metadata.get("source", "Unknown"),
                "preview": text[:100] + "..." if len(text) > 100 else text
            })

        answer = get_answer(relevant_chunks, request.question)

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=len(relevant_chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@app.get("/status/")
async def get_status():
    return {
        "status": "running",
        "vector_store": vector_store.get_stats(),
        "ready_for_chat": vector_store.get_stats()["total_documents"] > 0
    }

@app.post("/clear/")
async def clear_vector_store():
    vector_store.clear()
    return {"message": "Vector store cleared"}

@app.get("/health/")
async def health_check():
    try:
        test_embedding = get_embedding("test")
        return {
            "status": "healthy",
            "openai_connection": "ok",
            "vector_store_documents": vector_store.get_stats()["total_documents"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "vector_store_documents": vector_store.get_stats()["total_documents"]
        }

# Entry point for local dev (Render/production won't use this)
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"

    print("🚀 Starting server...")
    uvicorn.run(app, host=host, port=port, reload=debug)
