from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from dotenv import load_dotenv
from backend.pdf_utils import extract_text_from_pdf, validate_pdf_file
from backend.vector_store import VectorStore, chunk_text
from backend.embedder import get_embedding
from backend.chat_engine import get_answer
from fastapi.staticfiles import StaticFiles




# Load environment variables
load_dotenv()

app = FastAPI(title="RAG PDF Chat API", description="Upload PDFs and chat with their content")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store instance
vector_store = VectorStore()

class QueryRequest(BaseModel):
    question: str
    k: int = 5  # Number of relevant chunks to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: int

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file, add to vector store"""
    print(f"📥 Received file: {file.filename}")
    print(f"📋 Content type: {file.content_type}")
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create backend directory if it doesn't exist
    backend_dir = "backend"
    os.makedirs(backend_dir, exist_ok=True)
    
    temp_pdf_path = os.path.abspath(os.path.join(backend_dir, "temp.pdf"))

    try:
        # Save uploaded file
        await file.seek(0)
        with open(temp_pdf_path, "wb") as buffer:
            content = await file.read()
            if len(content) == 0:
                raise Exception("Uploaded file is empty")
            buffer.write(content)

        # Validate PDF
        is_valid, validation_msg = validate_pdf_file(temp_pdf_path)
        if not is_valid:
            raise Exception(f"Invalid PDF file: {validation_msg}")

        # Extract text
        print("🔄 Extracting text from PDF...")
        full_text = extract_text_from_pdf(temp_pdf_path)
        
        # Chunk the text
        print("✂️ Chunking text...")
        chunks = chunk_text(full_text, chunk_size=800, overlap=100)
        print(f"📄 Created {len(chunks)} text chunks")
        
        # Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings = []
        metadata = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                metadata.append({
                    "chunk_id": i,
                    "source": file.filename,
                    "chunk_length": len(chunk)
                })
                
                if (i + 1) % 10 == 0:  # Progress update every 10 chunks
                    print(f"   📊 Generated {i + 1}/{len(chunks)} embeddings")
            except Exception as e:
                print(f"⚠️ Error generating embedding for chunk {i}: {e}")
                continue
        
        if not embeddings:
            raise Exception("Failed to generate any embeddings")
        
        # Add to vector store
        print("💾 Adding to vector store...")
        vector_store.add(chunks, embeddings, metadata)
        
        # Clean up temp file
        try:
            os.remove(temp_pdf_path)
            print("🗑️ Temporary file cleaned up")
        except:
            pass
        
        # Get store statistics
        stats = vector_store.get_stats()
        
        return {
            "success": True,
            "message": "PDF processed and added to vector store",
            "file_info": {
                "filename": file.filename,
                "total_characters": len(full_text),
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings)
            },
            "vector_store_stats": stats
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        # Clean up on error
        try:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Chat with the documents using RAG"""
    
    if vector_store.get_stats()["total_documents"] == 0:
        raise HTTPException(
            status_code=400, 
            detail="No documents in vector store. Please upload a PDF first."
        )
    
    try:
        print(f"💬 Processing question: {request.question}")
        
        # Generate query embedding
        print("🧠 Generating query embedding...")
        query_embedding = get_embedding(request.question)
        
        # Search vector store
        print(f"🔍 Searching for {request.k} relevant chunks...")
        search_results = vector_store.search(query_embedding, k=request.k)
        
        if not search_results:
            return ChatResponse(
                answer="I couldn't find any relevant information in the uploaded documents.",
                sources=[],
                retrieved_chunks=0
            )
        
        # Extract chunks and prepare sources
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
        
        print(f"📚 Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Generate answer using LLM
        print("🤖 Generating answer...")
        answer = get_answer(relevant_chunks, request.question)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=len(relevant_chunks)
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat Error: {e}")

@app.get("/")
async def root():
    return {
        "message": "RAG PDF Chat API is running",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status/")
async def get_status():
    """Get system status and vector store info"""
    stats = vector_store.get_stats()
    return {
        "status": "running",
        "vector_store": stats,
        "ready_for_chat": stats["total_documents"] > 0
    }

@app.post("/clear/")
async def clear_vector_store():
    """Clear all documents from vector store"""
    vector_store.clear()
    return {"message": "Vector store cleared successfully"}

@app.get("/health/")
async def health_check():
    """Simple health check endpoint"""
    try:
        # Test OpenAI connection with a small embedding
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

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print("🚀 Starting RAG PDF Chat API...")
    print(f"📍 Server will run at: http://{host}:{port}")
    print("📚 Upload PDFs with POST /upload/")
    print("💬 Chat with documents using POST /chat/")
    print("📊 Check status with GET /status/")
    print("📖 API docs available at: /docs")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        reload=debug
    )