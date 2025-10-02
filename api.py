import os
import sys
import logging
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional
from onnx_embedder import get_embedder   # import file onnx_embedder.py của bạn

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# UTF-8 cho console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ===== Gemini Init =====
try:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-2.5-flash-lite")
    logger.info("✅ Gemini initialized")
except Exception as e:
    logger.error(f"❌ Gemini init failed: {e}")
    sys.exit(1)

# ===== ONNX Embedder =====
try:
    logger.info("🔄 Loading ONNX embedder...")
    embedder = get_embedder("./models")
    logger.info("✅ ONNX embedder ready")
except Exception as e:
    logger.error(f"❌ ONNX embedder failed: {e}")
    sys.exit(1)

# Custom embedding function cho ChromaDB
class ONNXEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path="./models"):
        self.embedder = get_embedder(model_path)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.embedder(input)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

# ===== ChromaDB Init =====
try:
    client = chromadb.PersistentClient(
        path="./vector_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    embedding_fn = ONNXEmbeddingFunction(model_path="./models")
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={"embedding_function": "onnx_custom"}
    )
    logger.info(f"✅ ChromaDB ready. Docs: {collection.count()}")
except Exception as e:
    logger.error(f"❌ ChromaDB failed: {e}")
    sys.exit(1)

# ===== FastAPI App =====
app = FastAPI(
    title="Chatbot API",
    description="Vietnamese-English Chatbot with ONNX RAG",
    version="2.0.0"
)

@app.middleware("http")
async def ensure_utf8(request, call_next):
    response = await call_next(request)
    if hasattr(response, "headers"):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

# ===== Schemas =====
class Query(BaseModel):
    user_id: str
    message: str
    max_results: Optional[int] = 5

    @validator("message")
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

    @validator("user_id")
    def user_id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    reply: str
    sources: Optional[list] = []
    user_id: str
    success: bool = True

# ===== Exception Handler =====
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "reply": "Đã xảy ra lỗi. Vui lòng thử lại.",
            "success": False,
            "error": str(exc)
        }
    )

# ===== Endpoints =====
@app.get("/")
async def root():
    return {
        "message": "🤖 Chatbot API (ONNX) is running!",
        "version": "2.0.0",
        "engine": "ONNX Runtime (lightweight)",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }

@app.get("/health")
async def health():
    try:
        collection.count()
        test_response = llm.generate_content("test")
        return {
            "status": "healthy",
            "chromadb": "connected",
            "gemini": "connected",
            "embedder": "onnx",
            "documents": collection.count()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        logger.info(f"Chat from {query.user_id}: {query.message[:100]}...")

        # Search trong Chroma
        results = collection.query(
            query_texts=[query.message],
            n_results=min(query.max_results, 30),  # Tăng lên 15
            include=["documents", "metadatas", "distances"]  # Thêm distances để filter
        )

        context = ""
        sources = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0]  # Thêm dòng này
            
            for i, doc in enumerate(docs):
                # Sửa logic: distance < 0.7 mới là relevant (không phải > 0.7)
                for i, doc in enumerate(docs):
                    if distances and len(distances) > i:
                        print(f"📊 Doc {i}: distance={distances[i]:.4f}")  # Debug xem distance thực tế
                    
                    if doc and doc.strip():
                        context += doc + "\n\n"
                        # ... rest
                    if (results.get("metadatas") and 
                        results["metadatas"][0] and 
                        len(results["metadatas"][0]) > i):
                        source = results["metadatas"][0][i].get("source", "Unknown")
                        if source not in sources:
                            sources.append(source)

            if len(context) > 12000:
                context = context[:12000] + "..."

        if not context.strip():
            context = "Không tìm thấy dữ liệu liên quan."

        prompt = f"""
Bạn là trợ lý AI thông minh và thân thiện. Trả lời câu hỏi dựa trên thông tin. Thông tin mà bạn được cung cấp có thể bao gồm các trích dẫn từ các nguồn khác nhau.
Nguồn thông tin của bạn là đến từ các ticket customer support và các tài liệu kỹ thuật về sản phẩm của công ty.
Có thể thông tin này không đầy đủ hoặc không chính xác và cũng có thể có rất nhiều thông tin bị thừa thãi bên trong cơ sở dữ liệu (như là email, tên người gửi, địa chỉ, etc...). Hãy tự cân nhắc.
Nếu bạn không chắc chắn về câu trả lời, hãy nói thẳng rằng bạn không biết thay vì đoán mò.
(You are a smart and friendly AI assistant. Answer questions based on provided information. The information you are given may include excerpts from various sources.
Your information sources come from customer support tickets and technical documents about the company's products.)

THÔNG TIN (Information):
{context}

NGƯỜI DÙNG (User) ({query.user_id}): {query.message}

HƯỚNG DẪN:
1. Về ngôn ngữ (Language): Trả lời lại cùng ngôn ngữ với người dùng. (Match user's language, Vietnamese, English, Japanese, Chinese, etc.)
2. Dựa vào thông tin có sẵn (Based on available info)
3. Nếu không có thông tin, nói thẳng (If no info, say so directly)
4. Ngắn gọn, rõ ràng, hữu ích (Concise, clear, helpful)
5. UTF-8 encoding đúng (Proper UTF-8 encoding)

TRẢ LỜI (Answer):
"""

        response = llm.generate_content(prompt)
        if not response or not response.text:
            raise Exception("Empty Gemini response")

        reply_text = response.text.strip()
        reply_text = reply_text.encode("utf-8").decode("utf-8")

        logger.info(f"✅ Response generated for {query.user_id}")

        return {
            "reply": reply_text,
            "sources": sources,
            "user_id": query.user_id,
            "success": True
        }

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    try:
        count = collection.count()
        return {
            "total_documents": count,
            "collection_name": collection.name,
            "engine": "ONNX Runtime",
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Run with Uvicorn =====
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
