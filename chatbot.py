import os
import sys
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from onnx_embedder import get_embedder  # import ONNX embedder bạn đã viết

# Đảm bảo UTF-8 encoding cho console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Load API key từ .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY không được tìm thấy trong .env file!")
    sys.exit(1)

genai.configure(api_key=api_key)
llm = genai.GenerativeModel("gemini-2.5-flash-lite")
print("✅ Khởi tạo Gemini thành công")

# Custom embedding function (dùng ONNXEmbedder)
class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path="./models"):
        self.embedder = get_embedder(model_path)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.embedder(input)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


# Khởi tạo ChromaDB client
try:
    client = chromadb.PersistentClient(path="./vector_db")
    embedding_fn = LocalEmbeddingFunction("./models")
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn
    )
    print(f"✅ Kết nối ChromaDB thành công. Collection: {collection.name}")
except Exception as e:
    print(f"❌ Lỗi kết nối ChromaDB: {e}")
    print("💡 Hãy chạy db_setup.py hoặc ingest.py trước")
    sys.exit(1)

def chat(user_query: str, max_context_length=10000):
    if not user_query or not user_query.strip():
        return "Vui lòng nhập câu hỏi."
    
    try:
        print("🔍 Đang tìm kiếm thông tin liên quan...")
        results = collection.query(
            query_texts=[user_query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        context = ""
        sources = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            for i, doc in enumerate(docs):
                if doc and doc.strip():
                    context += doc + "\n\n"
                    if (results.get("metadatas") and 
                        results["metadatas"][0] and 
                        len(results["metadatas"][0]) > i and
                        results["metadatas"][0][i]):
                        source = results["metadatas"][0][i].get("filename", "Unknown")
                        if source not in sources:
                            sources.append(source)
            
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            if sources:
                context += f"\n\n📚 Nguồn: {', '.join(sources)}"
        
        if not context.strip():
            context = "Không tìm thấy dữ liệu liên quan trong cơ sở dữ liệu."

        prompt = f"""
Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.

THÔNG TIN TÌM ĐƯỢC:
{context}

CÂU HỎI NGƯỜI DÙNG: {user_query}

HƯỚNG DẪN TRẢ LỜI:
1. Trả lời bằng tiếng Việt hoặc tiếng Anh tùy thuộc vào ngôn ngữ câu hỏi
2. Dựa chủ yếu vào thông tin được cung cấp
3. Nếu không có thông tin liên quan, hãy thành thật nói rằng không tìm thấy
4. Trả lời ngắn gọn, rõ ràng và hữu ích
"""

        print("🤖 Đang tạo phản hồi...")
        response = llm.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Xin lỗi, tôi không thể tạo phản hồi lúc này. Vui lòng thử lại."
    
    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        return f"Đã xảy ra lỗi: {str(e)}. Vui lòng thử lại."

def main():
    print("=" * 50)
    print("🤖 CHATBOT THÔNG MINH")
    print("💡 Nhập 'exit', 'quit' hoặc 'thoat' để kết thúc")
    print("💡 Nhập 'help' để xem hướng dẫn")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "thoat", "bye"]:
                print("👋 Tạm biệt! Hẹn gặp lại!")
                break
            
            if user_input.lower() in ["help", "giup", "hướng dẫn"]:
                print("""
📋 HƯỚNG DẪN SỬ DỤNG:
• Đặt câu hỏi về nội dung trong dữ liệu đã được ingest
• Hỗ trợ tiếng Việt và tiếng Anh
• Nhập 'exit' để thoát
• Nhập 'help' để xem hướng dẫn này
                """)
                continue
            
            response = chat(user_input)
            print(f"\n🤖 Chatbot: {response}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chatbot. Tạm biệt!")
            break
        except EOFError:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi không mong muốn: {e}")
            print("🔄 Vui lòng thử lại...")

if __name__ == "__main__":
    main()
