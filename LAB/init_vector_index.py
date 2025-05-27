from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import json

def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for idx, item in enumerate(data):
        content = item.get("translated_text") or item.get("text")
        if content:
            documents.append(Document(
                page_content=content,
                metadata={"id": idx}
            ))
    return documents

if __name__ == "__main__":
    docs = load_json_file("datasets/toysub_data.json")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("faiss_index")
    print("✅ 向量資料庫 faiss_index 已成功建立！")
