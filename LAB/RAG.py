from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator
from datetime import datetime
import os
import json
import re

ALLOWED_TOY_TERMS = [
    "搖鈴", "吊飾玩具", "BABY GYM NEST", "音樂盒", "不倒翁玩具", "文字學習", "平衡玩具",
    "手套玩偶", "樂器鼓玩具", "木琴玩具", "牽繩玩具", "推走玩具", "數字學習", "軟布製小屋",
    "軟性積木", "配對拼圖", "立方體拼圖", "堆疊玩具", "形狀配對玩具", "鑰匙開鎖玩具",
    "釘子玩具", "鋼琴玩具", "滑珠玩具", "組合軌道玩具", "拉丁字母學習", "方塊拼湊玩具",
    "料理玩具", "斜坡玩具", "槌子玩具", "簡單積木", "排列玩具", "磁鐵組合玩具", "記憶玩具",
    "嬰兒安撫蛋", "嬰兒健力架", "嬰兒音樂健力架", "嬰兒抓握環", "蟲蟲鏡", "不倒翁馬", "彩色嬰兒積木",
    "木製搖鈴", "音樂盒車", "彩色木鼓", "嬰兒音樂木琴", "彩色學習盒", "木製嬰兒積木", "益智火箭",
    "彩虹珠", "動物拼圖", "貓貓巴士", "蔬菜切切樂", "彩色積木", "漢字拼牌", "黑白棋釣手", "甜甜圈疊疊樂"
]

def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for idx, item in enumerate(data):
        content = item.get("translated_text") or item.get("text")
        if content:
            documents.append({"text": content})
    return documents

def extract_all(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    extracted = []
    for item in data:
        question = item.get("question", "").strip()
        output = item.get("output", "").strip()
        text = f"{question}: {output}"
        extracted.append({"translated_text": text})
    return extracted

def vectorize(documents, save_path="faiss_index"):
    if isinstance(documents[0], dict):
        documents = [
            Document(
                page_content=doc.get("translated_text") or doc.get("text", ""),
                metadata={"source": doc.get("url", "")}
            ) for doc in documents
        ]
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def rewrite_query(query):
    llm = Ollama(model="mistral", temperature=0.1)
    prompt = f"""
    請將以下使用者提問改寫為更清楚、具體且適合客服系統檢索的問題，避免模糊不清或口語化，保留原意並聚焦於核心需求。
    原始問題：
    「{query}」
    """
    rewritten = llm.invoke(prompt)
    return rewritten.strip()

def generate_answer(context, query,retriever,memory_chunk): #用mistral產生回答
    llm = Ollama(model="mistral", temperature=0.1, top_p=0.8)
    allowed_toys = ", ".join(ALLOWED_TOY_TERMS)
    system_prompt = f"""
    你是 TOYSUB 的專業客服助理，TOYSUB 是一個提供兒童教育玩具訂閱的服務平台。
    你必須嚴格僅根據提供的參考內容與對話記錄來回答問題，並遵守以下規則：
    
    1. 回覆請使用繁體中文，語氣簡潔自然、專業，限制為 6 句話以內。
    2. **不要重複任何語句**（例如「這些玩具可以...」只出現一次）。
    3. 結尾僅使用一次「如需更多協助，請讓我們知道。」之類句子。

    請根據參考內容與使用者提問回答問題
    若使用者提問與玩具推薦或玩具類別相關問題:
    請從下列玩具清單中，挑選出 **最符合提問中兒童年齡與需求** 的項目，並簡單說明原因。
    請不要列出所有玩具，也不要胡亂搭配，僅列出 3–5 項 **最合適** 的玩具名稱。
    若找不到符合條件者，請回答「目前無法提供相關資訊」。
    - 可用玩具名稱列表（只能從中選擇）：
    {allowed_toys}  

    【參考內容】：
    {{context}}

    【對話記錄】：
    {{memory_chunk}}

    【使用者提問】：
    {{query}}

    【你的回覆】：
    請以「回覆：」作為回答開頭，然後開始內容。
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "問題: {{input}}"),
        ]
    ) #提示模板
    document_chain = create_stuff_documents_chain(llm, prompt_template) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain)   #手動組成檢索
    result = retrieval_chain.invoke({"input": query,"query": query, "context": context,"memory_chunk": memory_chunk}) #用檢索invoke產生回答
    answer = result["answer"]  
    return answer

def rerank(translated_query, matched_chunks):
    reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").to("cpu")
    reranker_model.eval()
    pairs = [[translated_query, get_page_content(chunk)] for chunk in matched_chunks]
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cpu")
    with torch.no_grad():
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).cpu().numpy()
    reranked = sorted(zip(matched_chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:10]]

def get_page_content(chunk):
    if isinstance(chunk, dict):
        return chunk.get("page_content", chunk.get("text", ""))
    elif hasattr(chunk, "page_content"):
        return chunk.page_content
    else:
        return str(chunk)

def clean_answer(text):
    lines = re.split("[。！？]", text)
    seen = set()
    filtered = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            filtered.append(line)
    result = "。".join(filtered).strip("。") + "。"
    sentences = result.split("。")
    return "。".join(sentences[:6]) + "。" if len(sentences) > 6 else result

def init_memory_vectorstore(path="./memory_vectorstore"):
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
    if os.path.exists(path) and os.path.exists(f"{path}/index.faiss"):
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True), embedding_model
    else:
        return None, embedding_model

# 初始化全域資源
memory_index_path = "./memory_vectorstore"
memory_vectorstore, embedding_model = init_memory_vectorstore(memory_index_path)
manual_documents = extract_all("datasets/toysub_data.json")
scratch_documents_tw = load_json_file("datasets/toysub_chinese_data.json")
scratch_documents_jap = load_json_file("datasets/toysub_jap_data.json")
combined_documents = manual_documents + scratch_documents_tw + scratch_documents_jap
vectorstore = vectorize(combined_documents, save_path="faiss_index")

def get_rag_answer(query):
    global memory_vectorstore, embedding_model, vectorstore, memory_index_path
    refined_query = rewrite_query(query)
    memory_chunk = ""
    if memory_vectorstore:
        memory_results = memory_vectorstore.similarity_search(refined_query, k=3)
        memory_chunk = "\n".join([doc.page_content for doc in memory_results])
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(refined_query)
    reranked_chunks = rerank(refined_query, relevant_chunks)
    context = "\n".join(chunk.page_content for chunk in reranked_chunks)
    answer_result = generate_answer(context, refined_query, retriever, memory_chunk)
    final_ans = clean_answer(answer_result)
    memory_text = f"使用者問：{query}\n模型答：{final_ans}"
    memory_doc = Document(page_content=memory_text, metadata={"type": "memory", "timestamp": str(datetime.now()), "user_id": "yu"})
    if memory_vectorstore is None:
        memory_vectorstore = FAISS.from_documents([memory_doc], embedding_model)
    else:
        memory_vectorstore.add_documents([memory_doc])
    memory_vectorstore.save_local(memory_index_path)
    return final_ans

def main():
    while True:
        query = input("輸入問題:\n")
        if query.lower() == "exit":
            break
        ans = get_rag_answer(query)
        print(ans)

if __name__ == "__main__":
    main()
