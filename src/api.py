from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
import cohere
import os

load_dotenv()

app = FastAPI()
client = OpenAI()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="support_docs")

conversations = {}

class QuestionRequest(BaseModel):
    question: str
    session_id: str

def search(query, n_retrieve=3, n_final=2, threshold=0.3):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_retrieve
    )
    candidates = results["documents"][0]

    reranked = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=candidates,
        top_n=n_final
    )

    filtered = [
        candidates[r.index]
        for r in reranked.results
        if r.relevance_score > threshold
    ]

    return filtered

@app.get("/")
def root():
    return {"status": "Zomato Support Agent is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    if request.session_id not in conversations:
        conversations[request.session_id] = [
            {"role": "system", "content": """You are a Zomato customer support agent.
Answer using ONLY the information provided to you.
If the answer is not available, say 'I don't have that information.'"""}
        ]

    chunks = search(request.question)
    context = "\n\n".join(chunks) if chunks else ""

    user_message = f"Context:\n{context}\n\nQuestion: {request.question}" if context else request.question
    conversations[request.session_id].append(
        {"role": "user", "content": user_message}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversations[request.session_id]
    )

    answer = response.choices[0].message.content

    conversations[request.session_id].append(
        {"role": "assistant", "content": answer}
    )

    return {
        "session_id": request.session_id,
        "question": request.question,
        "answer": answer,
        "sources": chunks
    }