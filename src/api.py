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

class QuestionRequest(BaseModel):
    question: str

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
    chunks = search(request.question)

    if not chunks:
        return {
            "question": request.question,
            "answer": "I don't have information about that.",
            "sources": []
        }

    context = "\n\n".join(chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a Zomato customer support agent.
Answer using ONLY the information below.
If the answer is not in the information, say 'I don't have that information.'

INFORMATION:
{context}"""},
            {"role": "user", "content": request.question}
        ]
    )

    return {
        "question": request.question,
        "answer": response.choices[0].message.content,
        "sources": chunks
    }