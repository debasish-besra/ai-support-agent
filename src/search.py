from dotenv import load_dotenv
import chromadb
from openai import OpenAI
import cohere
import os

load_dotenv()

client = OpenAI()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
chroma = chromadb.PersistentClient(path="../chroma_db")
collection = chroma.get_or_create_collection(name="support_docs")

def search(query, n_retrieve=3, n_final=2, threshold=0.3):
    # Stage 1 — vector search
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

    # Stage 2 — reranker
    reranked = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=candidates,
        top_n=n_final
    )

    # Only keep chunks above threshold
    filtered = [
        candidates[r.index] 
        for r in reranked.results 
        if r.relevance_score > threshold
    ]

    return filtered

# Test
if __name__ == "__main__":
    question = "Can I get a refund if my order was cancelled?"
    chunks = search(question)
    print(f"Chunks returned: {len(chunks)}")
    for chunk in chunks:
        print(f"\n{chunk[:80]}...")