from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="support_docs")

def search(query, n_results=2):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]

def ask(question):
    # Step 1: Retrieve relevant chunks
    chunks = search(question)
    context = "\n\n".join(chunks)
    
    # Step 2: Generate answer using retrieved context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a Zomato customer support agent.
Answer the customer's question using ONLY the information below.
If the answer is not in the information, say 'I don't have that information.'

INFORMATION:
{context}"""},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Test it
questions = [
    "Can I get a refund if my order was cancelled?",
    "Do you accept PayPal?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print()