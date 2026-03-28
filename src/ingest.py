from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

# Our clients
client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="support_docs")

# Read the document
with open("data/policy.txt", "r") as f:
    text = f.read()

# Split into chunks (by paragraph for now)
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# Convert each chunk into an embedding and store it
for i, chunk in enumerate(chunks):
    response = client.embeddings.create(model="text-embedding-3-small", input=chunk)
    embedding = response.data[0].embedding

    collection.add(documents=[chunk], embeddings=[embedding], ids=[f"chunk_{i}"])
    print(f"Stored chunk {i}: {chunk[:50]}...")

print("\nAll chunks stored successfully!")