#!/Users/basyal/Desktop/projects/PythonProject/.venv/bin/python3
from ollama import Client
import json
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma")
COUNTER_FILE = os.path.join(SCRIPT_DIR, "counter.txt")
DATA_FILE = os.path.join(SCRIPT_DIR, "article.jsonl")

client = chromadb.PersistentClient(path=CHROMA_PATH)
remote_client = Client(host=f"http://localhost:11434")

# 🔥 Clear old data on startup to ensure we have the latest news
try:
    client.delete_collection(name="articles_demo")
    print("🧹  Cleared old database records...")
except:
    pass

collection = client.get_or_create_collection(name="articles_demo")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, separators=["."])

# Reset counter because we cleared the DB
with open(COUNTER_FILE, "w") as f:
    f.write("0")

try:
    with open(COUNTER_FILE, "r") as f:
        count = int(f.read().strip())
except FileNotFoundError:
    count = 0

print("🕯️  AI Assistant: Initialization")
print(f"📖  Reading article.jsonl and generating embeddings (from #{count})...")

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        for i, line in enumerate(f):
            if i < count:
                continue
            count += 1
            article = json.loads(line)
            content = article["content"]
            sentences = text_splitter.split_text(content)
            
            print(f"  🧠 Indexing: {article['title'][:40]}...")
            for idx, each_sentence in enumerate(sentences):
                response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {each_sentence}")
                embedding = response["embeddings"][0]

                collection.add(
                    ids=[f"article_{i}_sentence_{idx}"],
                    embeddings=[embedding],
                    documents=[content], # 💡 Store the FULL article content here!
                    metadatas=[{"title": article["title"]}],
                )

    print("✅  Database built successfully!")
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))
else:
    print(f"⚠️  Data file not found at {DATA_FILE}")

print("\n🤖  AI CHATBOT")
print("┈" * 30)

while True:
    query = input("🤔  How can I help you? (type 'break' to quit) \n>> ")
    if query == "break":
        print("👋  Goodbye!")
        break

    if not query.strip():
        continue

    print("🔍  Searching...")
    query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
    # 📈 Get the single most relevant article (it's already the full text!)
    results = collection.query(query_embeddings=[query_embed], n_results=1)

    if results["documents"] and results["documents"][0]:
        context = results["documents"][0][0] # The first result's full document

        prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer.
        Answer strictly using the context.
        If the topic is unrelated, say "I don't know".

        Context: {context}

        Question: {query}

        Answer:"""

        print("🧠  Thinking...")
        response = remote_client.generate(
                model="llama3.2:3b",
                prompt=prompt,
                options={
                    "temperature": 0.1
                }
            )

        print(f"\n🤖  {response['response']}\n")
    else:
        print("\n🤖  I don't know. No relevant articles found.\n")
    
    print("┈" * 30)
