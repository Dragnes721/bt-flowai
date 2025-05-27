import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# 1) Ayarlar
DATA_DIR = "ajans_chroma_data"
PERSIST_DIR = "ajans_chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# 2) OpenAI embedding client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# 3) Chroma client ve collection
client = chromadb.Client()
collection = client.create_collection(
    name="bt_ajans_docs",
    embedding_function=embed_fn,
    persist_directory=PERSIST_DIR
)

# 4) Dosyaları oku ve ingest et
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".txt"):
        continue
    text = open(os.path.join(DATA_DIR, fname), encoding="utf-8").read()
    collection.add(
        ids=[fname],
        documents=[text],
        metadatas=[{"source": fname}]
    )

# 5) Kalıcı kayıt
client.persist()
print(f"✅ {collection.count()} doküman ChromaDB’ye yüklendi.")
