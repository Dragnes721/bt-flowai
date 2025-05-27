import os
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1) Ayarlar
DATA_DIR = "ajans_chroma_data"
CHROMA_DIR = "ajans_chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# 2) Embedding modeli
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 3) Dosyaları oku, Document oluştur
docs = []
for fname in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, fname)
    if not fname.endswith(".txt"):
        continue
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    # Basit metadata: dosya adından tip çıkarabilirsiniz
    metadata = {"source": fname}
    docs.append(Document(page_content=text, metadata=metadata))

# 4) Chroma vektör deposuna yükle
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
vectordb.persist()
print(f"✅ {len(docs)} belge ChromaDB’ye yüklendi.")
