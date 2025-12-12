import os
import fitz  # PyMuPDF
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "..", "pdfs")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


# -------------------------
# PDF TEXT EXTRACTION
# -------------------------
def extract_pdf_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# -------------------------
# TOKEN-AWARE CHUNKING
# -------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap

    return chunks


# -------------------------
# EMBEDDING
# -------------------------
def embed_text(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


# -------------------------
# MAIN PIPELINE
# -------------------------
def process_pdfs():
    batch = []

    for pdf_name in os.listdir(PDF_FOLDER):
        if not pdf_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        print(f"📄 Extracting: {pdf_name}")

        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk)

            vector = {
                "id": f"{pdf_name.replace('.pdf','')}_chunk_{i}",
                "values": emb,
                "metadata": {
                    "title": pdf_name.replace(".pdf", ""),
                    "source": "PDF literature",
                    "chunk_index": i,
                    "content_type": "ingredient safety research",
                    "text": chunk
                }
            }

            batch.append(vector)

            if len(batch) >= 100:
                index.upsert(batch)
                batch = []

    if batch:
        index.upsert(batch)

    print("🎉 DONE – PDFs indexed with rich metadata!")


if __name__ == "__main__":
    process_pdfs()
