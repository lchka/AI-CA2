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
PDF_FOLDER = "pdfs"
CHUNK_SIZE = 800           # approx tokens
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
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
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
    all_vectors = []
    counter = 0

    for pdf_name in os.listdir(PDF_FOLDER):
        if not pdf_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        print(f"📄 Extracting: {pdf_name}")

        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)

        print(f"➡️  {len(chunks)} chunks created")

        for chunk in chunks:
            emb = embed_text(chunk)

            vector = {
                "id": f"doc-{counter}",
                "values": emb,
                "metadata": {
                    "source": pdf_name,
                    "text": chunk
                }
            }

            all_vectors.append(vector)
            counter += 1

        # Upsert in batches of 100
        if len(all_vectors) > 100:
            print("⬆️  Uploading batch...")
            index.upsert(all_vectors)
            all_vectors = []

    # Upload remaining vectors
    if len(all_vectors) > 0:
        print("⬆️  Uploading final batch...")
        index.upsert(all_vectors)

    print("🎉 DONE – All PDFs added to Pinecone!")


if __name__ == "__main__":
    process_pdfs()
