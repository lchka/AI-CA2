# STEP 4 — RETRIEVAL
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the .env file in this directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "skincare-rag"))


def retrieve_research(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant research snippets from Pinecone based on the query.
    """
    # Create embedding for the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_embedding = response.data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Format results
    context = ""
    for match in results.matches:
        context += f"Source: {match.metadata.get('source', 'Unknown')}\n"
        context += f"Text: {match.metadata.get('text', '')}\n"
        context += f"Score: {match.score}\n\n"

    return context


if __name__ == "__main__":
    query = "Sodium Lauryl Sulfate eczema safety"
    results = retrieve_research(query)

    print("---- RETRIEVED CONTEXT ----")
    print(results[:1000])
