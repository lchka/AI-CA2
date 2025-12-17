# STEP 1 — USER INPUT
# STEP 2 — BRAND VALIDATION
from services.ingredientfetcher import fetch_ingredients
from rag_builder.rag_retriever import retrieve_research
from services.openai_service import analyse_with_llm


def brand_present(product_name: str) -> bool:
    known_brands = ["pantene", "cerave", "la roche-posay", "nivea"]
    return any(b in product_name.lower() for b in known_brands)


def run_analysis(product_name, allergies, conditions):
    if not brand_present(product_name):
        return {
            "error": "Brand missing",
            "message": "Please specify a product brand."
        }

    # STEP 3 — INGREDIENT RETRIEVAL
    ingredients = fetch_ingredients(product_name)

    # STEP 4 — RAG RETRIEVAL
    research_snippets = retrieve_research(product_name)

    # STEP 5–6 — LLM ANALYSIS
    return analyse_with_llm(
        ingredients=ingredients,
        allergies=allergies,
        conditions=conditions,
        research=research_snippets
    )


if __name__ == "__main__":
    result = run_analysis(
        product_name="Pantene Anti-Dandruff Shampoo",
        allergies=["fragrance"],
        conditions=["eczema"]
    )

    print(result)
