# STEP 5 — PROMPT CONSTRUCTION
# STEP 6 — LLM EXECUTION
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the rag_builder directory
env_path = Path(__file__).parent.parent / 'rag_builder' / '.env'
load_dotenv(dotenv_path=env_path)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a skincare safety expert.
Return ONLY valid JSON.
No markdown or extra text.
"""),

    ("human",
     """Ingredients:
{ingredients}

User allergies:
{allergies}

User conditions:
{conditions}

Research snippets:
{research}

Return exactly this JSON schema:
{{
  "summary": "",
  "suitability_score": 0,
  "is_suitable": false,
  "recommendation": "",
  "explanation": "",
  "allergy_notes": "",
  "condition_notes": "",
  "key_ingredients": []
}}
""")
])


def analyse_with_llm(ingredients, allergies, conditions, research):
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "ingredients": ingredients,
        "allergies": allergies,
        "conditions": conditions,
        "research": research
    })

    return json.loads(response)
