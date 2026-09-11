import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# تحميل ملف .env من نفس مسار الملف
current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=current_dir / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "لم يتم العثور على GROQ_API_KEY! تأكد من وجوده في ملف .env بالشكل: GROQ_API_KEY=gsk_..."
    )

# استخدام كلاينت Groq المباشر
client = Groq(api_key=api_key)

AgentIntent = Literal[
    "calculation_agent",
    "graphical_agent",
    "rag_agent"
   
]

class RouteDecision(BaseModel):
    reasoning: str = Field(
        description="Reasoning analyzing whether medicine records, OCR, Tax Law, or inline numbers are involved."
    )
    intent: AgentIntent = Field(
        description="The designated agent destination."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0."
    )
    
SYSTEM_PROMPT = """You are an intent classification and routing engine for a pharmaceutical and tax compliance platform.
Classify the user query and extract entity information based on these rules:

1. 'rag_agent' (Image / OCR / Knowledge Retrieval):
   - Trigger when the user provides or references a new pill image/picture for OCR processing.
   - Trigger when the user asks for pill database lookups, tax law documents, or general retrieval.

2. 'calculation_agent' (Price / VAT / Financial Calculations):
   - Trigger when the user requests a calculation (e.g., total price, VAT calculation, discounts).
   - MANDATORY STEP FOR PILL CALCULATIONS:
     * If the query asks for a calculation regarding a pill:
       1. Identify and extract the Pill ID or pill name into 'extracted_pill_id'.
       2. Set 'requires_rag_lookup' to true so the orchestrator can first fetch the exact VAT percentage and base price from the RAG service before executing the math.
     * If all numbers (base price AND VAT percentage) are already explicitly provided in the prompt, set 'requires_rag_lookup' to false.

3. 'graphical_agent':
   - Trigger when the user explicitly requests a chart, graph, or plot with all data points provided."""

def route_query(user_query: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze and route this query: {user_query}"},
            ],
            response_format={
                "type": "json_object",
                "schema": RouteDecision.model_json_schema()
            },
            temperature=0.0
        )

        raw_content = completion.choices[0].message.content
        parsed = RouteDecision.model_validate_json(raw_content)

        return {
            "intent": parsed.intent,
            "query": user_query,
            "extracted_pill_id": parsed.extracted_pill_id,
            "requires_rag_lookup": parsed.requires_rag_lookup,
            "reasoning": parsed.reasoning,
            "confidence": parsed.confidence,
        }

    except ValidationError as ve:
        return {
            "intent": "rag_agent",
            "query": user_query,
            "extracted_pill_id": None,
            "requires_rag_lookup": False,
            "reasoning": f"Validation Error: {ve}",
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "intent": "rag_agent",
            "query": user_query,
            "extracted_pill_id": None,
            "requires_rag_lookup": False,
            "reasoning": f"API Error: {str(e)}",
            "confidence": 0.0,
        }

# from agents.rag import rag_agent
# from agents.calc import calculation_agent
# from agents.graph import graphical_agent

def rag_agent(query: str, image_path: str = None) -> dict:
    
    return {
        "text": "تم استرجاع البيانات بنجاح",
        "pill_id": "ID-8842",
        "base_price": 50.0,
        "vat_percentage": 14.0  # نسبة الضريبة
    }

def calculation_agent(prompt: str) -> str:
    
    return f"نتيجة الحساب للطلب: {prompt}"

def graphical_agent(prompt: str) -> str:
    
    return f"تم توليد الرسم البياني لـ: {prompt}"
def orchestrator(user_query: str, image_path: str = None):
    
    routing_result = route_query(user_query)
    intent = routing_result["intent"]
    pill_id = routing_result["extracted_pill_id"]
    needs_lookup = routing_result["requires_rag_lookup"]

    print(f"--> توجيه الطلب إلى: {intent} (Reason: {routing_result['reasoning']})")

  
    if intent == "rag_agent":
        
        return rag_agent(query=user_query, image_path=image_path)

    elif intent == "calculation_agent":
        
        if needs_lookup and pill_id:
            print(f"--> خطوة وسيطة: جلب بيانات الحبة ({pill_id}) من RAG...")
            
            
            rag_info = rag_agent(f"Get VAT and price for pill {pill_id}")
            vat = rag_info.get("vat_percentage", 0)
            base_price = rag_info.get("base_price", "N/A")

            
            enriched_prompt = (
                f"{user_query} | Additional Context: [Pill ID: {pill_id}, "
                f"Base Price: {base_price}, VAT Rate: {vat}%]"
            )
            
            
            return calculation_agent(enriched_prompt)
        else:
            
            return calculation_agent(user_query)

    elif intent == "graphical_agent":
        return graphical_agent(user_query)

    else:
        return "لم يتم التعرف على الوجهة المناسبة للطلب."
