import ollama

from typing import Literal
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# ROUTE DECISION
# ============================================================

AgentIntent = Literal[
    "calculation_agent",
    "graphical_agent",
    "rag_agent"
]


class RouteDecision(BaseModel):

    reasoning: str = Field(
        description="Short reasoning explaining why this agent was selected."
    )

    intent: AgentIntent = Field(
        description="The designated agent destination."
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )


# ============================================================
# ROUTER PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intent classification and routing engine
for an AI Invoice and Tax Compliance Assistant.

Your ONLY job is to decide which agent should handle
the user's request.

Available agents:

1. 'rag_agent'
Use this when the user asks for:
- Tax laws
- VAT regulations
- E-invoicing rules
- Legal requirements
- Tax compliance rules
- Information that must be retrieved from the knowledge base
- Explanations based on tax documents

2. 'calculation_agent'
Use this when the user asks for:
- VAT calculations
- Invoice totals
- Discounts
- Net amounts
- Checking calculated amounts
- Financial calculations
- Arithmetic related to invoices

3. 'graphical_agent'
Use this when the user asks for:
- A chart
- A graph
- A plot
- A visualization
- Sales or revenue trends
- VAT trends
- Invoice statistics that require visualization

Important rules:

- Do NOT perform calculations yourself.
- Do NOT answer the user's question.
- Do NOT calculate VAT.
- Do NOT extract values for another agent.
- ONLY decide which agent should handle the query.
- Return exactly one intent.

The intent MUST be exactly one of:

- calculation_agent
- graphical_agent
- rag_agent
"""


# ============================================================
# ROUTE QUERY
# ============================================================

def route_query(user_query: str) -> dict:

    try:

        response = ollama.chat(
            model="qwen2.5:7b",

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Analyze and route this query: {user_query}"
                }
            ],

            format="json",

            options={
                "temperature": 0.0
            }
        )

        raw_content = response.message.content

        parsed = RouteDecision.model_validate_json(
            raw_content
        )

        return {
            "intent": parsed.intent,
            "query": user_query,
            "reasoning": parsed.reasoning,
            "confidence": parsed.confidence,
        }

    except ValidationError as ve:

        return {
            "intent": "rag_agent",
            "query": user_query,
            "reasoning": f"Validation Error: {ve}",
            "confidence": 0.0,
        }

    except Exception as e:

        return {
            "intent": "rag_agent",
            "query": user_query,
            "reasoning": f"Ollama Error: {str(e)}",
            "confidence": 0.0,
        }


# ============================================================
# AGENTS
# ============================================================

# These functions are temporary placeholders.
# They will be replaced with the actual agents.

def rag_agent(query: str, image_path: str = None) -> dict:

    return {
        "text": f"RAG agent received: {query}"
    }


def calculation_agent(prompt: str) -> str:

    return f"Calculation agent received: {prompt}"


def graphical_agent(prompt: str) -> str:

    return f"Graphical agent received: {prompt}"


# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrator(user_query: str, image_path: str = None):

    routing_result = route_query(user_query)

    intent = routing_result["intent"]

    print(
        f"--> Routing request to: {intent} "
        f"(Reason: {routing_result['reasoning']})"
    )

    if intent == "rag_agent":

        return rag_agent(
            query=user_query,
            image_path=image_path
        )

    elif intent == "calculation_agent":

        return calculation_agent(
            user_query
        )

    elif intent == "graphical_agent":

        return graphical_agent(
            user_query
        )

    else:

        return "No suitable agent was identified."


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [

        # ====================================================
        # CALCULATION AGENT
        # ====================================================

        "احسب ضريبة القيمة المضافة على 9500 جنيه بنسبة 14%",

        "احسب إجمالي فاتورة قيمتها 10000 جنيه مع ضريبة 14%",

        "لو السعر 5000 جنيه والخصم 500 جنيه، كام المبلغ النهائي؟",

        "احسب صافي المبلغ بعد خصم 1000 جنيه من 15000 جنيه",

        "كم قيمة الضريبة على مبلغ 20000 جنيه بنسبة 14%؟",

        "الفاتورة قيمتها 12000 جنيه والضريبة 1680 جنيه، هل الحساب صحيح؟",

        "احسب الإجمالي لو المبلغ قبل الضريبة 8000 والضريبة 14%",

        "لو عندي فاتورة بـ 15000 جنيه وخصم 2000 جنيه، احسب الضريبة والإجمالي",

        "هل 1400 جنيه ضريبة صحيحة على مبلغ 10000 جنيه بنسبة 14%؟",

        "احسب قيمة VAT لفاتورة قيمتها 25000 جنيه بنسبة 14%",


        # ====================================================
        # RAG AGENT
        # ====================================================

        "ما هي متطلبات الفاتورة الإلكترونية في مصر؟",

        "ما هي قوانين ضريبة القيمة المضافة في مصر؟",

        "ما هي نسبة ضريبة القيمة المضافة في مصر؟",

        "ما هي البيانات الإلزامية التي يجب أن تحتوي عليها الفاتورة؟",

        "هل الفاتورة الإلكترونية إلزامية للشركات؟",

        "ما هي شروط التسجيل في منظومة الفاتورة الإلكترونية؟",

        "اشرح لي قانون ضريبة القيمة المضافة",

        "ما هي متطلبات إصدار فاتورة إلكترونية صحيحة؟",

        "هل يوجد حد معين للتسجيل في ضريبة القيمة المضافة؟",

        "ما هي القواعد الخاصة بالفواتير الضريبية في مصر؟",


        # ====================================================
        # GRAPHICAL AGENT
        # ====================================================

        "اعمل رسم بياني لإجمالي المبيعات لكل منتج",

        "اعمل chart للمبيعات الشهرية",

        "أريد رسم بياني يوضح الإيرادات خلال السنة",

        "اعمل visualization لعدد الفواتير لكل شهر",

        "اعرض اتجاه المبيعات في رسم بياني",

        "ارسم graph يوضح قيمة الضريبة لكل شهر",

        "اعمل plot للمبيعات حسب المنتج",

        "أريد رسم بياني يقارن إجمالي الفواتير بين الشهور",

        "اعمل chart يوضح توزيع الفواتير حسب المنتج",

        "اعرض بيانات المبيعات في رسم بياني",

        
        # ====================================================
        # BORDERLINE / MIXED
        # ====================================================

        "ما هي نسبة VAT وهل يمكنك حسابها على فاتورة بقيمة 10000؟",

        "اشرح لي ضريبة القيمة المضافة واحسبها على 5000 جنيه",

        "اعمل رسم بياني يوضح نسبة ضريبة القيمة المضافة خلال السنة",

        "هل الفاتورة دي متوافقة مع قانون الفاتورة الإلكترونية؟",

        "احسب الضريبة حسب قانون ضريبة القيمة المضافة المصري",

        "ما هي متطلبات الفاتورة الإلكترونية واحسب لي الإجمالي؟"
    ]


    for i, query in enumerate(test_queries, 1):

        print("\n" + "=" * 70)

        print(f"TEST {i}")

        print("QUERY:")
        print(query)

        result = route_query(query)

        print("\nROUTING RESULT:")
        print(result)