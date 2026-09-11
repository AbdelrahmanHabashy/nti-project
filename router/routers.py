import ollama

from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from Abdo.calculate_agent import CalculateAgent
from marwaAHassan.ntiproject import get_legal_explanation
from zeyad.agent import graph_agent, df


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
        default="",
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

Use this when the user asks for information that should
be retrieved from the tax/legal knowledge base, including:

- Egyptian tax laws
- Egyptian VAT laws and regulations
- VAT rates defined by law
- E-invoicing rules and requirements
- Legal requirements
- Tax compliance rules
- Tax regulations
- Explanations based on tax documents
- Questions asking what the law says
- Questions asking about legally defined tax rates
- Questions where the required information must be retrieved
  from the knowledge base

Also use rag_agent when the user asks for a calculation
but the numerical information required to perform the
calculation is missing and the question depends on a
tax law or regulation.

Examples:

"ما هي نسبة ضريبة القيمة المضافة في مصر؟"
→ rag_agent

"ما هي نسبة VAT حسب القانون المصري؟"
→ rag_agent

"اشرح قانون ضريبة القيمة المضافة المصري"
→ rag_agent

"احسب الضريبة حسب قانون ضريبة القيمة المضافة المصري"
→ rag_agent


2. 'calculation_agent'

Use this when the user asks for an actual numerical
calculation AND provides the numerical values needed
for that calculation.

Use this for:

- VAT calculations
- Invoice totals
- Discounts
- Net amounts
- Checking calculated amounts
- Financial calculations
- Arithmetic related to invoices

Examples:

"احسب ضريبة القيمة المضافة على 10000 جنيه بنسبة 14%"
→ calculation_agent

"كم قيمة الضريبة على مبلغ 20000 جنيه بنسبة 14%؟"
→ calculation_agent

"احسب صافي المبلغ بعد خصم 1000 جنيه من 15000 جنيه"
→ calculation_agent

"احسب إجمالي فاتورة قيمتها 10000 جنيه مع ضريبة 14%"
→ calculation_agent


3. 'graphical_agent'

Use this when the user asks for:

- A chart
- A graph
- A plot
- A visualization
- Sales or revenue trends
- VAT trends
- Invoice statistics that require visualization
- Any data visualization


IMPORTANT ROUTING RULES:

1. Do NOT choose calculation_agent only because the
   query contains words such as "calculate", "احسب",
   "كام", or "قيمة".

2. Choose calculation_agent ONLY when the user provides
   the numerical inputs required for an actual calculation.

3. If the user asks to calculate something but the required
   numerical input is missing, do NOT invent or assume a value.

4. If a query asks about Egyptian tax law, VAT regulations,
   legal VAT rates, or information that must come from the
   knowledge base, choose rag_agent.

5. If a query contains both a tax-law question and a numerical
   calculation, choose rag_agent when the calculation depends
   on obtaining legal or regulatory information first.

6. Do NOT infer a VAT rate from the user's wording unless
   the VAT rate is explicitly provided.

7. Do NOT perform calculations yourself.

8. Do NOT answer the user's question.

9. Do NOT extract values for another agent.

10. ONLY decide which agent should handle the query.

11. Return exactly one intent.

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

calculate_agent_instance = CalculateAgent(model="qwen2.5:7b")


def rag_agent(query: str, image_path: str = None):

    return get_legal_explanation(query)


def calculation_agent(prompt: str):

    return calculate_agent_instance.run(prompt)


def graphical_agent(prompt: str):

    return graph_agent(
        user_query=prompt,
        df=df
    )


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