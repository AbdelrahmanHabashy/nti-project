from openai import OpenAI
import json
import pandas as pd
import os


# =========================
# Load invoices
# =========================

with open(
    r"C:\Users\zeyad-mohamed\Desktop\Zeyad-Mohamed\Project\invoices.json",
    "r",
    encoding="utf-8"
) as f:
    invoices = json.load(f)


# =========================
# Prepare DataFrame
# =========================

df = pd.DataFrame(invoices)

df["date"] = pd.to_datetime(df["date"])

df["month"] = df["date"].dt.month_name()


# =========================
# OpenRouter
# =========================

client = OpenAI(
    api_key="API-HERE",
    base_url="https://openrouter.ai/api/v1"
)


# =========================
# Graph Agent
# =========================

def graph_agent(user_query, df):

    prompt = f"""
You are a Graph Agent.

The available dataframe columns are:

{list(df.columns)}

The user wants:

{user_query}

Your job is to decide the best chart configuration.

Return ONLY a valid JSON object with exactly these fields:

{{
    "chart_type": "bar",
    "x_column": "product",
    "y_column": "total",
    "title": "Sales by Product"
}}

Allowed chart types:
- bar
- line
- pie

Available columns:
- invoice_id
- date
- supplier_name
- tax_id
- product
- subtotal
- vat_amount
- total
- month

Rules:

1. Use "product" when the user asks to compare products.

2. Use "total" when the user asks about:
   - revenue
   - sales
   - invoice totals
   - total sales

3. Use "vat_amount" when the user asks about VAT.

4. Use "subtotal" when the user asks about:
   - revenue before VAT
   - subtotal
   - amount before tax

5. Use "date" or "month" for time-based charts.

6. Use "pie" when the user explicitly asks for:
   - pie chart
   - distribution
   - share

7. Use "line" for trends over time.

8. Use "bar" for comparisons between categories.

9. x_column must be an existing dataframe column.

10. y_column must be an existing dataframe column.

11. Do not invent column names.

12. Return JSON only.

13. Do not include markdown.

14. Do not include explanations.

User request:
{user_query}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "The model returned an empty response."
        )

    print("\nRAW MODEL RESPONSE:")
    print(result)

    try:
        spec = json.loads(result)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"The model did not return valid JSON.\n\n"
            f"Model response:\n{result}"
        ) from e

    # =========================
    # Validate chart specification
    # =========================

    required_fields = [
        "chart_type",
        "x_column",
        "y_column",
        "title"
    ]

    for field in required_fields:

        if field not in spec:
            raise ValueError(
                f"Missing field in chart specification: {field}"
            )

    if spec["chart_type"] not in ["bar", "line", "pie"]:
        raise ValueError(
            f"Invalid chart type: {spec['chart_type']}"
        )

    if spec["x_column"] not in df.columns:
        raise ValueError(
            f"Invalid x_column: {spec['x_column']}"
        )

    if spec["y_column"] not in df.columns:
        raise ValueError(
            f"Invalid y_column: {spec['y_column']}"
        )

    return spec