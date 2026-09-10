from openai import OpenAI
import json
import os


# =========================
# OpenRouter
# =========================

# API_KEY = os.getenv("OPENROUTER_API_KEY")

# if not API_KEY:
#     raise ValueError("OPENROUTER_API_KEY is not set.")


client = OpenAI(
    api_key="API-HERE",
    base_url="https://openrouter.ai/api/v1"
)


# =========================
# Invoice Extraction Agent
# =========================

def extract_invoice_data(ocr_text):

    prompt = f"""
You are an Invoice Information Extraction Agent.

Your task is to extract structured invoice information from OCR text.

The OCR may contain:
- Arabic text
- English text
- Missing spaces
- Broken lines
- OCR recognition errors
- Labels separated from their values

You MUST use only information explicitly present in the OCR text.

OCR TEXT:
----------------
{ocr_text}
----------------

Return ONLY valid JSON with exactly these fields:

{{
    "invoice_id": null,
    "date": null,
    "supplier_name": null,
    "tax_id": null,
    "product": null,
    "subtotal": null,
    "vat_amount": null,
    "total": null
}}

Extraction rules:

1. invoice_id:
   Extract the invoice number only if its value is explicitly present.
   Examples:
   "رقم الفاتورة INV-2026-004"
   "Invoice No: INV-2026-004"
   If the label exists but the value is missing, return null.

2. date:
   Extract the invoice date if explicitly present.
   Do not invent the year if it is not present.
   Preserve the date information that actually appears in the OCR.

3. supplier_name:
   Extract the supplier/company name.
   For example:
   "اسم الشركة الموردة شركة التقنية المصرية"
   -> "شركة التقنية المصرية"

4. tax_id:
   Extract the supplier tax identification number only if its value
   is explicitly present.
   "الرقم الضريبي للمورد" is only a label.
   Do NOT treat the label itself as the tax ID.
   If the number is missing, return null.

5. product:
   Extract the product/item name.

6. subtotal:
   Extract the amount explicitly associated with the total before VAT.

7. vat_amount:
   Extract the amount explicitly associated with VAT.

8. total:
   Extract the final invoice amount.

9. Monetary values must be returned as numbers, not strings.

10. Do NOT calculate missing values.

11. Do NOT infer missing values from mathematical relationships.

12. Do NOT guess.

13. If a field is not explicitly available or cannot be reliably determined,
    return null.

14. Return JSON only.
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
        raise ValueError("The model returned an empty response.")

    try:
        return json.loads(result)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON returned by model:\n{result}"
        ) from e