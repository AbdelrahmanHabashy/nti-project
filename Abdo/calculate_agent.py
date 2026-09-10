import ollama

from calculation import (
    calculate_net_amount,
    calculate_vat,
    calculate_total,
    validate_amount,
    calculate_invoice,
    validate_invoice,
)


class CalculateAgent:

    def __init__(self, model="qwen2.5:7b"):
        self.model = model

        # ============================================================
        # AVAILABLE PYTHON TOOLS
        # ============================================================

        self.tools = {
            "calculate_net_amount": calculate_net_amount,
            "calculate_vat": calculate_vat,
            "calculate_total": calculate_total,
            "validate_amount": validate_amount,
            "calculate_invoice": calculate_invoice,
            "validate_invoice": validate_invoice,
        }

        # ============================================================
        # OLLAMA TOOL DEFINITIONS
        # ============================================================

        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_net_amount",
                    "description": (
                        "Calculate the amount remaining after subtracting a discount "
                        "from a subtotal. Use ONLY when the user asks for the amount "
                        "after discount."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtotal": {
                                "type": "number",
                                "description": "Original amount before discount."
                            },
                            "discount": {
                                "type": "number",
                                "description": "Discount amount."
                            }
                        },
                        "required": ["subtotal", "discount"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "calculate_vat",
                    "description": (
                        "Calculate VAT/tax on a taxable amount. "
                        "Use when the user explicitly asks to calculate VAT or tax. "
                        "vat_rate must be a decimal, for example 14% = 0.14."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "taxable_amount": {
                                "type": "number",
                                "description": "Amount on which VAT is calculated."
                            },
                            "vat_rate": {
                                "type": "number",
                                "description": (
                                    "VAT rate as a decimal. "
                                    "Examples: 14% = 0.14, 5% = 0.05."
                                )
                            }
                        },
                        "required": ["taxable_amount"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "calculate_total",
                    "description": (
                        "Calculate the invoice total by adding an explicitly provided "
                        "net amount and VAT amount. Use only when both values are "
                        "already known."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "net_amount": {
                                "type": "number",
                                "description": "Net amount before VAT."
                            },
                            "vat_amount": {
                                "type": "number",
                                "description": "VAT amount."
                            }
                        },
                        "required": ["net_amount", "vat_amount"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "validate_amount",
                    "description": (
                        "Compare an already calculated amount with an amount provided "
                        "by the user. This tool does NOT calculate the expected amount. "
                        "Use only when the calculated/reference amount is already known."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "calculated": {
                                "type": "number",
                                "description": "Already known calculated/reference amount."
                            },
                            "provided": {
                                "type": "number",
                                "description": "Amount provided on the invoice."
                            },
                            "tolerance": {
                                "type": "number",
                                "description": (
                                    "Optional allowed difference. "
                                    "Only include when explicitly provided by the user."
                                )
                            }
                        },
                        "required": ["calculated", "provided"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "calculate_invoice",
                    "description": (
                        "Calculate a complete invoice: subtotal, discount, net amount, "
                        "VAT, and total. Use when the user explicitly asks to calculate "
                        "the complete invoice."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtotal": {
                                "type": "number",
                                "description": "Original amount before discount."
                            },
                            "discount": {
                                "type": "number",
                                "description": "Discount amount."
                            },
                            "vat_rate": {
                                "type": "number",
                                "description": (
                                    "Optional VAT rate as a decimal. "
                                    "14% = 0.14. If omitted, Python uses its default."
                                )
                            }
                        },
                        "required": ["subtotal"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "validate_invoice",
                    "description": (
                        "Validate a complete invoice by calculating expected VAT and "
                        "total and comparing them with the VAT and total provided on "
                        "the invoice. Requires subtotal, discount, VAT rate, provided "
                        "VAT, and provided total. Never infer the VAT rate from other values."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtotal": {
                                "type": "number",
                                "description": "Original amount before discount."
                            },
                            "discount": {
                                "type": "number",
                                "description": "Discount amount."
                            },
                            "vat_rate": {
                                "type": "number",
                                "description": "VAT rate as a decimal."
                            },
                            "provided_vat": {
                                "type": "number",
                                "description": "VAT amount written on the invoice."
                            },
                            "provided_total": {
                                "type": "number",
                                "description": "Total amount written on the invoice."
                            },
                            "tolerance": {
                                "type": "number",
                                "description": (
                                    "Optional allowed difference. "
                                    "Only include when explicitly provided."
                                )
                            }
                        },
                        "required": [
                            "subtotal",
                            "discount",
                            "vat_rate",
                            "provided_vat",
                            "provided_total"
                        ]
                    }
                }
            }
        ]

    # ============================================================
    # SYSTEM PROMPT
    # ============================================================

    def _system_prompt(self):
        return """
You are the Calculate Agent in an invoice analysis system.

Your ONLY job is to:
1. Understand the user's calculation or validation request.
2. Extract numerical values explicitly stated by the user.
3. Select the correct Python calculation tool.
4. Pass the extracted values to the tool.

Python tools perform ALL financial arithmetic.

============================================================
STRICT RULES
============================================================

1. NEVER perform arithmetic yourself.

Do not calculate:
- VAT
- totals
- discounts
- net amounts
- differences
- percentages
- validation results

Always let the Python tool perform the calculation.

------------------------------------------------------------

2. NEVER INVENT NUMERICAL VALUES.

Only use numbers explicitly provided by the user.

Do NOT infer a missing value from other values.

For example:

User:
"subtotal = 9500 and VAT = 1330. Is the invoice valid?"

Do NOT calculate:
1330 / 9500 = 14%

Do NOT invent vat_rate = 0.14.

If a required value is missing, do not create one.

------------------------------------------------------------

3. DO NOT ADD OPTIONAL ARGUMENTS THAT THE USER DID NOT PROVIDE.

If an optional parameter is not mentioned, OMIT it.

For example, if the user says:

"احسب ضريبة 14% على 10000"

Use:

{
    "taxable_amount": 10000,
    "vat_rate": 0.14
}

Do NOT add:
"tolerance": 0.0

Python will use its own default value when an optional argument is omitted.

------------------------------------------------------------

4. PERCENTAGE CONVERSION

Convert explicitly stated percentages into decimals.

14% = 0.14
5% = 0.05
7.5% = 0.075
10% = 0.10
15% = 0.15

Do not invent a percentage that the user did not provide.

------------------------------------------------------------

5. USE ONLY ONE TOOL CALL.

Select the single most appropriate tool for the user's request.

Do not try to perform multiple tool calls.

If the request requires multiple calculation steps that cannot be
performed by one available tool, do not invent intermediate values.

============================================================
TOOL SELECTION
============================================================

1. calculate_vat

Use when the user asks to calculate VAT/tax for a known taxable amount.

Examples:

"احسب ضريبة 14% على 10000"

"Calculate 5% VAT on 20000"

Use:
calculate_vat

Arguments:
{
    "taxable_amount": 10000,
    "vat_rate": 0.14
}

If the user says "after a discount" and provides both the original
amount and discount, but only asks for the VAT, use calculate_vat
with the resulting taxable amount ONLY if that taxable amount is
explicitly stated by the user.

Do not calculate the discounted amount yourself.

------------------------------------------------------------

2. calculate_net_amount

Use when the user asks for the amount after a discount.

Example:

"احسب صافي 15000 بعد خصم 1000"

Use:
calculate_net_amount

Arguments:
{
    "subtotal": 15000,
    "discount": 1000
}

------------------------------------------------------------

3. calculate_total

Use when the user explicitly provides BOTH:
- net amount
- VAT amount

and asks for the total.

Example:

"صافي المبلغ 10000 والضريبة 1400، احسب الإجمالي"

Use:
calculate_total

Arguments:
{
    "net_amount": 10000,
    "vat_amount": 1400
}

------------------------------------------------------------

4. calculate_invoice

Use when the user asks for the COMPLETE invoice calculation:

subtotal
→ discount
→ net amount
→ VAT
→ total

Example:

"احسب الفاتورة: 20000 قبل الخصم، خصم 2000، وضريبة 14%"

Use:
calculate_invoice

Arguments:
{
    "subtotal": 20000,
    "discount": 2000,
    "vat_rate": 0.14
}

If VAT rate is not explicitly provided, the Python function's default
VAT rate may be used.

------------------------------------------------------------

5. validate_amount

Use when the user wants to compare an existing amount against an
ALREADY KNOWN calculated amount.

This tool does NOT calculate the expected amount.

Example:

"القيمة المحسوبة للضريبة 1400، والفاتورة فيها 1400. هل متساويين؟"

Use:
validate_amount

Arguments:
{
    "calculated": 1400,
    "provided": 1400
}

If tolerance is explicitly provided, include it.

Otherwise OMIT tolerance.

------------------------------------------------------------

6. validate_invoice

Use when the user wants to validate a COMPLETE invoice AND provides
all required values:

- subtotal
- discount
- VAT rate
- provided VAT
- provided total

Example:

"الفاتورة: 10000، خصم 500، ضريبة 14%، الضريبة المكتوبة 1330،
والإجمالي المكتوب 10830. هل صحيحة؟"

Use:
validate_invoice

Arguments:
{
    "subtotal": 10000,
    "discount": 500,
    "vat_rate": 0.14,
    "provided_vat": 1330,
    "provided_total": 10830
}

Do NOT infer the VAT rate from provided VAT.

If VAT rate is missing, do not invent it.

============================================================
MISSING INFORMATION
============================================================

If a required value is missing, do not invent or derive it.

If no appropriate tool can be called safely, do not fabricate a tool call.

============================================================
FINAL RULE
============================================================

The LLM understands the request and selects the tool.

Python performs the mathematics.

Never replace Python calculations with your own arithmetic.
"""

    # ============================================================
    # PARSE USER REQUEST
    # ============================================================

    def parse_request(self, prompt: str):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            tools=self.tool_definitions
        )

        if not response.message.tool_calls:
            raise ValueError(
                "The model did not select a calculation tool."
            )

        tool_call = response.message.tool_calls[0]

        return {
            "tool": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }

    # ============================================================
    # EXECUTE PYTHON TOOL
    # ============================================================

    def execute_tool(self, tool_name: str, arguments: dict):

        if tool_name not in self.tools:
            raise ValueError(
                f"Unknown calculation tool: {tool_name}"
            )

        # Normalize numeric arguments.
        numeric_fields = {
            "subtotal",
            "discount",
            "taxable_amount",
            "vat_rate",
            "net_amount",
            "vat_amount",
            "calculated",
            "provided",
            "tolerance",
            "provided_vat",
            "provided_total"
        }

        for key in arguments:
            if key in numeric_fields:
                arguments[key] = float(arguments[key])

        tool = self.tools[tool_name]

        return tool(**arguments)

    # ============================================================
    # RUN AGENT
    # ============================================================

    def run(self, prompt: str):

        request = self.parse_request(prompt)

        tool_name = request["tool"]
        arguments = request["arguments"]

        result = self.execute_tool(
            tool_name=tool_name,
            arguments=arguments
        )

        return {
            "success": True,
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        }