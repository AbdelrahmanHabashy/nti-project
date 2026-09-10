# Calculate Agent

AI-based calculation agent for invoice calculations.

## Requirements

* Python 3.10+
* Ollama
* Qwen 2.5 7B

## Installation

### 1. Install Python dependency

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download and install Ollama from:

https://ollama.com/

### 3. Download the Qwen model

Open a terminal and run:

```bash
ollama pull qwen2.5:7b
```

### 4. Start Ollama

Make sure Ollama is running before using the agent.

### Usage

```python
from calculate_agent import CalculateAgent

agent = CalculateAgent(model="qwen2.5:7b")

result = agent.run(
    "احسب ضريبة القيمة المضافة على مبلغ 9500 جنيه بنسبة 14%"
)

print(result)
```

Example output:

```python
{
    "success": True,
    "tool": "calculate_vat",
    "arguments": {
        "taxable_amount": 9500.0,
        "vat_rate": 0.14
    },
    "result": 1330.0
}
```
