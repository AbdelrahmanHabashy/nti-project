from validation import validate_dataset


test_records = [
    {
        "law_name": "vat-law no.67-2016",
        "page_number": 1,
        "content": "Valid content"
    },
    {
        "law_name": "vat-law no.67-2016",
        "page_number": "two",
        "content": "Invalid page number"
    },
    {
        "law_name": "vat-law no.67-2016",
        "page_number": 3,
        "content": ""
    },
    {
        "law_name": "vat-law no.67-2016",
        "page_number": 1,
        "content": "Duplicate page"
    }
]


result = validate_dataset(test_records)

print("TEST RESULT")
print("===========")

print("Status:", result["status"])
print("Is Valid:", result["is_valid"])

print("\nErrors:")

for error in result["validation_errors"]:
    print(
        f"- Record {error['record_index']}: "
        f"{error['error_code']} - "
        f"{error['message']}"
    )