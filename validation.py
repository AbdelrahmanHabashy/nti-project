import json


# ==========================================
# 1. Validate one record
# ==========================================

def validate_record(record, index):
    errors = []

    required_fields = [
        "law_name",
        "page_number",
        "content"
    ]

    # Check required fields
    for field in required_fields:

        if field not in record:
            errors.append({
                "record_index": index,
                "field": field,
                "error_code": "MISSING_FIELD",
                "message": f"Field '{field}' is missing."
            })

    # If a required field is missing,
    # don't continue checking this record
    if errors:
        return errors

    # ======================================
    # Check law_name
    # ======================================

    if not isinstance(record["law_name"], str):

        errors.append({
            "record_index": index,
            "field": "law_name",
            "error_code": "INVALID_DATA_TYPE",
            "message": "law_name must be a string."
        })

    elif not record["law_name"].strip():

        errors.append({
            "record_index": index,
            "field": "law_name",
            "error_code": "EMPTY_FIELD",
            "message": "law_name cannot be empty."
        })

    # ======================================
    # Check page_number
    # ======================================

    if not isinstance(record["page_number"], int):

        errors.append({
            "record_index": index,
            "field": "page_number",
            "error_code": "INVALID_DATA_TYPE",
            "message": "page_number must be an integer."
        })

    elif record["page_number"] <= 0:

        errors.append({
            "record_index": index,
            "field": "page_number",
            "error_code": "INVALID_PAGE_NUMBER",
            "message": "page_number must be greater than 0."
        })

    # ======================================
    # Check content
    # ======================================

    if not isinstance(record["content"], str):

        errors.append({
            "record_index": index,
            "field": "content",
            "error_code": "INVALID_DATA_TYPE",
            "message": "content must be a string."
        })

    elif not record["content"].strip():

        errors.append({
            "record_index": index,
            "field": "content",
            "error_code": "EMPTY_FIELD",
            "message": "content cannot be empty."
        })

    return errors


# ==========================================
# 2. Validate the whole dataset
# ==========================================

def validate_dataset(records):

    errors = []

    # Used to detect duplicate pages
    seen_pages = set()

    # Check every record
    for index, record in enumerate(records):

        # Validate individual record
        record_errors = validate_record(
            record,
            index
        )

        errors.extend(record_errors)

        # ==================================
        # Check duplicate pages
        # ==================================

        if (
            isinstance(record, dict)
            and isinstance(record.get("law_name"), str)
            and isinstance(record.get("page_number"), int)
        ):

            page_key = (
                record["law_name"],
                record["page_number"]
            )

            if page_key in seen_pages:

                errors.append({
                    "record_index": index,
                    "field": "page_number",
                    "error_code": "DUPLICATE_PAGE",
                    "message": (
                        f"Duplicate page "
                        f"{record['page_number']} "
                        f"for law "
                        f"'{record['law_name']}'."
                    )
                })

            else:

                seen_pages.add(page_key)

    # ==========================================
    # Calculate summary
    # ==========================================

    total_records = len(records)

    failed_records = {
        error["record_index"]
        for error in errors
    }

    failed_count = len(failed_records)

    passed_count = total_records - failed_count

    # ==========================================
    # Final validation result
    # ==========================================

    result = {

        "status": "FAILED" if errors else "PASSED",

        "is_valid": len(errors) == 0,

        "total_records_checked": total_records,

        "summary": {
            "passed_count": passed_count,
            "failed_count": failed_count
        },

        "validation_errors": errors
    }

    return result


# ==========================================
# 3. Load JSON file
# ==========================================

def load_json_file(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ==========================================
# 4. Main Program
# ==========================================

if __name__ == "__main__":

    file_name = "laws.json"

    try:

        # Load JSON data
        records = load_json_file(file_name)

        print(
            f"Loaded {len(records)} records."
        )

        print(
            "Running validation...\n"
        )

        # Run validation
        result = validate_dataset(records)

        # ==================================
        # Print result
        # ==================================

        print("Validation Result")
        print("=================")

        print(
            f"Status: {result['status']}"
        )

        print(
            f"Is Valid: {result['is_valid']}"
        )

        print(
            f"Total Records: "
            f"{result['total_records_checked']}"
        )

        print(
            f"Passed: "
            f"{result['summary']['passed_count']}"
        )

        print(
            f"Failed: "
            f"{result['summary']['failed_count']}"
        )

        # ==================================
        # Print errors
        # ==================================

        if result["validation_errors"]:

            print("\nValidation Errors:")

            for error in result["validation_errors"]:

                print(
                    f"- Record "
                    f"{error['record_index']}: "
                    f"{error['message']}"
                )

        else:

            print(
                "\nNo validation errors found."
            )

    # ======================================
    # File doesn't exist
    # ======================================

    except FileNotFoundError:

        print(
            f"Error: File "
            f"'{file_name}' "
            f"was not found."
        )

        print(
            "Make sure laws.json "
            "is in the same folder "
            "as validation.py."
        )

    # ======================================
    # Invalid JSON
    # ======================================

    except json.JSONDecodeError:

        print(
            "Error: The JSON file "
            "is not valid."
        )

    # ======================================
    # Any other error
    # ======================================

    except Exception as e:

        print(
            f"Unexpected error: {e}"
        )