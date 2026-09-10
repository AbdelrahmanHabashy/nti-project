from ocr import extract_text
from invoice_extractor import extract_invoice_data


def process_invoice(image_path):

    ocr_text = extract_text(image_path)

    invoice_data = extract_invoice_data(ocr_text)

    return invoice_data

# =========================
# Test
# =========================

if __name__ == "__main__":

    image_path = r"C:\Users\zeyad-mohamed\Desktop\Zeyad-Mohamed\Project\Data\Images\Valid_INV-2026-005.jpeg"

    invoice = process_invoice(image_path)

    print("\n================ INVOICE DATA ================\n")

    for key, value in invoice.items():
        print(f"{key}: {value}")