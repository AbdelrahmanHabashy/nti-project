from typing import Any
import easyocr


reader = easyocr.Reader(
    ['ar'],
    gpu=False
)


def extract_text(image_path):

    results: list[Any] = reader.readtext(image_path)

    texts = []

    for result in results:
        text = result[1]
        texts.append(text)

    return "\n".join(texts)