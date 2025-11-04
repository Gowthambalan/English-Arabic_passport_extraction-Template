import ollama
import json
import re
from pdf2image import convert_from_path
from io import BytesIO
from pathlib import Path
import time


def pdf_to_images_bytes(pdf_path):
    pages = convert_from_path(pdf_path, dpi=150) 
    image_bytes_list = []
    for i, page in enumerate(pages):
        buf = BytesIO()
        page.save(buf, format="JPEG")
        image_bytes_list.append(buf.getvalue())
    return image_bytes_list


def extract_passport_from_pdf(pdf_path: str, model_name="qwen2.5vl:7b"):
    print(f"\n📄 Processing PDF: {pdf_path}")
    start_time = time.time()
    image_bytes_list = pdf_to_images_bytes(pdf_path)

    #  Updated bilingual extraction prompt
    # prompt = """
    # You are an expert OCR and document understanding model.
    # Analyze the uploaded passport or visa document.

    # Extract all visible text in both English and Arabic, and provide a structured JSON
    # where each key is a field name and value contains both English and Arabic texts.

    # Use this JSON structure exactly:

    # {
    #   "PassportNumber": {"english": "...", "arabic": "..."},
    #   "FullName": {"english": "...", "arabic": "..."},
    #   "Nationality": {"english": "...", "arabic": "..."},
    #   "DateOfBirth": {"english": "...", "arabic": "..."},
    #   "PlaceOfBirth": {"english": "...", "arabic": "..."},
    #   "Sex": {"english": "...", "arabic": "..."},
    #   "DateOfIssue": {"english": "...", "arabic": "..."},
    #   "DateOfExpiry": {"english": "...", "arabic": "..."},
    #   "PassportType": {"english": "...", "arabic": "..."},
    #   "Authority": {"english": "...", "arabic": "..."}
    # }

    # If a field is not visible, leave it empty but keep the key.
    # Output must be valid JSON only — no explanation text.
    # """
    prompt = """
    You are a senior OCR document understanding model.
    Analyze the uploaded passport or visa page image(s).

     Rules:
    - Keep both Arabic and English text exactly as visible.
    - Do NOT translate.
    - If a label (key) exists in both Arabic and English, combine them using " / ".
    - If a value exists in both Arabic and English, also combine them using " / ".
    - Output must be valid JSON only, no explanations.
    - dont skip any visible fields, include all possible fields.

     Example format:
    {
      "ID Number / رقم الهوية": "784199787632597",
      "File No / رقم الملف": "201/2023/7/663922",
      "Passport No / رقم الجواز": "VT1337002",
      "Name / الاسم": "MUHAMMAD AMIR IQBAL MUHAMMAD IQBAL / محمد امير اقبال محمد اقبال",
      "Profession / المهنة": "PARTNER / شريك",
      "Employer / صاحب العمل": "H S P INTERNATIONAL FOODSTUFF TRADING L.L.C / اتش اس بي انترناشونال لتجارة المواد الغذائية ش ذ م م",
      "Place of Issue / جهة الإصدار": "دبي",
      "Issue Date / تاريخ إصدار الإقامة": "15/11/2023",
      "Expiry Date / تاريخ إنتهاء الإقامة": "14/11/2025",
      "Country / الدولة": "UNITED ARAB EMIRATES / دولة الإمارات العربية المتحدة",
      "Type / نوع الإقامة": "RESIDENCE / إقامة"
      ...
    }

    Only output JSON..
    """
    # 🔹 Combine all pages in one request (faster)
    print(" Running model inference on all pages together...")
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": image_bytes_list  # send all pages
            }
        ],
    )

    content = response["message"]["content"]

    # Try extracting JSON safely
    try:
        match = re.search(r"(\{[\s\S]*\})", content)
        data = json.loads(match.group(1)) if match else {"raw_text": content}
    except Exception as e:
        data = {"raw_text": content, "error": str(e)}

    output = {"file": str(pdf_path), "extracted_data": data}
    print(f"\n Extraction completed in {time.time() - start_time:.2f} seconds.")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    extract_passport_from_pdf(pdf_path)
