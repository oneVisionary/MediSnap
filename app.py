import os
import io
import re
import json
import traceback
import requests
from typing import Dict, Any
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# === Load API Keys from Environment ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_API_KEY = os.getenv("CSE_API_KEY")
CSE_ID = os.getenv("CSE_ID")

# === Validate API Keys ===
if not all([GOOGLE_API_KEY, CSE_API_KEY, CSE_ID]):
    raise RuntimeError("Missing one or more required API keys: GOOGLE_API_KEY, CSE_API_KEY, CSE_ID")

# === Configure Gemini ===
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Initialize FastAPI App ===
app = FastAPI(title="MediSnap OCR API", description="AI-powered prescription reader")

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Step 1: Gemini Extraction ===
def extract_data_from_image(image: Image.Image) -> Dict[str, Any]:
    prompt = """
    Read the handwritten prescription and extract:
    - Medicine names
    - Dosages (e.g., 0-1-0, or 500mg 2x daily)
    - Taking time (e.g., morning, afternoon, night)
    - Diagnoses or reasons (if mentioned)
    - Benefits of using these drugs
    - Do's and Don'ts
    - Possible conditions

    Return strictly in JSON format:
    {
      "drugs": [
        {"name": "DrugA", "dosage": "0-1-0", "taking_time": "morning"},
        {"name": "DrugB", "dosage": "500mg 2x daily", "taking_time": "night"}
      ],
      "diagnosis": "...",
      "benefits": "...",
      "dos_donts": "...",
      "possible_conditions": "..."
    }
    """
    response = gemini_model.generate_content([prompt, image])
    text = response.text.strip()
    print("[Gemini Raw Response]\n", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse Gemini response as JSON")

# === Step 2: Validate Capsule Image ===
def is_valid_capsule_image(image_url: str) -> bool:
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(io.BytesIO(response.content))

        prompt = "Is this image a medicine capsule, tablet, or drug package? Answer only 'yes' or 'no'."
        check = gemini_model.generate_content([prompt, img])
        verdict = check.text.strip().lower()
        return "yes" in verdict
    except Exception as e:
        print(f"[Validation Error] {e}")
        return False

# === Step 3: Fetch Drug Image URL ===
def get_drug_image_url(drug_name: str) -> str:
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": f"{drug_name} medicine capsule",
            "cx": CSE_ID,
            "key": CSE_API_KEY,
            "searchType": "image",
            "num": 1
        }
        response = requests.get(search_url, params=params, timeout=10)
        image_url = response.json()["items"][0]["link"]

        if is_valid_capsule_image(image_url):
            return image_url
        else:
            print(f"[Image Rejected] Not a capsule: {drug_name}")
            return None
    except Exception as e:
        print(f"[Image Fetch Error] {drug_name}: {e}")
        return None

# === Step 4: FastAPI Endpoint ===
@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    try:
        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Step 1: Extract drug info
        extracted_data = extract_data_from_image(image)

        # Step 2: Add capsule image URLs if valid
        for drug in extracted_data.get("drugs", []):
            name = drug.get("name", "")
            drug["image_url"] = get_drug_image_url(name) or "Not found"

        # Ensure all expected keys exist
        for key in ["diagnosis", "benefits", "dos_donts", "possible_conditions"]:
            if key not in extracted_data:
                extracted_data[key] = None

        return JSONResponse(content={"status": "success", "data": extracted_data})

    except Exception as e:
        stack = traceback.format_exc()
        print("[Server Error]\n", stack)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to extract data from prescription",
                "stack": stack
            }
        )
