import os
import io
import re
import json
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
    raise RuntimeError("One or more required API keys (GOOGLE_API_KEY, CSE_API_KEY, CSE_ID) are missing in environment variables.")

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

# === Gemini-based Extraction Function ===
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

    Return JSON format like this:
    {
      "drugs": [
        {"name": "DrugA", "dosage": "0-1-0", "taking_time": "morning"},
        {"name": "DrugB", "dosage": "500mg 2x daily", "taking_time": "night"}
      ],
      "diagnosis": "Diabetes, Hypertension"
    }
    """
    response = gemini_model.generate_content([prompt, image])
    text = response.text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"drugs": [], "diagnosis": ""}

# === Google CSE Image Fetcher ===
def get_drug_image_url(drug_name: str) -> str:
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": drug_name + " medicine capsule",
        "cx": CSE_ID,
        "key": CSE_API_KEY,
        "searchType": "image",
        "num": 1
    }
    try:
        response = requests.get(search_url, params=params, timeout=10)
        return response.json()["items"][0]["link"]
    except Exception as e:
        print(f"Image not found for {drug_name}: {e}")
        return None

# === FastAPI POST Endpoint ===
@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    try:
        # Read and open image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Step 1: Extract info from Gemini
        extracted_data = extract_data_from_image(image)

        # Step 2: Append drug image URLs
        for drug in extracted_data.get("drugs", []):
            name = drug.get("name", "")
            drug["image_url"] = get_drug_image_url(name) or "Not found"

        return JSONResponse(content={"status": "success", "data": extracted_data})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
