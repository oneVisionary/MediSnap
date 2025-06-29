import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import google.generativeai as genai

# === Gemini API Setup ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyDLF6v7tHsvP2Jej8OKn1MuJlLMVSip5PM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# === FastAPI App ===
app = FastAPI(title="MediSnap OCR API", description="AI-powered prescription reader")

# === OCR Extraction Function ===
def extract_with_gemini(image: Image.Image) -> str:
    response = model.generate_content([
        "Read and extract all medicine names, dosages, and diagnosis from this handwritten medical prescription. Output in a clear bullet list.",
        image
    ])
    return response.text

# === FastAPI Route ===
@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    try:
        # ✅ Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # ✅ Extract using Gemini
        result_text = extract_with_gemini(image)

        # ✅ Return structured response
        return JSONResponse(content={"status": "success", "extracted_data": result_text})
    
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
