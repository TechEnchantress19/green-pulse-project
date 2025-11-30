import os
import joblib
import uvicorn
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import io

# --- 1. CONFIGURATION ---
load_dotenv()
app = FastAPI(title="Green Pulse AI - High Precision API")

# Setup Gemini (for Disease & Schemes)
GENAI_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

# Load AI Artifacts
try:
    crop_model = joblib.load("models/crop_model.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    scaler = joblib.load("models/scaler.joblib") # Load the scaler!
    print("✅ High-Precision Models Loaded Successfully.")
except FileNotFoundError:
    print("❌ ERROR: Models not found. You MUST run 'train_model.py' first.")
    crop_model = None

# --- 2. DATA STRUCTURES ---
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class SchemeQuery(BaseModel):
    question: str
    user_state: str = "India"

# --- 3. API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "System Operational", "accuracy_mode": "Stacked Ensemble"}

# 🌱 CROP RECOMMENDATION (High Precision)
@app.post("/recommend_crop")
def recommend_crop(data: CropInput):
    if not crop_model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # 1. Prepare Raw Input
        raw_features = np.array([[
            data.N, data.P, data.K, 
            data.temperature, data.humidity, 
            data.ph, data.rainfall
        ]])

        # 2. Scale Input (CRITICAL STEP)
        # We must scale the input using the SAME scaler from training
        scaled_features = scaler.transform(raw_features)

        # 3. Predict (Returns Class ID, e.g., 7)
        prediction_id = crop_model.predict(scaled_features)[0]

        # 4. Decode (Returns Name, e.g., "Mango")
        crop_name = label_encoder.inverse_transform([prediction_id])[0]

        # 5. Get Confidence Score (Optional but good for UI)
        # StackingClassifier usually supports predict_proba
        confidence = np.max(crop_model.predict_proba(scaled_features))

        return {
            "recommended_crop": crop_name,
            "confidence_score": float(f"{confidence:.4f}")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🌿 DISEASE DIAGNOSIS (Gemini Vision)
@app.post("/diagnose_disease")
async def diagnose_disease(file: UploadFile = File(...)):
    if not GENAI_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key missing.")
    
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Expert Prompt
    prompt = """
    Act as a senior agronomist. Analyze this crop leaf image.
    1. DIAGNOSIS: Identify the specific disease or say 'Healthy'.
    2. CONFIDENCE: Rate your confidence (High/Medium/Low).
    3. SYMPTOMS: List 3 visual symptoms you observe.
    4. TREATMENT: Provide a treatment plan (Chemical & Organic).
    Return response in clean JSON format.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, image])
    return {"analysis": response.text}

# 📚 SCHEME Q&A (RAG Logic)
@app.post("/ask_scheme")
def ask_scheme(query: SchemeQuery):
    if not GENAI_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key missing.")
    
    # System Prompt to reduce hallucinations
    prompt = f"""
    You are an AI Assistant for Indian Farmers. 
    Query: {query.question}
    Context: User is in {query.user_state}.
    Task: Explain relevant government schemes (like PM-KISAN, Fasal Bima Yojana). 
    Be strictly factual. Do not invent subsidy amounts.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return {"response": response.text}

# --- 4. SERVER START ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
