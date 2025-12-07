import os
import joblib
import uvicorn
import numpy as np
# Fix for numpy compatibility with joblib models
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Suppress importlib metadata warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import io

# --- 1. CONFIGURATION ---
load_dotenv()
app = FastAPI(title="Green Pulse AI - High Precision API")

# --- API KEY CONFIGURATION ---
# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Load AI Artifacts with NumPy compatibility fix
def load_model_safe(model_path):
    """Load joblib model with NumPy compatibility workaround"""
    try:
        # Set numpy random state before loading
        np.random.seed(42)
        # Try normal loading first
        return joblib.load(model_path)
    except (ValueError, AttributeError, TypeError) as e:
        error_str = str(e)
        if "BitGenerator" in error_str or "MT19937" in error_str:
            # NumPy version incompatibility - model needs to be retrained
            raise ValueError(
                f"Model compatibility issue detected: {e}\n"
                "The model was saved with a different NumPy version.\n"
                "Solution: Retrain the model by running: python3 train_model.py"
            )
        else:
            raise e

try:
    crop_model = load_model_safe("models/crop_model.joblib")
    label_encoder = load_model_safe("models/label_encoder.joblib")
    scaler = load_model_safe("models/scaler.joblib")
    print("✅ High-Precision Models Loaded Successfully.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Models not found. You MUST run 'train_model.py' first. {e}")
    crop_model = None
    label_encoder = None
    scaler = None
except Exception as e:
    print(f"❌ ERROR: Failed to load models. {e}")
    print("💡 Tip: This might be a version compatibility issue. Try retraining the model with:")
    print("   python3 train_model.py")
    crop_model = None
    label_encoder = None
    scaler = None

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

# 🌱 CROP RECOMMENDATION
@app.post("/recommend_crop")
def recommend_crop(data: CropInput):
    if not crop_model or not label_encoder or not scaler:
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Please run 'train_model.py' first or check for compatibility issues."
        )

    try:
        raw_features = np.array([[
            data.N, data.P, data.K, 
            data.temperature, data.humidity, 
            data.ph, data.rainfall
        ]])
        scaled_features = scaler.transform(raw_features)
        prediction_id = crop_model.predict(scaled_features)[0]
        crop_name = label_encoder.inverse_transform([prediction_id])[0]
        confidence = np.max(crop_model.predict_proba(scaled_features))

        return {
            "recommended_crop": crop_name,
            "confidence_score": float(f"{confidence:.4f}")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🌿 DISEASE DIAGNOSIS
@app.post("/diagnose_disease")
async def diagnose_disease(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key missing.")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        prompt = """
        Act as a senior agronomist. Analyze this crop leaf image.
        1. DIAGNOSIS: Identify the specific disease or say 'Healthy'.
        2. CONFIDENCE: Rate your confidence (High/Medium/Low).
        3. SYMPTOMS: List 3 visual symptoms you observe.
        4. TREATMENT: Provide a treatment plan (Chemical & Organic).
        Return response in clean JSON format.
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt, image])
        return {"analysis": response.text}
    except Exception as e:
        print(f"❌ GEMINI ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# 📚 SCHEME Q&A
@app.post("/ask_scheme")
def ask_scheme(query: SchemeQuery):
    try:
        prompt = f"""
        You are an AI Assistant for Indian Farmers. 
        Query: {query.question}
        Context: User is in {query.user_state}.
        Task: Explain relevant government schemes (like PM-KISAN, Fasal Bima Yojana). 
        Be strictly factual. Do not invent subsidy amounts.
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        print(f"❌ SCHEME ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# --- 4. SERVER START ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
