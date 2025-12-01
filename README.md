# 🌿 Green Pulse AI: Farmer's Best Friend

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green)
![AI](https://img.shields.io/badge/AI-Gemini%20Vision-orange)
![ML](https://img.shields.io/badge/Model-Stacked%20Ensemble-red)

**Green Pulse AI** is a high-precision, voice-enabled AI assistant designed to help farmers make data-driven decisions. It combines a state-of-the-art **Stacked Ensemble Machine Learning model** for crop recommendation with **Generative AI (Google Gemini)** for plant disease diagnosis and government scheme assistance.

## 🚀 Key Features

### 1. High-Precision Crop Recommendation
* **Technology:** Stacked Ensemble (Random Forest + XGBoost + LightGBM).
* **Meta-Learner:** Logistic Regression.
* **Accuracy:** Optimized for high precision using Stratified K-Fold validation and feature scaling.
* **Input:** N, P, K, Temperature, Humidity, pH, Rainfall.

### 2. AI Disease Diagnosis
* **Technology:** Google Gemini 1.5 Flash (Vision).
* **Capability:** Analyzes leaf images to identify diseases, list symptoms, and suggest organic/chemical treatments.

### 3. Government Scheme Assistant (RAG-Lite)
* **Technology:** Generative AI Contextual Prompting.
* **Capability:** Answers questions about Indian government schemes (PM-KISAN, Fasal Bima Yojana) based on the user's state.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Generative AI:** Google Gemini API (via `google-generativeai`)
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Joblib

---

## 📂 Project Structure

```text
green-pulse-project/
│
├── main.py                 # The FastAPI Backend Server
├── train_model.py          # Script to train the Stacked Ensemble Model
├── requirements.txt        # Project dependencies
├── .env                    # API Keys (Not committed to Git)
│
└── models/                 # Binary model files (Generated locally)
    ├── crop_model.joblib   # The trained Stacked Ensemble
    ├── label_encoder.joblib # Decodes class IDs to Crop Names
    └── scaler.joblib       # Standard Scaler for high precision
```
⚡ Installation & Setup
Follow these steps to set up the project locally.

1. Clone the Repository
```
git clone [https://github.com/TechEnchantress19/green-pulse-project.git](https://github.com/TechEnchantress19/green-pulse-project.git)
cd green-pulse-project
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Set Up API Keys
Create a file named .env in the root folder and add your Google Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```
4. Train the AI Model
Since the trained model files are large and binary, they are not stored on GitHub. You must generate them locally:
```
python train_model.py
```
Wait for the message: ✅ Model Trained!

5. Run the Server
Start the FastAPI backend:
```
python main.py
```
The server will start at http://0.0.0.0:8000.

## 📖 API Usage
Once the server is running, you can access the Interactive API Docs at: 👉 http://localhost:8000/docs

Endpoints
### 1. Recommend Crop
URL: /recommend_crop

Method: POST

Body:
```
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```
### 2. Diagnose Disease
URL: /diagnose_disease

Method: POST

Body: multipart/form-data (Upload an image file)

### 3. Ask Scheme
URL: /ask_scheme

Method: POST

Body:
```
{
  "question": "What is the subsidy for PM-KISAN?",
  "user_state": "Punjab"
}
```
## 🤝 Contributing
Contributions are welcome!

#### 1. Fork the repository.

#### 2. Create a new branch (```git checkout -b feature-branch```).

#### 3. Commit your changes.

#### 4. Push to the branch.

#### 5. Open a Pull Request.

## 📄 License
This project is licensed under the MIT License.
