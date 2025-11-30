import streamlit as st
import requests
from PIL import Image
from googletrans import Translator

# --- CONFIGURATION ---
BACKEND_URL = "http://localhost:8000"
st.set_page_config(page_title="Green Pulse AI", page_icon="🌿", layout="wide")

# --- TRANSLATION SETUP ---
translator = Translator()

# List of supported languages
lang_options = {
    'English': 'en',
    'Hindi (हिंदी)': 'hi',
    'Punjabi (ਪੰਜਾਬੀ)': 'pa',
    'Marathi (मराठी)': 'mr',
    'Tamil (தமிழ்)': 'ta',
    'Telugu (తెలుగు)': 'te',
    'Bengali (বাংলা)': 'bn',
    'Gujarati (ગુજરાતી)': 'gu'
}

# Sidebar Language Selector
st.sidebar.header("🗣️ Language / ਭਾਸ਼ਾ")
selected_lang_name = st.sidebar.selectbox(
    "Select Language", 
    list(lang_options.keys())
)
target_lang = lang_options[selected_lang_name]

# --- TRANSLATION FUNCTION (Fix: Removed Caching) ---
def t(text):
    if target_lang == 'en': 
        return text
    try:
        # Translate and return text
        return translator.translate(text, dest=target_lang).text
    except Exception as e:
        # If it fails, print the error to the terminal and keep English
        print(f"Translation Error for '{text}': {e}")
        return text

# --- HEADER ---
st.title(t("🌿 Green Pulse AI"))
st.markdown(f"### *{t("Farmer's Smart Assistant for Crops, Diseases & Schemes")}*")
st.divider()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header(t("Navigation"))
nav_options = ["🌱 Crop Recommendation", "🍂 Disease Diagnosis", "📜 Government Schemes"]
page = st.sidebar.radio(t("Go to:"), nav_options, format_func=t)

# --- PAGE 1: CROP RECOMMENDATION ---
if page == "🌱 Crop Recommendation":
    st.header(t("🌱 Precision Crop Recommendation"))
    st.info(t("Enter the soil details below to get the best crop suggestion."))

    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.number_input(t("Nitrogen (N)"), 0, 200, 90)
        p = st.number_input(t("Phosphorus (P)"), 0, 200, 42)
        k = st.number_input(t("Potassium (K)"), 0, 200, 43)
    
    with col2:
        temp = st.number_input(t("Temperature (°C)"), -10.0, 60.0, 20.8)
        humidity = st.number_input(t("Humidity (%)"), 0.0, 100.0, 82.0)
        ph = st.number_input(t("Soil pH"), 0.0, 14.0, 6.5)
    
    with col3:
        rain = st.number_input(t("Rainfall (mm)"), 0.0, 500.0, 202.9)

    if st.button(t("🔍 Recommend Crop")):
        payload = {
            "N": n, "P": p, "K": k,
            "temperature": temp, "humidity": humidity,
            "ph": ph, "rainfall": rain
        }
        
        try:
            with st.spinner(t("Analyzing soil data...")):
                response = requests.post(f"{BACKEND_URL}/recommend_crop", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                crop_en = result['recommended_crop']
                st.success(f"✅ {t('Recommended Crop')}: **{t(crop_en)}**")
                st.metric(label=t("Confidence Score"), value=f"{result['confidence_score']*100:.2f}%")
            else:
                st.error(t("Error connecting to server. Is backend running?"))
                
        except Exception as e:
            st.error(f"{t('Connection Failed')}: {e}")

# --- PAGE 2: DISEASE DIAGNOSIS ---
elif page == "🍂 Disease Diagnosis":
    st.header(t("🍂 Plant Disease Diagnosis"))
    st.info(t("Upload a clear photo of the affected leaf."))

    uploaded_file = st.file_uploader(t("Choose an image..."), type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=t("Uploaded Leaf"), use_column_width=True)

        if st.button(t("🧬 Diagnose Disease")):
            files = {"file": uploaded_file.getvalue()}
            
            try:
                with st.spinner(t("Consulting AI Agronomist...")):
                    response = requests.post(f"{BACKEND_URL}/diagnose_disease", files=files)
                
                if response.status_code == 200:
                    analysis_en = response.json()["analysis"]
                    st.success(t("Analysis Complete!"))
                    st.write(t(analysis_en))
                else:
                    st.error(t("Server Error. Check API Key."))
            except Exception as e:
                st.error(f"{t('Connection Failed')}: {e}")

# --- PAGE 3: SCHEMES ---
elif page == "📜 Government Schemes":
    st.header(t("📜 Government Scheme Assistant"))
    st.info(t("Ask about subsidies, insurance, and benefits."))

    state_options = ["India", "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Tamil Nadu", "Other"]
    user_state = st.selectbox(t("Select Your State"), state_options, format_func=t)
    
    question = st.text_area(t("Your Question"), t("What subsidies are available for wheat farming?"))

    if st.button(t("🤖 Ask AI")):
        payload = {"question": question, "user_state": user_state}
        
        try:
            with st.spinner(t("Searching scheme database...")):
                response = requests.post(f"{BACKEND_URL}/ask_scheme", json=payload)
            
            if response.status_code == 200:
                answer_en = response.json()["response"]
                st.markdown(f"### {t('AI Response:')}")
                st.write(t(answer_en))
            else:
                st.error(t("Server Error."))
        except Exception as e:
            st.error(f"{t('Connection Failed')}: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(t("Built with 💚 using Streamlit & FastAPI"))
