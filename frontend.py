import streamlit as st
import requests
from PIL import Image
from googletrans import Translator

# --- CONFIGURATION ---
BACKEND_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Green Pulse AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TRANSLATION SETUP ---
translator = Translator()

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

# --- SIDEBAR STYLING ---
st.sidebar.image("https://img.icons8.com/color/96/000000/tractor.png", width=80)
st.sidebar.header("🗣️ Language / ਭਾਸ਼ਾ")
selected_lang_name = st.sidebar.selectbox("Select Language", list(lang_options.keys()))
target_lang = lang_options[selected_lang_name]

# --- TRANSLATION FUNCTION ---
def t(text):
    if target_lang == 'en': return text
    try:
        return translator.translate(text, dest=target_lang).text
    except Exception as e:
        print(f"Translation Error for '{text}': {e}")
        return text

# --- MAIN HEADER ---
st.markdown(f"<h1 style='text-align: center; color: #2E7D32;'>🌿 {t('Green Pulse AI')}</h1>", unsafe_allow_html=True)
farmer_text = "Farmer's Smart Assistant for Crops, Diseases & Schemes"
st.markdown(f"<h4 style='text-align: center; color: #555;'>* {t(farmer_text)} *</h4>", unsafe_allow_html=True)
st.divider()

# --- NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.header(t("Navigation"))
nav_options = ["🌱 Crop Recommendation", "🍂 Disease Diagnosis", "📜 Government Schemes"]
page = st.sidebar.radio(t("Go to:"), nav_options, format_func=t)

# --- PAGE 1: CROP RECOMMENDATION ---
if page == "🌱 Crop Recommendation":
    st.header(t("🌱 Precision Crop Recommendation"))
    st.info(t("Fill in the details below to get an AI-powered crop suggestion."))

    # Create two columns for better layout
    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.subheader(t("🧪 Soil Nutrients"))
            
            # Nitrogen
            n = st.number_input(t("Nitrogen (N) [kg/ha]"), 0, 500, 90)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Target: 240-480 kg/ha"))

            # Phosphorus
            p = st.number_input(t("Phosphorus (P) [kg/ha]"), 0, 300, 42)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Target: 11-22 kg/ha"))
            
            # Potassium (Updated Limit: 350)
            k = st.number_input(t("Potassium (K) [kg/ha]"), 0, 350, 43)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Target: 110-280 kg/ha"))

            # pH
            ph = st.number_input(t("Soil pH"), 0.0, 14.0, 6.5)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Ideal: 6.0 - 7.5"))

    with right_col:
        with st.container(border=True):
            st.subheader(t("🌦️ Climate Conditions"))
            
            # Temperature
            temp = st.number_input(t("Temperature (°C)"), -10.0, 60.0, 20.8)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Ideal: 18°C - 30°C"))

            # Humidity
            humidity = st.number_input(t("Humidity (%)"), 0.0, 100.0, 82.0)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Optimal: 50% - 80%"))

            # Rainfall (Updated Limit: 2000)
            rain = st.number_input(t("Rainfall (mm)"), 0.0, 2000.0, 202.9)
            with st.expander(t("ℹ️ Guidelines")):
                st.caption(t("Range: 300 - 1000+ mm"))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Large Action Button
    if st.button(t("🔍 Recommend Crop"), type="primary", use_container_width=True):
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
                st.balloons() # Fun effect!
                
                # Result Display
                st.success(f"### ✅ {t('Best Crop to Grow')}: {t(crop_en)}")
                st.progress(result['confidence_score'], text=t("Confidence Score"))
            else:
                st.error(t("Error connecting to server. Is backend running?"))
                
        except Exception as e:
            st.error(f"{t('Connection Failed')}: {e}")

# --- PAGE 2: DISEASE DIAGNOSIS ---
elif page == "🍂 Disease Diagnosis":
    st.header(t("🍂 Plant Disease Diagnosis"))
    st.warning(t("Upload a clear photo of the affected leaf for instant analysis."))

    uploaded_file = st.file_uploader(t("Choose an image..."), type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption=t("Uploaded Leaf"), use_column_width=True)
        
        with col2:
            if st.button(t("🧬 Diagnose Disease"), type="primary"):
                files = {"file": uploaded_file.getvalue()}
                
                try:
                    with st.spinner(t("Consulting AI Agronomist...")):
                        response = requests.post(f"{BACKEND_URL}/diagnose_disease", files=files)
                    
                    if response.status_code == 200:
                        analysis_en = response.json()["analysis"]
                        st.success(t("Analysis Complete!"))
                        with st.container(border=True):
                            st.markdown(t(analysis_en))
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
    
    question = st.text_area(t("Your Question"), t("What subsidies are available for wheat farming?"), height=100)

    if st.button(t("🤖 Ask AI"), type="primary"):
        payload = {"question": question, "user_state": user_state}
        
        try:
            with st.spinner(t("Searching scheme database...")):
                response = requests.post(f"{BACKEND_URL}/ask_scheme", json=payload)
            
            if response.status_code == 200:
                answer_en = response.json()["response"]
                with st.chat_message("assistant"):
                    st.markdown(t(answer_en))
            else:
                st.error(t("Server Error."))
        except Exception as e:
            st.error(f"{t('Connection Failed')}: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #888;'>{t('Built with 💚 using Streamlit & FastAPI')}</div>", unsafe_allow_html=True)
