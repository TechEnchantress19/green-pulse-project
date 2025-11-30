import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIGURATION ---
BACKEND_URL = "http://localhost:8000"  # Points to your FastAPI server
st.set_page_config(
    page_title="Green Pulse AI",
    page_icon="🌿",
    layout="wide"
)

# --- HEADER ---
st.title("🌿 Green Pulse AI")
st.markdown("### *Farmer's Smart Assistant for Crops, Diseases & Schemes*")
st.divider()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["🌱 Crop Recommendation", "🍂 Disease Diagnosis", "📜 Government Schemes"])

# --- PAGE 1: CROP RECOMMENDATION ---
if page == "🌱 Crop Recommendation":
    st.header("🌱 Precision Crop Recommendation")
    st.info("Enter the soil details below to get the best crop suggestion.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
    
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=20.8)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    
    with col3:
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.9)

    if st.button("🔍 Recommend Crop"):
        payload = {
            "N": n, "P": p, "K": k,
            "temperature": temp, "humidity": humidity,
            "ph": ph, "rainfall": rain
        }
        
        try:
            with st.spinner("Analyzing soil data..."):
                response = requests.post(f"{BACKEND_URL}/recommend_crop", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Recommended Crop: **{result['recommended_crop']}**")
                st.metric(label="Confidence Score", value=f"{result['confidence_score']*100:.2f}%")
            else:
                st.error("Error connecting to server. Is backend running?")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# --- PAGE 2: DISEASE DIAGNOSIS ---
elif page == "🍂 Disease Diagnosis":
    st.header("🍂 Plant Disease Diagnosis")
    st.info("Upload a clear photo of the affected leaf.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        if st.button("🧬 Diagnose Disease"):
            # Prepare file for API
            files = {"file": uploaded_file.getvalue()}
            
            try:
                with st.spinner("Consulting AI Agronomist..."):
                    response = requests.post(f"{BACKEND_URL}/diagnose_disease", files=files)
                
                if response.status_code == 200:
                    analysis = response.json()["analysis"]
                    st.success("Analysis Complete!")
                    st.write(analysis)
                else:
                    st.error("Server Error. Check API Key.")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- PAGE 3: SCHEMES ---
elif page == "📜 Government Schemes":
    st.header("📜 Government Scheme Assistant")
    st.info("Ask about subsidies, insurance, and benefits.")

    user_state = st.selectbox("Select Your State", ["India", "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Tamil Nadu", "Other"])
    question = st.text_area("Your Question", "What subsidies are available for wheat farming?")

    if st.button("🤖 Ask AI"):
        payload = {"question": question, "user_state": user_state}
        
        try:
            with st.spinner("Searching scheme database..."):
                response = requests.post(f"{BACKEND_URL}/ask_scheme", json=payload)
            
            if response.status_code == 200:
                answer = response.json()["response"]
                st.markdown("### AI Response:")
                st.write(answer)
            else:
                st.error("Server Error.")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with 💚 using Streamlit & FastAPI")
