import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

# ==== Load environment variables ====
st.write("📂 Loading environment variables...")
load_dotenv()
API_KEY = os.getenv("SCGC_API_KEY")
st.write(f"🔑 Loaded API_KEY: {API_KEY if API_KEY else 'None'}")

MODELS_URL = "https://scgc-llmproxy.scg.com/models"
CHAT_URL = "https://scgc-llmproxy.scg.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
    "Content-Type": "application/json"
}

def list_models():
    """ดึงรายการโมเดลจาก API พร้อม debug"""
    st.write("📡 Calling list_models()...")
    if not API_KEY:
        st.error("❌ API_KEY is missing. Please set SCGC_API_KEY in .env")
        return []
    try:
        st.write(f"🌍 Sending GET {MODELS_URL}")
        response = requests.get(MODELS_URL, headers=headers, timeout=30)
        st.write(f"📥 Response Status: {response.status_code}")
        st.write("📄 Raw Response Text:", response.text)
        response.raise_for_status()
        
        result = response.json()
        st.write("📦 Parsed JSON:", result)
        models = [m["id"] for m in result.get("data", [])]
        st.write(f"✅ Found {len(models)} models:", models)
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {e}")
        return []

def call_llm(model_name, user_prompt):
    """เรียก LLM API พร้อม debug"""
    st.write(f"🤖 Calling LLM with model: {model_name}")
    if not API_KEY:
        st.error("❌ API_KEY is missing. Cannot call LLM API.")
        return "Error: Missing API key"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    st.write("📤 Payload:", payload)

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
        st.write(f"📥 Response Status: {response.status_code}")
        st.write("📄 Raw Response Text:", response.text)
        response.raise_for_status()

        result = response.json()
        st.write("📦 Parsed JSON:", result)
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        else:
            return "❌ No choices returned from API."
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {e}")
        return f"Error: {e}"

# ========== UI จาก app.py เดิม ==========
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

def main():
    st.header("Step 1: Collect Personal Data (for TDEE & Goals)")

    # === โหลดโมเดลและสร้าง dropdown ===
    models = list_models()
    if models:
        selected_model = st.selectbox("เลือก LLM Model", models, index=0)
    else:
        st.warning("⚠ ไม่พบโมเดลจาก API — ตรวจสอบ SCGC_API_KEY หรือการเชื่อมต่อ")
        return  # ไม่โหลด UI ต่อถ้าไม่มีโมเดล

    gender = st.radio(
        "Gender",
        ["Male", "Female", "Other / Prefer not to say"],
        horizontal=True
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Age", options=list(range(10, 101)), index=20)
        body_fat = st.selectbox("Current Body Fat % (optional)", options=[round(x * 0.5, 1) for x in range(0, 201)], index=40)
        target_body_fat = st.selectbox("Target Body Fat % (optional)", options=[round(x * 0.5, 1) for x in range(0, 201)], index=30)
    with col2:
        weight_kg = st.selectbox("Weight (kg)", options=[round(w, 1) for w in range(30, 151)], index=30)
        height_cm = st.selectbox("Height (cm)", options=list(range(120, 221)), index=45)

    col3, col4 = st.columns(2)
    with col3:
        activity = st.selectbox("Activity Level (daily life, not workouts)", [
            "Sedentary (desk job, little movement)",
            "Light (walks sometimes)",
            "Moderate (on feet often)",
            "Active (physical job or lots of movement)"
        ])
    with col4:
        goal = st.selectbox("Goal", [
            "Feel better / Healthier",
            "Gentle fat loss",
            "Gentle muscle tone",
            "Reduce stress & move more"
        ])

    st.markdown("---")

    input_key = f"{gender}-{age}-{weight_kg}-{height_cm}-{body_fat}-{target_body_fat}-{activity}-{goal}-{selected_model}"

    if "last_inputs" in st.session_state and st.session_state.last_inputs == input_key:
        st.markdown("✅ Loaded from previous session")
        st.markdown(st.session_state.last_result)
    else:
        if st.button("Calculate and Run"):
            with st.spinner("Processing your personalized plan..."):
                prompt = f"""
You are a fitness and nutrition coach.
Use the information below to create a **clear final result** only.
Do NOT show python code or calculation steps.
Return the result as a structured text with emojis, ready to display.

**User profile**
- Gender: {gender}
- Age: {age}
- Weight: {weight_kg} kg
- Height: {height_cm} cm
- Current Body Fat: {body_fat}%
- Target Body Fat: {target_body_fat}%
- Activity Level: {activity}
- Goal: {goal}

**What to output**
1. ประเมินความเป็นไปได้จาก User profile ว่าใช้เวลาเท่าไหร่ ถึงจะลด body fat ได้ตาม Target พร้อมคำอธิบาย
2. 🏋️ Workout Plan (weekly structure, warm-up & at-home alternatives แบบละเอียด เช่น บอกชื่อท่า บอกจำนวนเซท จำนวนครั้ง)
3. 🍱 Nutrition Guide (TDEE estimate, macros in grams, และตัวอย่างอาหารไทยที่ช่วยให้ถึงโปรตีนเป้า)

**Important:** 
- Make it short and easy to read, similar to a fitness infographic.
- Use emojis and bullet points.
- ตอบเป็นภาษาไทยเท่านั้น
"""
                result = call_llm(selected_model, prompt)
                st.markdown(result)
                st.session_state.last_inputs = input_key
                st.session_state.last_result = result

if __name__ == "__main__":
    main()