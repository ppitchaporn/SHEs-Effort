import os
import requests
from dotenv import load_dotenv
import math
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Fitness Planner AI")
load_dotenv()

# Constants
API_KEY = os.getenv("SCGC_API_KEY")
CHAT_URL = "https://scgc-llmproxy.scg.com/v1/chat/completions"
MODELS_URL = "https://scgc-llmproxy.scg.com/models"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

DEFAULTS = {
    "gender": "Female", "age": 30, "height_cm": 158, "weight_kg": 59.0,
    "body_fat": 34.0, "muscle_mass": 23.0, "target_body_fat": 30.0,
    "activity": "Active (physical job or lots of movement)",
    "goal": "Gentle muscle tone", "frequency": "3-4 times a week",
}

# --- Logic Layer (Python does the Math - Reliability) ---
# Improvement based on AgentBench [7]: Don't let LLM do math.

def calculate_metrics(gender, age, weight, height, activity_str, goal_str, body_fat, target_fat):
    """Calculates BMR, TDEE, BMI, and Time estimations deterministically."""
    
    # 1. BMR (Mifflin-St Jeor)
    if gender == "Male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    # 2. Activity Multiplier
    if "Sedentary" in activity_str: factor = 1.2
    elif "Light" in activity_str: factor = 1.375
    elif "Moderate" in activity_str: factor = 1.55
    else: factor = 1.725
    tdee = bmr * factor

    # 3. Goal Adjustment & Macros
    if "fat loss" in goal_str.lower():
        cal_factor = 0.85
        protein_ratio = 0.4
    elif "muscle" in goal_str.lower():
        cal_factor = 1.05 if body_fat < target_fat else 0.95 # Recomp
        protein_ratio = 0.35
    else:
        cal_factor = 1.0
        protein_ratio = 0.3

    target_calories = tdee * cal_factor
    
    # Macro Calculation (Gram estimation)
    protein_g = (target_calories * protein_ratio) / 4
    fat_g = (target_calories * 0.3) / 9
    carb_g = (target_calories * (1 - protein_ratio - 0.3)) / 4

    # 4. Timeline Estimation
    weekly_deficit = (tdee - target_calories) * 7
    # Note: 7700 kcal approx 1kg fat
    weekly_fat_change = weekly_deficit / 7700.0 if weekly_deficit > 0 else 0
    
    fat_mass_diff = (weight * (body_fat/100)) - (weight * (target_fat/100))
    weeks_to_target = 0
    if fat_mass_diff > 0 and weekly_fat_change > 0:
        weeks_to_target = math.ceil(fat_mass_diff / weekly_fat_change)
    
    # Cap weeks for realism
    weeks_to_target = max(1, min(weeks_to_target, 52))

    # BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    return {
        "bmr": int(bmr),
        "tdee": int(tdee),
        "target_calories": int(target_calories),
        "macros": {"p": int(protein_g), "f": int(fat_g), "c": int(carb_g)},
        "weeks": weeks_to_target,
        "bmi": round(bmi, 1),
        "bmi_cat": "Normal" if 18.5 <= bmi < 25 else "Over/Under weight" # Simplified
    }

# --- Interface Layer (LLM Interaction - ACI) ---
# Improvement based on SWE-agent [3]: Clear instructions, structured output.

def list_models():
    if not API_KEY: return []
    try:
        resp = requests.get(MODELS_URL, headers=HEADERS, timeout=10)
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        st.error(f"Model API Error: {e}")
        return []

def generate_plan_html(model, user_data, metrics):
    """Generates the HTML/CSS using LLM with pre-calculated metrics."""
    
    # Tree of Thoughts [5]: Ask LLM to reason about the 'Split' before writing code
    prompt = f"""
    You are a professional fitness coach and frontend developer.
    
    **Task:** Create a responsive, beautiful HTML infographic for a fitness plan.
    **Constraint:** Return ONLY the raw HTML code. No markdown fences (```), no conversational text.
    
    **User Profile:**
    - Gender: {user_data['gender']}, Age: {user_data['age']}
    - Goal: {user_data['goal']} ({user_data['frequency']})
    - Body Stats: {user_data['weight_kg']}kg, {user_data['height_cm']}cm, BF {user_data['body_fat']}%
    
    **Pre-calculated Science (USE THESE EXACT VALUES):**
    - BMI: {metrics['bmi']} ({metrics['bmi_cat']})
    - TDEE: {metrics['tdee']} kcal (Maintenance)
    - Target Calories: {metrics['target_calories']} kcal
    - Macros: Protein {metrics['macros']['p']}g, Carbs {metrics['macros']['c']}g, Fat {metrics['macros']['f']}g
    - Estimated Time to Goal: {metrics['weeks']} weeks
    
    **Content Requirements:**
    1. **Workout Plan:** Analyze the profile. Decide if they need Cardio-focus, Hypertrophy, or Hybrid. 
       - Create a bulleted list for {user_data['frequency']} schedule.
       - Focus on compound movements.
    2. **Nutrition:** Suggest 3 Thai meals (Menu Name + Rough Ingredients) that fit the macro definition.
    3. **Design:** Modern CSS, Soft shadows, Rounded corners, Mobile-friendly. Use a clean color palette (Teal/White/Grey).
    
    **Output:**
    <!DOCTYPE html>
    <html>
    <head><style>...css...</style></head>
    <body>...content...</body>
    </html>
    """

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5 # Lower temp for more deterministic code structure
    }
    
    try:
        resp = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=60)
        content = resp.json()["choices"][0]["message"]["content"]
        
        # Reflexion [8] Guardrail: Self-correction of formatting
        if "```" in content:
            content = content.replace("```html", "").replace("```", "")
        return content.strip()
        
    except Exception as e:
        return f"<h3>Error generating plan: {e}</h3>"

# --- Presentation Layer (Streamlit UI) ---
def main():
    st.title("🏋️‍♀️ AI Fitness Planner (Optimized)")
    
    # 1. Setup & Config
    models = list_models()
    if not models:
        st.warning("API Key missing or Service down.")
        return

    # 2. Sidebar / Input Section (Cleaned & Organized)
    with st.sidebar:
        st.header("Profile Settings")
        selected_model = st.selectbox("Model", models, index=0)
        
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", 18, 70, DEFAULTS["age"])
        height = st.number_input("Height (cm)", 140, 220, DEFAULTS["height_cm"])
        weight = st.number_input("Weight (kg)", 40.0, 150.0, DEFAULTS["weight_kg"])
        
        col_a, col_b = st.columns(2)
        with col_a: body_fat = st.number_input("Body Fat %", 5.0, 60.0, DEFAULTS["body_fat"])
        with col_b: target_fat = st.number_input("Target Fat %", 5.0, 60.0, DEFAULTS["target_body_fat"])
        
        activity = st.selectbox("Activity", [
            "Sedentary (Office)", "Light (1-2 days)", 
            "Moderate (3-4 days)", "Active (Physical Job/Sports)"
        ])
        goal = st.selectbox("Goal", ["Fat Loss", "Muscle Gain", "Maintenance/Health"])
        freq = st.selectbox("Frequency", ["2 days/week", "3-4 days/week", "5+ days/week"])
        
        generate = st.button("🚀 Generate Plan", type="primary")

    # 3. Main Execution Flow
    if generate:
        with st.spinner("Analyzing biometrics & Generating Plan..."):
            # A. Calculate Metrics (Reliability Layer - Python handles Math) [7]
            # ใช้ Python คำนวณค่า BMR/TDEE แทนการให้ LLM คิดเองเพื่อความแม่นยำ
            metrics = calculate_metrics(gender, age, weight, height, activity, goal, body_fat, target_fat)
            
            # B. Generate Content (Reasoning Layer - LLM handles Text/HTML) [5][7]
            user_data = {
                "gender": gender, "age": age, "height_cm": height, "weight_kg": weight,
                "body_fat": body_fat, "target_body_fat": target_fat,
                "activity": activity, "goal": goal, "frequency": freq
            }
            html_plan = generate_plan_html(selected_model, user_data, metrics)
            
            # C. Visualize Progress (Chart)
            st.subheader("📉 Projected Progress")
            weeks_arr = np.arange(1, metrics['weeks'] + 1)
            # Simple linear projection
            weight_loss_per_week = (metrics['tdee'] - metrics['target_calories']) * 7 / 7700
            proj_weights = [weight - (weight_loss_per_week * w) for w in weeks_arr]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weeks_arr, y=proj_weights, mode='lines+markers', name='Weight'))
            fig.update_layout(title=f"Estimated Path to {metrics['target_calories']} kcal/day", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # D. Render HTML & Image Download Button (Client-side rendering) [12]
            st.subheader("📋 Your Personalized Plan")

            # ห่อ HTML Plan ด้วย Script html2canvas เพื่อให้ Browser ทำหน้าที่แปลงเป็นรูปภาพ
            html_with_js = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                <style>
                    /* ปุ่ม Download ตกแต่งให้เข้ากับ Streamlit */
                    .btn-download {{
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #ff4b4b;
                        color: white;
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        border: none;
                        font-family: sans-serif;
                        font-weight: 600;
                        cursor: pointer;
                        margin-bottom: 20px;
                        text-decoration: none;
                        transition: background-color 0.2s;
                    }}
                    .btn-download:hover {{
                        background-color: #ff3333;
                    }}
                    /* พื้นที่สำหรับ Capture */
                    #capture-area {{
                        background-color: white; 
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                </style>
            </head>
            <body>
                <div style="text-align: right;">
                    <button class="btn-download" onclick="downloadImage()">
                        📸 Download as Image
                    </button>
                </div>

                <div id="capture-area">
                    {html_plan}
                </div>

                <script>
                    function downloadImage() {{
                        const element = document.getElementById('capture-area');
                        html2canvas(element, {{
                            scale: 2, // เพิ่มความละเอียด (High DPI)
                            useCORS: true,
                            backgroundColor: "#ffffff"
                        }}).then(canvas => {{
                            const link = document.createElement('a');
                            link.download = 'my-fitness-plan.png';
                            link.href = canvas.toDataURL('image/png');
                            link.click();
                        }}).catch(err => {{
                            console.error("Capture failed:", err);
                            alert("Unable to generate image at this time.");
                        }});
                    }}
                </script>
            </body>
            </html>
            """
            
            # ใช้ components.html เพื่อ render และปรับความสูงให้พอดี
            components.html(html_with_js, height=900, scrolling=True)

if __name__ == "__main__":
    main()