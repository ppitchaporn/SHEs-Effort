import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import math
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- Configuration and Setup ---

st.set_page_config(layout="wide", page_title="Fitness Planner")

load_dotenv()
API_KEY = os.getenv("SCGC_API_KEY")

MODELS_URL = "https://scgc-llmproxy.scg.com/models"
CHAT_URL = "https://scgc-llmproxy.scg.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Default values for first-time use ---
DEFAULTS = {
    "gender": "Female",
    "age": 30,
    "body_fat": 34.0,
    "muscle_mass": 23.0,
    "target_body_fat": 25.0,
    "weight_kg": 59.0,
    "height_cm": 158,
    "activity": "Sedentary (desk job, little movement)",
    "goal": "Feel better / Healthier",
    "frequency": "3-4 times a week",
}

def get_default(name, fallback):
    """Return value from session_state if exists, else fallback."""
    return st.session_state.get(name, fallback)

# --- API Functions ---

@st.cache_data(ttl=3600)
def list_models():
    """Fetches and caches the list of available LLM models from the API."""
    if not API_KEY:
        st.error("API_KEY is not configured. Please set SCGC_API_KEY in your .env file.")
        return []
    try:
        response = requests.get(MODELS_URL, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        models = [m["id"] for m in result.get("data", [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the model API: {e}")
        return []

def call_llm(model_name, user_prompt):
    """Call LLM and return the assistant content."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
    }
    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to call the LLM API: {e}")
        return f"An error occurred: {e}"

# --- Prompt Template ---

def get_html_generation_prompt(
    gender,
    age,
    weight_kg,
    height_cm,
    body_fat,
    muscle_mass,
    target_body_fat,
    activity,
    goal,
    frequency,
    estimated_weeks,
):
    """Prompt for HTML infographic generation."""
    return f"""
You are an expert front-end developer and fitness coach who writes clean, modern, self-contained HTML and CSS, precisely replicating a given design style. Your task is to generate a complete, data-driven health infographic as a single HTML file based on the user data and the detailed design specification below.

You must perform the necessary calculations (BMI, TDEE, etc.) based on the user data to fill in the metric values.

---
### **Part 1: Content to Include**

**User Profile:**
- **Age:** {age}
- **Gender:** {gender}
- **Height:** {height_cm} cm
- **Weight:** {weight_kg} kg
- **Current Body Fat:** {body_fat}%
- **Current Muscle Mass:** {muscle_mass}%
- **Target Body Fat:** {target_body_fat}%
- **Activity Level:** {activity}
- **Primary Goal:** {goal}

**Key Health Metrics (Calculated):**
- **BMI:** [Calculate Value] and provide its category (e.g., Normal, Overweight).
- **Estimated Daily Calories:** [Calculate TDEE Value] for the user's goal.
    - **Macro Breakdown:** Provide approximate grams per day for Protein, Carbs, and Fats.
        - **Always Example food of each Macro:** [Provide brief daily gram values for Protein, Carbs, Fats.]
        - **Example Calculation:** [Provide brief explanation of how TDEE and macros were calculated.]
        - **Estimated Body Fat Change:** [Calculate Value] per week based on calorie deficit/surplus.
    - **Body Fat to Lose/Gain:** Calculate the amount of fat to lose/gain to reach the target body fat percentage.

**Weekly Exercise Plan Overview:**
- **Initial Analysis and Phasing:**
    - First, analyze the user's complete profile (age, goal, body fat, muscle mass, activity level).
    - Based on this analysis, determine if they should start with a focus on cardio, weight training, or a hybrid approach. **You must state your reasoning clearly.**
    - Structure the plan into two phases: a **Short-term Plan (First 4 Weeks)** and a **Long-term Plan (Moving Forward)**.

- **Detailed Workout Plan (for the Short-term Plan):**
    - **You MUST present the workout plan as a simple, nested bulleted list (`<ul>` and `<li>` tags).** Do not use tables or complex column layouts.
    - Create a main bullet point for each workout day based on the user's frequency ('{frequency}'). Each bullet should state the day and muscle group.
    - Under each main day, create a nested bulleted list of the gym-based exercises with sets and reps.
    - After the list, add a concluding sentence about rest days.

**Nutrition Highlights:**
- **Daily Calorie Target:** State the calculated value.
- **Macro Focus:** Describe a focus on high protein, moderate carbs, and healthy fats.
    - **Sample Meals:** Provide 1–2 brief examples of meals/snacks that fit each macro focus.
    - **Guidelines:** Provide 5 simple meals in Thai style (เมนูอาหารไทย).

**Goal Tracking:**
- **Short-term Goals (4 weeks):** List 2–3 specific goals.
- **Medium-term Goals (3 months):** List 2–3 broader outcome goals.
- **Estimated time to reach target body fat:** {estimated_weeks} weeks based on current rate of change.

**Footer:**
- Brief explanation of what TDEE and Macros are.
- A friendly motivational message.

---
### **Part 2: Design & Style Specification**

[Keep the detailed style system, layout, responsive rules, colors, components, and text styles exactly as specified in the original prompt.]

---
### **Part 3: Technical & Code Requirements**

- All CSS in a `<style>` tag and all JS in a `<script>` tag (no external files).
- No external libraries.
- Use an initialization guard (`if (!window.infographicInitialized)`) and a debounced resize handler for any chart inside the HTML.

---
### **Final Output Format**

Generate the complete HTML code as a single block. Do not add any conversational text, explanations, or markdown fences. The output must be ready to be saved directly as a `.html` file.
"""

# --- Streamlit UI ---

def main():
    st.title("✨ Your Personalized Fitness Planner ✨")
    st.write("Tell us a bit about yourself and your goals, and we'll create a plan just for you!")

    models = list_models()
    if not models:
        st.warning("Could not load LLM models. The application cannot proceed.")
        return

    default_model_name = "GPT-4o"
    default_index = models.index(default_model_name) if default_model_name in models else 0

    # ---------------- Input Section ----------------
    with st.container(border=True):
        selected_model = st.selectbox("Select LLM Model", models, index=default_index, key="selected_model")
        st.divider()

        st.subheader("Personal Details")

        # Gender
        gender_default = get_default("gender", DEFAULTS["gender"])
        gender_options = ["Male", "Female", "Other / Prefer not to say"]
        if gender_default not in gender_options:
            gender_default = DEFAULTS["gender"]
        gender = st.radio(
            "Gender",
            gender_options,
            horizontal=True,
            index=gender_options.index(gender_default),
            key="gender",
        )

        col1, col2 = st.columns(2)
        with col1:
            # Age
            age_options = list(range(20, 65))
            age_default = get_default("age", DEFAULTS["age"])
            if age_default not in age_options:
                age_default = DEFAULTS["age"]
            age = st.selectbox(
                "Age",
                options=age_options,
                index=age_options.index(age_default),
                key="age",
            )

            # Body fat
            fat_options = [round(x * 0.5, 1) for x in range(5, 81)]
            body_fat_default = get_default("body_fat", DEFAULTS["body_fat"])
            if body_fat_default not in fat_options:
                body_fat_default = DEFAULTS["body_fat"]
            
            # --- Muscle Mass Input ---
            muscle_options = [round(x * 0.5, 1) for x in range(10, 101)]
            muscle_mass_default = get_default("muscle_mass", DEFAULTS["muscle_mass"])
            if muscle_mass_default not in muscle_options:
                muscle_mass_default = DEFAULTS["muscle_mass"]
                
            body_fat = st.selectbox(
                "Current Body Fat %",
                options=fat_options,
                index=fat_options.index(body_fat_default),
                key="body_fat",
            )

            muscle_mass = st.selectbox(
                "Current Muscle Mass %",
                options=muscle_options,
                index=muscle_options.index(muscle_mass_default),
                key="muscle_mass",
            )

            target_body_fat_default = get_default("target_body_fat", DEFAULTS["target_body_fat"])
            if target_body_fat_default not in fat_options:
                target_body_fat_default = DEFAULTS["target_body_fat"]
            
            target_body_fat = st.selectbox(
                "Target Body Fat %",
                options=fat_options,
                index=fat_options.index(target_body_fat_default),
                key="target_body_fat",
            )

        with col2:
            # Weight
            weight_options = [w * 0.5 for w in range(30, 301)]  # Expanded range
            weight_default = get_default("weight_kg", DEFAULTS["weight_kg"])
            if weight_default not in weight_options:
                weight_default = DEFAULTS["weight_kg"]
            weight_kg = st.selectbox(
                "Weight (kg)",
                options=weight_options,
                index=weight_options.index(weight_default),
                key="weight_kg",
            )

            # Height
            height_options = list(range(140, 201))
            height_default = get_default("height_cm", DEFAULTS["height_cm"])
            if height_default not in height_options:
                height_default = DEFAULTS["height_cm"]
            height_cm = st.selectbox(
                "Height (cm)",
                options=height_options,
                index=height_options.index(height_default),
                key="height_cm",
            )

        st.divider()
        st.subheader("Activity & Goals")
        col3, col4, col5 = st.columns(3)

        with col3:
            activity_options = [
                "Sedentary (desk job, little movement)",
                "Light (walks sometimes)",
                "Moderate (on feet often)",
                "Active (physical job or lots of movement)",
            ]
            activity_default = get_default("activity", DEFAULTS["activity"])
            if activity_default not in activity_options:
                activity_default = DEFAULTS["activity"]
            activity = st.selectbox(
                "Activity Level (daily life, not workouts)",
                activity_options,
                index=activity_options.index(activity_default),
                key="activity",
            )

        with col4:
            goal_options = [
                "Feel better / Healthier",
                "Gentle fat loss",
                "Gentle muscle tone",
                "Reduce stress & move more",
            ]
            goal_default = get_default("goal", DEFAULTS["goal"])
            if goal_default not in goal_options:
                goal_default = DEFAULTS["goal"]
            goal = st.selectbox(
                "Goal",
                goal_options,
                index=goal_options.index(goal_default),
                key="goal",
            )

        with col5:
            frequency_options = [
                "1-2 times a week",
                "3-4 times a week",
                "5+ times a week",
            ]
            frequency_default = get_default("frequency", DEFAULTS["frequency"])
            if frequency_default not in frequency_options:
                frequency_default = DEFAULTS["frequency"]
            frequency = st.selectbox(
                "Preferred Workout Frequency",
                frequency_options,
                index=frequency_options.index(frequency_default),
                key="frequency",
            )

        generate_btn = st.button("Generate My Plan", type="primary")

    # ---------------- Calculation & Logic ----------------

    # Ensure defaults are used if needed
    start_weight = float(weight_kg)
    start_body_fat = float(body_fat)
    start_muscle_mass = float(muscle_mass)
    target_body_fat_val = float(target_body_fat)
    height_m = float(height_cm) / 100.0

    # Include muscle_mass in the cache key
    input_key = f"{selected_model}-{gender}-{age}-{start_weight}-{height_cm}-{start_body_fat}-{start_muscle_mass}-{target_body_fat_val}-{activity}-{goal}-{frequency}"

    if "cached_results" not in st.session_state:
        st.session_state["cached_results"] = {}
    cached = st.session_state["cached_results"]

    html_content_to_display = None
    
    # Run only if button clicked or if we already have a cached result for this exact input
    if generate_btn or (input_key in cached):
        
        # --- Basic Math ---
        # Mifflin-St Jeor
        if gender == "Male":
            bmr = (10 * start_weight) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * start_weight) + (6.25 * height_cm) - (5 * age) - 161

        if "Sedentary" in activity:
            activity_factor = 1.2
        elif "Light" in activity:
            activity_factor = 1.375
        elif "Moderate" in activity:
            activity_factor = 1.55
        else:
            activity_factor = 1.725

        tdee = bmr * activity_factor

        # --- [FIX 1] Improved Calorie Factor Logic ---
        if goal == "Gentle fat loss":
            calorie_factor = 0.8
        elif goal == "Gentle muscle tone":
            calorie_factor = 1.05
        else:
            # For "Feel better" or others:
            # If current fat > target fat, allow a small deficit (5%) to make progress visible.
            if start_body_fat > target_body_fat_val:
                calorie_factor = 0.95
            else:
                calorie_factor = 1.0

        target_calories = tdee * calorie_factor

        # Weekly "Energy Balance" (not just deficit)
        weekly_energy_balance = (target_calories - tdee) * 7
        
        # --- [FIX 2] Sign Correction ---
        # If target < tdee (Deficit), balance is Negative.
        # Fat change should also be Negative (Weight Loss).
        weekly_fat_change_kg = weekly_energy_balance / 7700.0 

        # Calculate Mass
        start_fat_mass = start_weight * (start_body_fat / 100.0)
        target_fat_mass = start_weight * (target_body_fat_val / 100.0)
        
        # How much fat to lose (absolute value)
        fat_to_lose_kg = max(0.0, start_fat_mass - target_fat_mass)

        # Estimate Weeks
        # If we need to lose fat (fat_to_lose > 0) AND we are losing weight (weekly_fat_change_kg < 0)
        if weekly_fat_change_kg < 0 and fat_to_lose_kg > 0:
            weeks_to_target = fat_to_lose_kg / abs(weekly_fat_change_kg)
        else:
            weeks_to_target = 0.0

        estimated_weeks = max(1, math.ceil(weeks_to_target))
        max_weeks_for_chart = 52
        estimated_weeks = min(estimated_weeks, max_weeks_for_chart)

        # --- LLM + cache ---
        if input_key in cached:
            st.success("Loaded from cache!")
            html_result = cached[input_key]
            html_content_to_display = cached[input_key]
        else:
            with st.spinner("Processing your personalized plan..."):
                prompt = get_html_generation_prompt(
                    gender,
                    age,
                    weight_kg,
                    height_cm,
                    body_fat,
                    muscle_mass,
                    target_body_fat,
                    activity,
                    goal,
                    frequency,
                    estimated_weeks=estimated_weeks,
                )
                html_result = call_llm(selected_model, prompt)

                if html_result.strip().startswith("```html"):
                    html_result = html_result.strip()[7:-3].strip()

                cached[input_key] = html_result
                st.session_state["cached_results"] = cached
                html_content_to_display = html_result

        # --------------- Display HTML + Chart ---------------
        if html_content_to_display:
            wrapped = (
                "<div style='display:flex;justify-content:flex-end;margin-bottom:8px'>"
                "  <button id='download-infographic' style='background:#1abc9c;border:none;color:white;padding:8px 12px;border-radius:8px;cursor:pointer;'>ดาวน์โหลดภาพ (PNG)</button>"
                "</div>"
                "<div id='infographic-root'>"
                + html_content_to_display +
                "</div>"
                "<script src='https://html2canvas.hertzen.com/dist/html2canvas.min.js'></script>"
                "<script>"
                "  (function(){"
                "    const btn = document.getElementById('download-infographic');"
                "    btn.addEventListener('click', async function(){"
                "      const node = document.getElementById('infographic-root');"
                "      try {"
                "        const canvas = await html2canvas(node, {scale: 2, useCORS: true});"
                "        canvas.toBlob(function(blob){"
                "          const url = URL.createObjectURL(blob);"
                "          const a = document.createElement('a');"
                "          a.href = url;"
                "          a.download = 'healthinfographic.png';"
                "          document.body.appendChild(a);"
                "          a.click();"
                "          a.remove();"
                "          URL.revokeObjectURL(url);"
                "        }, 'image/png');"
                "      } catch (e) {"
                "        alert('เกิดข้อผิดพลาดขณะสร้างภาพ: ' + e);"
                "      }"
                "    });"
                "  })();"
                "</script>"
            )

            components.html(wrapped, height=1200, scrolling=True)

            # -------- Estimated Body Composition Progress (Weeks Until Target) --------
            st.subheader("Estimated Body Composition Progress (Weeks Until Target)")

            # Weeks axis
            weeks = np.arange(1, estimated_weeks + 1)
            week_labels = [f"Week {w}" for w in weeks]

            bmi = start_weight / (height_m ** 2) if height_m > 0 else None

            # Simulation variables
            current_weight = start_weight
            current_fat_mass = start_weight * (start_body_fat / 100.0)
            
            # Assume Muscle Mass (kg) stays constant (Lean retention)
            constant_muscle_mass_kg = start_weight * (start_muscle_mass / 100.0)
            
            weights = []
            body_fats = []
            muscle_pcts = []

            for _ in weeks:
                # Update absolute values
                # If weekly_fat_change_kg is negative, weight decreases
                current_weight += weekly_fat_change_kg
                current_fat_mass += weekly_fat_change_kg 
                
                # Safety checks
                current_weight = max(current_weight, constant_muscle_mass_kg + 1.0)
                current_fat_mass = max(current_fat_mass, 0.0)

                # Recalculate percentages
                if current_weight > 0:
                    current_fat_pct = (current_fat_mass / current_weight) * 100.0
                    est_muscle_pct = (constant_muscle_mass_kg / current_weight) * 100.0
                else:
                    current_fat_pct = 0.0
                    est_muscle_pct = 0.0
                
                # Cap logic
                current_fat_pct = max(0.0, min(100.0, current_fat_pct))
                est_muscle_pct = max(0.0, min(100.0, est_muscle_pct))

                if goal == "Gentle fat loss" or (start_body_fat > target_body_fat_val):
                     # Visual smoother: don't let it dip below target too aggressively in chart
                    current_fat_pct = max(target_body_fat_val - 2.0, current_fat_pct)

                weights.append(current_weight)
                body_fats.append(current_fat_pct)
                muscle_pcts.append(est_muscle_pct)

            weights = np.array(weights)
            body_fats = np.array(body_fats)
            muscle_pcts = np.array(muscle_pcts)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weeks,
                y=body_fats,
                name="Body Fat %",
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                yaxis="y1",
                hovertemplate="Week %{x}<br>Body Fat: %{y:.1f}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=weeks,
                y=muscle_pcts,
                name="Est. Muscle Mass %", 
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                yaxis="y1",
                hovertemplate="Week %{x}<br>Muscle: %{y:.1f}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=weeks,
                y=weights,
                name="Weight (kg)",
                mode="lines+markers",
                line=dict(color="#e67e22", width=2),
                yaxis="y2",
                hovertemplate="Week %{x}<br>Weight: %{y:.1f} kg<extra></extra>",
            ))

            fig.update_layout(
                title={
                    "text": f"Estimated {estimated_weeks}-Week Body Composition Progress",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 0.95,
                },
                xaxis=dict(
                    title="Week",
                    tickmode="array",
                    tickvals=weeks,
                    ticktext=week_labels,
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Body Fat % / Muscle %",
                    range=[0, max(70, float(body_fats.max()) + 5, float(muscle_pcts.max()) + 5)],
                    tickmode="linear",
                    dtick=5,
                    showgrid=True,
                    gridcolor="rgba(200,200,200,0.2)",
                ),
                yaxis2=dict(
                    title="Weight (kg)",
                    overlaying="y",
                    side="right",
                    range=[0, max(100, start_weight + 10)],
                    tickmode="linear",
                    dtick=5,
                    showgrid=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(l=60, r=80, t=80, b=60),
                height=420,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------- Explanation under the chart --------
            st.markdown("### How This Projection Is Calculated (Based on Your Inputs)")

            bmi_str = f"{bmi:.1f}" if bmi is not None else "N/A"
            
            # Display Deficit as a positive number for readability ("Deficit of 500")
            display_deficit = abs(weekly_energy_balance)
            display_fat_change = abs(weekly_fat_change_kg)

            st.markdown(
                f"""
**1. Your starting point**

- Gender: **{gender}**
- Age: **{age}**
- Height: **{height_cm} cm**, Weight: **{start_weight:.1f} kg**
- Current Body Fat: **{start_body_fat:.1f}%**, Target Body Fat: **{target_body_fat_val:.1f}%**
- **Current Muscle Mass:** **{start_muscle_mass:.1f}%**
- Activity Level: **{activity}**
- Primary Goal: **{goal}**

- **BMI:** {bmi_str}  
- **BMR (Basal Metabolic Rate):** ~**{bmr:.0f} kcal/day**  
- **Estimated TDEE (maintenance):** ~**{tdee:.0f} kcal/day**, based on your activity level
"""
            )

            st.markdown(
                f"""
**2. Calories, fat change, and time to target**

- **Target calories for your goal:** ~**{target_calories:.0f} kcal/day**  
- **Estimated Energy Balance:** ~**{weekly_energy_balance:.0f} kcal/week** (Negative = Deficit)
- Using ~**7,700 kcal ≈ 1 kg of body fat**, this gives an estimated fat change of  
**{weekly_fat_change_kg:.3f} kg per week** (Negative = Fat Loss).

From your starting and target body fat we estimate you need to change fat mass by about
**{fat_to_lose_kg:.2f} kg**, which corresponds to **≈ {weeks_to_target:.1f} weeks**.
For the chart, we use **{estimated_weeks} weeks** (capped if very long) to show your trajectory.
"""
            )

            st.markdown(
                """
**3. How to read the lines**

- **Body Fat % (red)**  
  Each week we adjust your weight using the estimated weekly fat change. We assume most weight change comes from fat stores (in a moderate deficit), and recalculate your body fat percentage accordingly.

- **Est. Muscle Mass % (blue)**  
  We assume you are training to **maintain** your muscle mass (kg) while losing fat. 
  As your total body weight goes down, your **Muscle Percentage** will naturally go **UP**.

- **Weight (kg) (orange)**  
  Your total body weight is updated week by week based on the estimated fat change.
"""
            )

            st.markdown(
                """
**4. Interpretation**

- This is a **coaching-style projection**, not a medical prediction.
- Real progress is usually **non-linear**.
- Use this as a guide:
  - Re-measure every **2–4 weeks**,
  - Compare real numbers to the projection,
  - Adjust calories, training, and recovery as needed.
"""
            )


    elif st.session_state.get("cached_results"):
        pass

if __name__ == "__main__":
    main()