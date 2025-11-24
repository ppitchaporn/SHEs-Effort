import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# --- Configuration and Setup ---

# Set page configuration for a wider layout and a title
st.set_page_config(layout="wide", page_title="Fitness Planner")

# Load environment variables from a .env file
load_dotenv()
API_KEY = os.getenv("SCGC_API_KEY")

# API Endpoints
MODELS_URL = "https://scgc-llmproxy.scg.com/models"
CHAT_URL = "https://scgc-llmproxy.scg.com/v1/chat/completions"

# Common headers for API requests
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- API Functions ---

@st.cache_data(ttl=3600) # Cache the model list for 1 hour
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
    """Sends a prompt to the specified LLM model and returns the response.

    NOTE: Streamlit's `@st.cache_data` decorator was removed to avoid the
    automatic "Running call_llm(...)" status message in the UI. Caching is
    handled manually via `st.session_state['cached_results']` elsewhere in
    the app (per-session caching).
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
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
def get_html_generation_prompt(gender, age, weight_kg, height_cm, body_fat,target_body_fat,activity,goal,frequency): 
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
    - **Body Fat to Lose/Gain:** Calculate the amount of fat to lose/gain to reach the target body fat percentag


**Weekly Exercise Plan Overview:**
- **Exercise Days:** Based on a frequency of '{frequency}'.
- **Weekly workout plan:** structure, warm-up, detailed at-home alternatives with exercise names, sets, and reps.

**Daily Routine Snapshot (Example Day):**
- **Morning:** Include a healthy breakfast example.
- **Midday:** Include a lunch example.
- **Evening:** Workout example.
- **Hydration:** State a clear target (e.g., 2-2.5 L per day).

**Nutrition Highlights:**
- **Daily Calorie Target:** State the calculated value.
- **Macro Focus:** Describe a focus on high protein, moderate carbs, and healthy fats.
    - **Sample Meals:** Provide 1-2 brief examples of meals/snacks that fit each macro focus.
- **Guidelines:** Provide 5 simple meals in Thai style (เมนูอาหารไทย).

**Goal Tracking:**
- **Short-term Goals (4 weeks):** List 2-3 specific goals (e.g., walk 8,000 steps/day).
- **Medium-term Goals (3 months):** List 2-3 broader outcome goals.
- **Estimated time to reach target body fat:** [Calculate Value] weeks based on current rate of change.

**Footer:**
- **Explanation what's TDEE and Macros are.
- **Motivation:** Include a friendly motivational message to encourage the user on their fitness journey.

---
### **Part 2: Design & Style Specification (Replicate This Exactly)**

**Overall Page & Layout:**
- **Main Container (`.page`):** Wrap the infographic in a `div.page` with `max-width: 1200px`, `border-radius: 24px`, a soft `box-shadow`, and a semi-transparent background with `backdrop-filter: blur(10px)`.
- **Layout:** Use CSS Flexbox and CSS Grid for a responsive two-column layout for the main content.
- **Typography:** Use a `system-ui` font stack.

**Content language & Tone:**
- **Use friendly, encouraging, and motivational language throughout the infographic.
- **Incorporate emojis subtly to enhance engagement (e.g., ❤️ for health, 💪 for exercise).


### **Part 3: Responsive Design (VERY IMPORTANT)**
- **Mobile-First Layout:** The design must look excellent on both mobile and desktop screens.
- **Viewport Meta Tag:** You MUST include `<meta name="viewport" content="width=device-width, initial-scale=1.0">` in the `<head>` of the HTML.
- **Fluid Layout:** Use CSS Grid or Flexbox for the main layout. For example, the main content grid should be two columns on desktop.
- **Media Query for Stacking:** You MUST include a `@media (max-width: 768px)` query. Inside this query, change the multi-column grid layout to a single-column layout, so that the cards stack vertically on smaller screens.
- **Relative Units:** Use `rem` for font sizes and `%` for widths where possible to ensure elements scale gracefully.


**Color Palette (Use these CSS Variables in a `:root` block):**
- `--accent-teal: #1abc9c;`
- `--accent-blue: #3498db;`
- `--accent-green: #2ecc71;`
- `--text-main: #1f2933;`
- `--text-muted: #6b727a;`
- `--border-soft: #dde5ec;`

**Components:**
- **Cards (`.card`):** Use cards with large rounded corners (`18px`) and a subtle `box-shadow`. On hover, they must lift slightly (`transform: translateY(-2px)`).
- **Pills & Badges (`.pill`, `.chip`):** Use fully rounded (`border-radius: 999px`) elements for metadata.
- **Icons:** Use emojis or simple inline SVGs for icons (❤️ for health, 💪 for exercise, etc.).


**Text Styles:**
- **Headings:** Use larger, bold fonts for headings (e.g., `2rem` for main headings, `1.5rem` for subheadings).
- **Paragraphs:** Use `1rem` font size for body text with `line-height: 1.6` for readability.
- **All text must have sufficient contrast against the background for accessibility.
- **All text must be easily readable on both mobile and desktop devices.

---
### **Part 3: Technical & Code Requirements**

- **Self-Contained:** ALL CSS must be in a `<style>` tag and ALL JS in a `<script>` tag. No external files.
- **No Libraries:** Do not use any external CSS or JS libraries (No D3.js, Chart.js, etc.).
- **JavaScript:** The script must use an initialization guard (`if (!window.infographicInitialized)`) and a debounced resize handler for the chart.

---
### **Final Output Format**

Generate the complete HTML code as a single block. Do not add any conversational text, explanations, or markdown fences like ` ```html ` around the final code. Your response must be ready to be saved directly as a `.html` file.
"""

# --- Streamlit UI ---


def main():
    """Defines the main UI and logic for the Streamlit application."""
    st.title("✨ Your Personalized Fitness Planner ✨")
    st.write("Tell us a bit about yourself and your goals, and we'll create a plan just for you!")

    # Fetch available models and create a dropdown selector
    models = list_models()
    if not models:
        st.warning("Could not load LLM models. The application cannot proceed.")
        return

    # Set the default model to 'GPT-4o' if available, otherwise default to the first model
    default_model_name = "GPT-4o"
    default_index = 0
    if default_model_name in models:
        default_index = models.index(default_model_name)
    
    # --- Input Section ---
    # Group all inputs inside a main container for a cleaner look
    with st.container(border=True):
        selected_model = st.selectbox("Select LLM Model", models, index=default_index)
        st.divider()

        # Group personal data inputs
        st.subheader("Personal Details")
        gender = st.radio("Gender", ["Male", "Female", "Other / Prefer not to say"], horizontal=True)
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.selectbox("Age", options=list(range(20, 65)), index=10) # Default age 30
            
            # Options for body fat percentage from 5.0% to 40.0%
            fat_options = [round(x * 0.5, 1) for x in range(10, 81)]
            body_fat = st.selectbox("Current Body Fat %", options=fat_options, index=30) # Default 20%
            target_body_fat = st.selectbox("Target Body Fat %", options=fat_options, index=20) # Default 15%
        
        with col2:
            # Create weight options with 0.5 increments by doubling the range and dividing by 2
            weight_options = [w * 0.5 for w in range(80, 301)] # This creates values from 40.0, 40.5, ..., 150.0
            weight_kg = st.selectbox("Weight (kg)", options=weight_options, index=40) # Default 60kg

            height_cm = st.selectbox("Height (cm)", options=list(range(140, 201)), index=25) # Default 165cm

        st.divider()

        # Group goal-related inputs
        st.subheader("Activity & Goals")
        col3, col4, col5 = st.columns(3)
        with col3:
            activity = st.selectbox("Activity Level (daily life, not workouts)", ["Sedentary (desk job, little movement)", "Light (walks sometimes)", "Moderate (on feet often)", "Active (physical job or lots of movement)"])
        with col4:
            goal = st.selectbox("Goal", ["Feel better / Healthier", "Gentle fat loss", "Gentle muscle tone", "Reduce stress & move more"])
        with col5:
            frequency = st.selectbox("Workout Frequency", ["1-2 times a week", "3-4 times a week", "5-6 times a week", "Every day"])

    # Create space before the button
    st.write("") 

    if st.button("✨ Generate My Plan!", use_container_width=True):
        # Create a unique, normalized key for caching based on all user inputs
        # Normalize numeric values to a consistent string format to avoid
        # mismatches like '60' vs '60.0'.
        input_key = (
            f"{gender}-{int(age)}-{float(weight_kg):.1f}-{int(height_cm)}-"
            f"{float(body_fat):.1f}-{float(target_body_fat):.1f}-"
            f"{activity}-{goal}-{selected_model}--{frequency}"
        )

        # Use a dict stored in session_state to cache multiple different inputs
        cached = st.session_state.get("cached_results", {})
    
        html_content_to_display = None # Initialize a variable to hold our HTML


        if input_key in cached:
            st.success("Loaded from cache!")
            html_content_to_display = cached[input_key]
        else:
            with st.spinner("Processing your personalized plan..."):
                prompt = get_html_generation_prompt(gender, age, weight_kg, height_cm, body_fat, target_body_fat, activity, goal, frequency)
                
                html_result = call_llm(selected_model, prompt)

                # Clean the result to ensure it's just HTML
                # LLMs sometimes wrap the code in ```html ... ```
                if html_result.strip().startswith("```html"):
                    html_result = html_result.strip()[7:-3].strip()
                # Cache the *clean* result
                cached[input_key] = html_result
                st.session_state["cached_results"] = cached
                
                html_content_to_display = html_result
                # --- THIS IS THE CRITICAL DISPLAY PART ---
                # Use the components.html function to render the result
                # Adjust height as needed, or allow scrolling.
                if html_content_to_display:
                        # Wrap the generated HTML in a container and inject a client-side
                        # download button that uses html2canvas to capture the infographic
                        # element as a PNG. This runs in the user's browser (no server
                        # dependencies) and provides a direct image download.
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

                
                # result = call_llm(selected_model, prompt)
                # ... cache the result ...
                # st.markdown(html_result, unsafe_allow_html=True) # Using unsafe_allow_html can help render tables better.
                # # store result in the session cache map
                # cached[input_key] = html_result
                # st.session_state["cached_results"] = cached
                # st.markdown(html_result)

    # Optionally show previously cached result if available when page loads
    elif st.session_state.get("cached_results"):
        # Do nothing by default; cached results are shown when the user clicks
        # the button with matching inputs. You could add a dropdown here to
        # select and view older cached results if desired.
        pass    

if __name__ == "__main__":
    main()