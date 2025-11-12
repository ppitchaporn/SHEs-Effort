import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

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
    """Sends a prompt to the specified LLM model and returns the response."""
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
        col3, col4 = st.columns(2)
        with col3:
            activity = st.selectbox("Activity Level (daily life, not workouts)", ["Sedentary (desk job, little movement)", "Light (walks sometimes)", "Moderate (on feet often)", "Active (physical job or lots of movement)"])
        with col4:
            goal = st.selectbox("Goal", ["Feel better / Healthier", "Gentle fat loss", "Gentle muscle tone", "Reduce stress & move more"])

    # Create space before the button
    st.write("") 

    if st.button("✨ Generate My Plan!", use_container_width=True):
        # Create a unique key for session state based on all user inputs
        input_key = f"{gender}-{age}-{weight_kg}-{height_cm}-{body_fat}-{target_body_fat}-{activity}-{goal}-{selected_model}"

        # Check for cached results to avoid redundant API calls
        if "last_inputs" in st.session_state and st.session_state.last_inputs == input_key:
            st.markdown("Loaded from previous session")
            st.markdown(st.session_state.last_result)
        else:
            with st.spinner("Processing your personalized plan..."):
                prompt = f"""
You are a fitness and nutrition coach. Use the information below to create a clear final result only. Do NOT show python code or calculation steps. Return the result as a structured text, ready to display.

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
1. An estimation of the time required to reach the target body fat percentage, with an explanation.
2. A weekly Workout Plan (structure, warm-up, detailed at-home alternatives with exercise names, sets, and reps).
3. A Nutrition Guide (TDEE estimate, macros in grams, and examples of Thai food to meet protein targets).

**Important:** 
- Make it easy to read, similar to an infographic.
- Use bullet points.
- The response must be in Thai only. (ตอบเป็นภาษาไทยเท่านั้น)
"""
                result = call_llm(selected_model, prompt)
                st.session_state.last_inputs = input_key
                st.session_state.last_result = result
                st.markdown(result)
    
    # Display result if it exists in session state (for reruns after the button is pressed)
    elif "last_inputs" in st.session_state:
        st.markdown(st.session_state.last_result)

if __name__ == "__main__":
    main()