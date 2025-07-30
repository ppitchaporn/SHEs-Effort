import os
from dotenv import load_dotenv
from pathlib import Path
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.groq import Groq
import streamlit as st

# Load environment variables for API keys and configurations
load_dotenv()

# Set up the base directory for temporary files
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

# Path to the local IMDB CSV file
local_csv_path = "IMDB-Movie-Data.csv"

# Ensure the CSV file exists in the project directory
if not os.path.exists(local_csv_path):
    raise FileNotFoundError(f"The file {local_csv_path} does not exist. Please ensure it is placed in the project directory.")

# Configure the PythonAgent with Groq's Llama model and the local CSV file
python_agent = PythonAgent(
    #model=Groq(id="llama-3.1-70b-versatile"),  # Use Groq's Llama 3.1 70B Versatile model
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),  # Use Groq's Llama 3.1 70B Versatile model
    base_dir=tmp,  # Temporary directory for intermediate files
    files=[
        CsvFile(
            path=local_csv_path,  # Integrate the IMDB CSV file
            description="A dataset containing information about movies, genres, directors, and IMDB ratings.",
        )
    ],
    markdown=True,  # Enable markdown formatting for responses
    pip_install=True,  # Install required dependencies automatically
    show_tool_calls=True,  # Display tool calls for better transparency
)

# Main function for the Streamlit app
def main():
    st.header("Step 1: Collect Personal Data (for TDEE & Goals)")

    # Gender at the top
    gender = st.radio(
        "Gender",
        ["Male", "Female", "Other / Prefer not to say"],
        horizontal=True
    )

    # Personal details: 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Age",
            options=list(range(10, 101)),
            index=20  # default to 30
        )
        body_fat = st.selectbox(
            "Current Body Fat % (optional)",
            options=[round(x * 0.5, 1) for x in range(0, 201)],
            index=40  # default 20.0%
        )
        target_body_fat = st.selectbox(
            "Target Body Fat % (optional)",
            options=[round(x * 0.5, 1) for x in range(0, 201)],
            index=30  # default 15.0%
        )

    with col2:
        weight_kg = st.selectbox(
            "Weight (kg)",
            options=[round(w, 1) for w in range(30, 151)],
            index=30  # default 60.0
        )
        height_cm = st.selectbox(
            "Height (cm)",
            options=list(range(120, 221)),
            index=45  # default 165
        )

    # Activity and Goal in next row
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

    # Spacer
    st.markdown("---")

    # Action button
    # Unique key for the current user input combination
    input_key = f"{gender}-{age}-{weight_kg}-{height_cm}-{body_fat}-{target_body_fat}-{activity}-{goal}"

    # Check for cached result
    if "last_inputs" in st.session_state and st.session_state.last_inputs == input_key:
        st.markdown("✅ Loaded from previous session")
        st.markdown(st.session_state.last_result)
    else:
        if st.button("Calculate and Run"):
            try:
                with st.spinner("Processing your personalized plan..."):

                    # Prompt
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
    3. 🍱 Nutrition Guide (TDEE estimate, macros in grams, 
    and examples of how to hit the protein target with food items like chicken, eggs, milk, tofu, whey, etc. บอกตัวอย่างอาหาร เน้นอาหารไทย)

    **Important:** 
    - Make it short and easy to read, similar to a fitness infographic.
    - Use emojis and bullet points.
    - Do not output python code or calculation steps, only the final readable guide.
    ตอบเป็นภาษาไทยเท่านั้น
    """

                    # Run agent
                    response = python_agent.run(prompt)
                    st.markdown(response.content)

                    # Save in session state
                    st.session_state.last_inputs = input_key
                    st.session_state.last_result = response.content

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
