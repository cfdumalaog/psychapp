import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
from dotenv import dotenv_values

# --- 1. PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="MindfulAI Screening",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR "APP-LIKE" FEEL ---
st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the chat container */
    .stChatInput {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ROBUST API KEY LOADING ---
def load_api_key():
    # Looks for .env in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        env_vals = dotenv_values(env_path)
        key = env_vals.get("GEMINI_API_KEY", "")
        if key: return key.strip().replace("\n", "").replace("\r", "")
    return None

api_key = load_api_key()

if not api_key:
    st.error("🚨 CRITICAL: GEMINI_API_KEY not found in .env file.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- 4. SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    # We use a strict system prompt to force the AI to behave professionally
    st.session_state.chat_history = (
        "System: You are Dr. Gemini, an empathetic and professional psychological screening assistant. "
        "Your goal is to screen for Depression (PHQ-9) and Anxiety (GAD-7). "
        "RULES:\n"
        "1. Ask exactly ONE question at a time.\n"
        "2. Wait for the user to answer before asking the next one.\n"
        "3. Do not diagnose. Use phrases like 'The responses suggest'.\n"
        "4. If the user mentions self-harm, immediately provide emergency resources.\n"
    )

if "messages" not in st.session_state:
    # Add an initial greeting from the AI
    welcome_msg = "Hello. I am an AI Screening Assistant. I'm here to ask you a few questions about how you've been feeling lately. All answers are confidential. Shall we begin?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    st.session_state.chat_history += f"Assistant: {welcome_msg}\n"

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.success("System Online")
    
    with st.expander("ℹ️ Disclaimer"):
        st.caption("This tool is for screening purposes only and does not provide a medical diagnosis. If you are in crisis, please call emergency services.")

    if st.button("🔄 Reset Interview", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = "System: Start fresh.\n"
        st.session_state.report_generated = False
        st.rerun()

# --- 6. CHAT INTERFACE ---
st.title("🧠 MindfulAI Screener")

# Display chat messages with avatars
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🩺"): # Doctor icon
            st.write(msg["content"])
    else:
        with st.chat_message("user", avatar="👤"): # User icon
            st.write(msg["content"])

# --- 7. CHAT LOGIC ---
if not st.session_state.report_generated:
    user_input = st.chat_input("Type your answer here...")

    if user_input:
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history += f"User: {user_input}\n"
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        # 2. AI Response
        with st.chat_message("assistant", avatar="🩺"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(st.session_state.chat_history)
                    ai_text = response.text
                    message_placeholder.write(ai_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.session_state.chat_history += f"Assistant: {ai_text}\n"
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- 8. REPORT & ANALYTICS ---
if not st.session_state.report_generated:
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Ready to finish? Click the button to analyze the session.")
    with col2:
        finish_btn = st.button("📋 End & Analyze", type="primary", use_container_width=True)

    if finish_btn:
        if len(st.session_state.messages) < 3:
            st.toast("⚠️ Please answer a few more questions first!", icon="⚠️")
        else:
            with st.spinner("Generative AI is analyzing clinical markers..."):
                analysis_prompt = (
                    st.session_state.chat_history + 
                    "\n\nCOMMAND: The interview is over. Act as a Senior Clinical Analyst. "
                    "Analyze the conversation above and produce a structured Markdown report. "
                    "Include:\n"
                    "1. **Clinical Summary**: A brief paragraph observing the patient's state.\n"
                    "2. **Risk Assessment Table**: Columns for [Condition, Risk Level (Low/Med/High), Evidence]. Cover Depression, Anxiety, and Burnout.\n"
                    "3. **Recommendations**: 3 bullet points for next steps."
                )
                
                try:
                    report_resp = model.generate_content(analysis_prompt)
                    report_text = report_resp.text
                    st.session_state.report_generated = True
                    
                    # Force a rerun to hide the chat input and show the report
                    st.session_state.final_report = report_text 
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

# --- 9. DISPLAY REPORT (Post-Session) ---
if st.session_state.report_generated and "final_report" in st.session_state:
    st.success("Assessment Complete")
    
    with st.container(border=True):
        st.markdown(st.session_state.final_report)
    
    # Download Section
    csv_data = pd.DataFrame({
        "Timestamp": [pd.Timestamp.now()],
        "Report": [st.session_state.final_report],
        "Transcript": [st.session_state.chat_history]
    }).to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="💾 Download Clinical Record (CSV)",
        data=csv_data,
        file_name="clinical_assessment.csv",
        mime="text/csv",
        type="primary"
    )
    
    if st.button("Start New Patient"):
        st.session_state.clear()
        st.rerun()