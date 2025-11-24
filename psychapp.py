import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
from dotenv import dotenv_values

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MindfulAI Screening",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
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
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        env_vals = dotenv_values(env_path)
        key = env_vals.get("GEMINI_API_KEY", "")
        if key: return key.strip().replace("\n", "").replace("\r", "")
        
    return None

api_key = load_api_key()

if not api_key:
    st.error("🚨 CRITICAL: GEMINI_API_KEY not found in .env file or Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- 4. SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Changed to list for multimodal support
    # Add System Prompt as the first "user" message (hidden logic)
    system_prompt = (
        "System: You are Dr. Gemini, an empathetic and professional psychological screening assistant. "
        "Your goal is to screen for Depression (PHQ-9) and Anxiety (GAD-7). "
        "RULES:\n"
        "1. Ask exactly ONE question at a time.\n"
        "2. Wait for the user to answer before asking the next one.\n"
        "3. Do not diagnose. Use phrases like 'The responses suggest'.\n"
        "4. If the user mentions self-harm, immediately provide emergency resources.\n"
    )
    st.session_state.chat_history.append({"role": "user", "parts": [system_prompt]})
    st.session_state.chat_history.append({"role": "model", "parts": ["Understood. I am ready to begin the screening."]})

if "messages" not in st.session_state:
    welcome_msg = "Hello. I am an AI Screening Assistant. I'm here to ask you a few questions about how you've been feeling lately. You can type or use the microphone. Shall we begin?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    st.session_state.chat_history.append({"role": "model", "parts": [welcome_msg]})

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
        st.session_state.chat_history = [] 
        # Re-initialize system prompt logic handled in section 4 on rerun
        st.session_state.report_generated = False
        if "final_report_json" in st.session_state:
            del st.session_state.final_report_json
        st.rerun()

# --- 6. CHAT INTERFACE ---
st.title("🧠 MindfulAI Screener")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🩺"): 
            st.write(msg["content"])
    else:
        with st.chat_message("user", avatar="👤"): 
            st.write(msg["content"])

# --- 7. CHAT LOGIC (MULTIMODAL) ---
if not st.session_state.report_generated:
    
    # A. AUDIO INPUT
    audio_val = st.audio_input("🎙️ Record your answer")
    
    # B. TEXT INPUT
    text_val = st.chat_input("Type your answer here...")

    user_content = None
    input_type = None

    # Handle Input (Audio takes priority if both exist, usually one at a time)
    if audio_val:
        input_type = "audio"
        user_content = {
            "mime_type": "audio/wav",
            "data": audio_val.getvalue()
        }
        display_content = "🎤 *[Audio Message Sent]*"
    elif text_val:
        input_type = "text"
        user_content = text_val
        display_content = text_val

    # Process Input
    if user_content:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": display_content})
        with st.chat_message("user", avatar="👤"):
            st.write(display_content)
        
        # 2. Add to Gemini History
        if input_type == "audio":
            st.session_state.chat_history.append({"role": "user", "parts": [user_content]})
        else:
            st.session_state.chat_history.append({"role": "user", "parts": [user_content]})

        # 3. AI Response
        with st.chat_message("assistant", avatar="🩺"):
            message_placeholder = st.empty()
            with st.spinner("Dr. Gemini is listening..."):
                try:
                    # Send entire history (System prompt + conversation + new audio/text)
                    response = model.generate_content(st.session_state.chat_history)
                    ai_text = response.text
                    message_placeholder.write(ai_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.session_state.chat_history.append({"role": "model", "parts": [ai_text]})
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
                # Construct a text-only history for the analysis step to avoid complexity
                transcript_text = ""
                for msg in st.session_state.messages:
                    transcript_text += f"{msg['role'].upper()}: {msg['content']}\n"

                analysis_prompt = (
                    transcript_text + 
                    "\n\nCOMMAND: The interview is over. Act as a Senior Clinical Analyst. "
                    "Analyze the conversation transcript above. "
                    "Output a strictly valid JSON object with the following structure:\n"
                    "{\n"
                    '  "clinical_summary": "string paragraph",\n'
                    '  "risk_assessment": [\n'
                    '    {"Condition": "Depression (PHQ-9)", "Risk Level": "Low/Medium/High", "Evidence": "short text"},\n'
                    '    {"Condition": "Anxiety (GAD-7)", "Risk Level": "Low/Medium/High", "Evidence": "short text"},\n'
                    '    {"Condition": "Burnout", "Risk Level": "Low/Medium/High", "Evidence": "short text"}\n'
                    '  ],\n'
                    '  "recommendations": ["string", "string", "string"]\n'
                    "}"
                )
                
                try:
                    report_resp = model.generate_content(
                        analysis_prompt,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    st.session_state.final_report_json = report_resp.text
                    st.session_state.report_generated = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

# --- 9. DISPLAY REPORT (Post-Session) ---
if st.session_state.report_generated and "final_report_json" in st.session_state:
    try:
        report_data = json.loads(st.session_state.final_report_json)
        
        st.success("Assessment Complete")
        st.markdown("### 📄 Clinical Summary")
        st.info(report_data.get("clinical_summary", "No summary available."))
        
        st.markdown("### 📊 Risk Assessment Matrix")
        if "risk_assessment" in report_data:
            df = pd.DataFrame(report_data["risk_assessment"])
            st.table(df)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Risk Report (CSV)",
                data=csv_data,
                file_name="risk_assessment_matrix.csv",
                mime="text/csv",
                type="primary"
            )
        
        st.markdown("### 🩺 Recommendations")
        for rec in report_data.get("recommendations", []):
            st.write(f"- {rec}")
            
    except json.JSONDecodeError:
        st.error("Error parsing the AI report.")
        st.text(st.session_state.final_report_json)

    if st.button("Start New Patient"):
        st.session_state.clear()
        st.rerun()
