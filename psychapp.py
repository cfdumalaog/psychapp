import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
import io
from dotenv import dotenv_values
from gtts import gTTS

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MindfulAI Screening",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. THERAPY THEME & CUSTOM CSS ---
st.markdown("""
    <style>
    /* 1. Force Light/Calming Theme Background & Dark Text */
    .stApp {
        background-color: #FDFBF7; /* Soft Cream */
        color: #31333F !important; /* Force Dark Grey Text */
    }
    
    /* 2. Style Chat Messages */
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        color: #31333F !important; /* Force text color inside bubbles */
    }
    
    /* 3. Force Markdown text inside bubbles to be dark */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #31333F !important;
    }

    /* 4. Custom Button Styles (Teal/Sage) */
    .stButton > button {
        background-color: #E0F2F1;
        color: #004D40 !important;
        border: 1px solid #B2DFDB;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #B2DFDB;
        color: #004D40 !important;
        border-color: #004D40;
    }
    
    /* 5. Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #263238 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 6. Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    st.error("🚨 CRITICAL: GEMINI_API_KEY not found.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- 4. SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
    # SYSTEM PROMPT
    # We explicitly ask the model to transcribe audio if present.
    system_prompt = (
        "System: You are Dr. Gemini, an empathetic psychological screening assistant. "
        "Your goal is to screen for Depression (PHQ-9) and Anxiety (GAD-7). "
        "PROTOCOL:\n"
        "1. If the user sends AUDIO, start your response with 'TRANSCRIPT: [what they said]'.\n"
        "2. Ask exactly ONE question at a time.\n"
        "3. Do not diagnose.\n"
    )
    st.session_state.chat_history.append({"role": "user", "parts": [system_prompt]})
    st.session_state.chat_history.append({"role": "model", "parts": ["Understood. I am ready."]})

if "messages" not in st.session_state:
    welcome_msg = "Hello. I am here to listen. How have you been feeling lately? (You can type or speak)"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    st.session_state.chat_history.append({"role": "model", "parts": [welcome_msg]})

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🍃 Controls")
    
    with st.expander("ℹ️ About this Session"):
        st.caption("This is a safe, confidential space for screening. Not a replacement for professional help.")

    if st.button("🔄 Start Fresh", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = [] 
        st.session_state.report_generated = False
        if "final_report_json" in st.session_state:
            del st.session_state.final_report_json
        st.rerun()

# --- 6. CHAT INTERFACE ---
st.title("🍃 MindfulAI Screener")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🌿"): 
            st.write(msg["content"])
            # If there was audio associated with this message, play it
            if "audio" in msg:
                 st.audio(msg["audio"], format="audio/mp3", start_time=0)
    else:
        with st.chat_message("user", avatar="👤"): 
            st.write(msg["content"])

# --- 7. CHAT LOGIC (MULTIMODAL + TTS) ---
if not st.session_state.report_generated:
    
    # A. INPUTS
    audio_val = st.audio_input("🎙️ Speak your answer")
    text_val = st.chat_input("Or type your answer here...")

    user_content = None
    input_type = None
    display_content = ""

    # B. HANDLE INPUTS
    if audio_val:
        input_type = "audio"
        audio_bytes = audio_val.getvalue()
        user_content = {
            "mime_type": "audio/wav",
            "data": audio_bytes
        }
        display_content = "🔊 *Audio received... processing transcript...*"
        
    elif text_val:
        input_type = "text"
        user_content = text_val
        display_content = text_val

    # C. PROCESS INTERACTION
    if user_content:
        # 1. Show User Message Immediately
        st.session_state.messages.append({"role": "user", "content": display_content})
        with st.chat_message("user", avatar="👤"):
            st.write(display_content)
        
        # 2. Add to History
        st.session_state.chat_history.append({"role": "user", "parts": [user_content]})

        # 3. Generate AI Response
        with st.chat_message("assistant", avatar="🌿"):
            message_placeholder = st.empty()
            
            with st.spinner("Listening & Thinking..."):
                try:
                    # Call Gemini
                    response = model.generate_content(st.session_state.chat_history)
                    full_response = response.text
                    
                    # 4. Parse Transcript (if audio was used)
                    clean_response = full_response
                    transcript_text = ""
                    
                    if "TRANSCRIPT:" in full_response:
                        parts = full_response.split("TRANSCRIPT:")
                        if len(parts) > 1:
                            temp = parts[1].split("\n", 1)
                            transcript_text = temp[0].strip()
                            if len(temp) > 1:
                                clean_response = temp[1].strip()
                            else:
                                clean_response = "" 
                    
                    # 5. Display Updates
                    if transcript_text:
                        st.info(f"📝 I heard you say: \"{transcript_text}\"")
                        st.session_state.messages[-1]["content"] = f"🎙️ \"{transcript_text}\""
                        
                    message_placeholder.write(clean_response)
                    
                    # 6. Generate Voice (TTS)
                    if clean_response:
                        try:
                            # Using gTTS for voice generation
                            tts = gTTS(text=clean_response, lang='en')
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            
                            # Play Audio
                            st.audio(audio_fp, format='audio/mp3', autoplay=True)
                            
                            # Save to state
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": clean_response,
                                "audio": audio_fp
                            })
                            st.session_state.chat_history.append({"role": "model", "parts": [clean_response]})
                            
                        except Exception as e:
                            st.warning(f"Could not generate voice: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": clean_response})
                            st.session_state.chat_history.append({"role": "model", "parts": [clean_response]})

                except Exception as e:
                    st.error(f"API Error: {e}")

# --- 8. REPORT GENERATION ---
if not st.session_state.report_generated:
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Ready to view results?")
    with col2:
        finish_btn = st.button("📋 Generate Report", type="primary", use_container_width=True)

    if finish_btn:
        with st.spinner("Analyzing session data..."):
            transcript_text = ""
            for msg in st.session_state.messages:
                content = msg.get('content', '')
                transcript_text += f"{msg['role'].upper()}: {content}\n"

            analysis_prompt = (
                transcript_text + 
                "\n\nCOMMAND: Act as a Senior Clinical Analyst. "
                "Analyze the transcript. Output strictly valid JSON:\n"
                "{\n"
                '  "clinical_summary": "string",\n'
                '  "risk_assessment": [\n'
                '    {"Condition": "Depression", "Risk": "Low/Med/High", "Notes": "text"},\n'
                '    {"Condition": "Anxiety", "Risk": "Low/Med/High", "Notes": "text"},\n'
                '    {"Condition": "Burnout", "Risk": "Low/Med/High", "Notes": "text"}\n'
                '  ],\n'
                '  "recommendations": ["string", "string"]\n'
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

# --- 9. DISPLAY REPORT ---
if st.session_state.report_generated and "final_report_json" in st.session_state:
    try:
        report_data = json.loads(st.session_state.final_report_json)
        
        st.success("Analysis Complete")
        st.markdown("### 📄 Clinical Summary")
        st.info(report_data.get("clinical_summary", "N/A"))
        
        st.markdown("### 📊 Risk Assessment")
        if "risk_assessment" in report_data:
            df = pd.DataFrame(report_data["risk_assessment"])
            st.table(df)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download CSV", csv_data, "assessment.csv", "text/csv")
            
    except Exception:
        st.error("Report Parsing Error")
        st.text(st.session_state.final_report_json)

    if st.button("Start New Patient"):
        st.session_state.clear()
        st.rerun()
