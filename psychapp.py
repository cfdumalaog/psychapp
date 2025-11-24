import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
import io
import time
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
    /* 1. Force Light/Calming Theme */
    .stApp {
        background-color: #FDFBF7; /* Soft Cream */
        color: #31333F !important;
    }
    
    /* 2. Chat Bubbles */
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* 3. Text Colors */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #31333F !important;
    }

    /* 4. Button Styles */
    .stButton > button {
        background-color: #E0F2F1;
        color: #004D40 !important;
        border: 1px solid #B2DFDB;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #B2DFDB;
    }
    
    /* 5. Hide standard elements */
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
# Use Flash for fast reasoning
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- 4. SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
    # SYSTEM PROMPT
    system_prompt = (
        "System: You are Dr. Gemini, an empathetic psychological screening assistant. "
        "Your goal is to screen for Depression (PHQ-9) and Anxiety (GAD-7). "
        "PROTOCOL:\n"
        "1. Ask exactly ONE question at a time.\n"
        "2. Do not diagnose.\n"
        "3. Keep responses concise (2-3 sentences max) to allow for spoken conversation."
    )
    st.session_state.chat_history.append({"role": "user", "parts": [system_prompt]})
    st.session_state.chat_history.append({"role": "model", "parts": ["Understood."]})

if "messages" not in st.session_state:
    welcome_msg = "Hello. I am here to listen. How have you been feeling lately?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    st.session_state.chat_history.append({"role": "model", "parts": [welcome_msg]})

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🍃 Controls")
    if st.button("🔄 Start Fresh", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = [] 
        st.session_state.report_generated = False
        st.session_state.last_processed_audio = None
        if "final_report_json" in st.session_state:
            del st.session_state.final_report_json
        st.rerun()

# --- 6. HELPER FUNCTIONS ---
def transcribe_audio(audio_bytes):
    """Explicitly transcribe audio to text using Gemini before processing."""
    try:
        # We make a separate call just to get the text, ensuring accuracy
        transcribe_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = transcribe_model.generate_content([
            "Transcribe the spoken speech in this audio exactly. Output ONLY the text.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        return f"[Error in Transcription: {e}]"

def text_to_speech_autoplay(text):
    """Generates audio and returns the bytes for autoplay"""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None

# --- 7. MAIN INTERFACE ---
st.title("🍃 MindfulAI Screener")

# Display History
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🌿"): 
            st.write(msg["content"])
            if "audio" in msg and msg["audio"] is not None:
                 st.audio(msg["audio"], format="audio/mp3", start_time=0)
    else:
        with st.chat_message("user", avatar="👤"): 
            st.write(msg["content"])

# --- 8. INPUT LOGIC ---
if not st.session_state.report_generated:
    
    # We put inputs in a container at the bottom
    with st.container():
        # Audio Input
        audio_val = st.audio_input("🎙️ Tap to Speak (Auto-Transcribe)")
        # Text Input
        text_val = st.chat_input("Or type your answer...")

    user_text = ""
    is_audio = False

    # A. HANDLE AUDIO
    # We check if audio_val exists and if it's different from the last one we processed
    if audio_val and audio_val != st.session_state.last_processed_audio:
        st.session_state.last_processed_audio = audio_val
        is_audio = True
        
        with st.spinner("🎧 Transcribing your voice..."):
            audio_bytes = audio_val.getvalue()
            # 1. Transcribe First (Fixes "Not Recording" issue)
            transcript = transcribe_audio(audio_bytes)
            user_text = transcript
            
    # B. HANDLE TEXT
    elif text_val:
        user_text = text_val

    # C. PROCESS RESPONSE
    if user_text:
        # 1. Display User Msg
        display_text = f"🎙️ {user_text}" if is_audio else user_text
        st.session_state.messages.append({"role": "user", "content": display_text})
        
        # Add to Gemini History (Using Text ensures clarity for the model)
        st.session_state.chat_history.append({"role": "user", "parts": [user_text]})
        
        # Force a UI update immediately to show user text
        st.rerun()

# --- 9. AI REPLY GENERATION (Runs on Rerun) ---
# Check if the last message was from User, if so, generate AI reply
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Dr. Gemini is thinking..."):
            try:
                # 1. Get Text Response
                response = model.generate_content(st.session_state.chat_history)
                ai_text = response.text
                
                # 2. Generate Voice
                audio_bytes = text_to_speech_autoplay(ai_text)
                
                # 3. Save & Display
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_text,
                    "audio": audio_bytes
                })
                st.session_state.chat_history.append({"role": "model", "parts": [ai_text]})
                
                # 4. Auto-Play (By rerunning, the new message with audio will be rendered at top)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- 10. REPORT SECTION ---
if not st.session_state.report_generated:
    st.markdown("---")
    if st.button("📋 End Session & Generate Report", type="primary", use_container_width=True):
        with st.spinner("Analyzing session..."):
            # Prepare transcript
            full_transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            
            prompt = (
                full_transcript + 
                "\n\nCOMMAND: Act as a Senior Clinical Analyst. "
                "Analyze the transcript above. Output strictly valid JSON:\n"
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
                resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                st.session_state.final_report_json = resp.text
                st.session_state.report_generated = True
                st.rerun()
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- 11. REPORT DISPLAY ---
if st.session_state.report_generated and "final_report_json" in st.session_state:
    try:
        data = json.loads(st.session_state.final_report_json)
        st.success("Analysis Complete")
        st.info(f"**Clinical Summary:** {data.get('clinical_summary', 'N/A')}")
        
        if "risk_assessment" in data:
            df = pd.DataFrame(data["risk_assessment"])
            st.table(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download CSV", csv, "assessment.csv", "text/csv")
            
    except:
        st.error("Could not parse report.")
        st.write(st.session_state.final_report_json)
    
    if st.button("Start New Patient"):
        st.session_state.clear()
        st.rerun()
