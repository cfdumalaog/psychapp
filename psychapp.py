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
    initial_sidebar_state="collapsed" # Hide sidebar by default for cleaner look
)

# --- 2. CSS STYLING (Voice Mode & Chat Mode) ---
st.markdown("""
    <style>
    /* GLOBAL THEME */
    .stApp {
        background-color: #FDFBF7; /* Soft Cream */
        color: #31333F !important;
    }
    
    /* HIDE DEFAULT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* CHAT MODE STYLES */
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* VOICE MODE ORB ANIMATION */
    @keyframes breathe {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(100, 181, 246, 0.2); }
        50% { transform: scale(1.1); box-shadow: 0 0 50px rgba(100, 181, 246, 0.5); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(100, 181, 246, 0.2); }
    }
    
    .voice-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        margin-top: 50px;
    }
    
    .voice-orb {
        width: 180px;
        height: 180px;
        /* Beautiful abstract gradient like ChatGPT */
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
        border-radius: 50%;
        animation: breathe 4s ease-in-out infinite;
        margin-bottom: 40px;
    }
    
    .voice-status {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 18px;
        color: #546E7A;
        font-weight: 500;
        margin-bottom: 20px;
        text-align: center;
    }

    /* CENTER AUDIO INPUT WIDGET IN VOICE MODE */
    div[data-testid="stAudioInput"] {
        margin: 0 auto;
        width: 100%;
        max-width: 400px;
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
    st.error("🚨 CRITICAL: GEMINI_API_KEY not found.")
    st.stop()

genai.configure(api_key=api_key)
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
        "3. Keep responses concise (1-2 sentences) suitable for voice conversation."
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

# NEW: MODE SWITCH STATE
if "mode" not in st.session_state:
    st.session_state.mode = "chat" # Options: 'chat', 'voice'

# --- 5. HELPER FUNCTIONS ---
def transcribe_audio(audio_bytes):
    try:
        transcribe_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = transcribe_model.generate_content([
            "Transcribe this audio exactly. Output ONLY the text.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        return f"[Transcription Error]"

def text_to_speech_autoplay(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None

def process_ai_response():
    """Common logic to get AI response and generate speech"""
    try:
        response = model.generate_content(st.session_state.chat_history)
        ai_text = response.text
        audio_bytes = text_to_speech_autoplay(ai_text)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_text,
            "audio": audio_bytes
        })
        st.session_state.chat_history.append({"role": "model", "parts": [ai_text]})
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# ==========================================
# VIEW 1: VOICE MODE (ChatGPT Style)
# ==========================================
if st.session_state.mode == "voice":
    
    # 1. The Visuals (Orb)
    st.markdown("""
        <div class="voice-container">
            <div class="voice-orb"></div>
            <div class="voice-status">Dr. Gemini is listening...</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. The Input (Centered Audio Widget)
    audio_val = st.audio_input("🎙️ Tap to Speak")
    
    # 3. Handle Voice Input
    if audio_val and audio_val != st.session_state.last_processed_audio:
        st.session_state.last_processed_audio = audio_val
        
        # A. Transcribe
        with st.spinner("Processing voice..."):
            audio_bytes = audio_val.getvalue()
            transcript = transcribe_audio(audio_bytes)
            
            # Save User Input
            st.session_state.messages.append({"role": "user", "content": f"🎙️ {transcript}"})
            st.session_state.chat_history.append({"role": "user", "parts": [transcript]})
            
            # B. Generate AI Response
            if process_ai_response():
                # Force rerun to play the audio immediately
                st.rerun()

    # 4. Play Latest AI Audio (Auto-play in voice mode)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if "audio" in last_msg and last_msg["audio"]:
             st.audio(last_msg["audio"], format="audio/mp3", autoplay=True)

    # 5. Exit Button (Bottom)
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("❌ Exit Voice Mode", use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()


# ==========================================
# VIEW 2: STANDARD CHAT MODE
# ==========================================
else: # st.session_state.mode == "chat"
    
    st.title("🍃 MindfulAI Screener")
    
    # 1. Mode Switcher (Top Rightish)
    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🎙️ Voice Mode", type="primary"):
            st.session_state.mode = "voice"
            st.rerun()

    # 2. Display History
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🌿"): 
                st.write(msg["content"])
                # Optional: Small play button if audio exists, but don't autoplay in chat mode
                if "audio" in msg and msg["audio"]:
                     st.audio(msg["audio"], format="audio/mp3", start_time=0)
        else:
            with st.chat_message("user", avatar="👤"): 
                st.write(msg["content"])

    # 3. Text Input Logic
    if not st.session_state.report_generated:
        text_val = st.chat_input("Type your answer here...")
        
        if text_val:
            st.session_state.messages.append({"role": "user", "content": text_val})
            st.session_state.chat_history.append({"role": "user", "parts": [text_val]})
            
            with st.chat_message("user", avatar="👤"):
                st.write(text_val)

            with st.chat_message("assistant", avatar="🌿"):
                with st.spinner("Thinking..."):
                    if process_ai_response():
                        st.rerun()

    # 4. Report Generation (Only visible in chat mode)
    if not st.session_state.report_generated:
        st.markdown("---")
        if st.button("📋 End Session & Generate Report", type="secondary", use_container_width=True):
            with st.spinner("Analyzing session..."):
                transcript_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
                prompt = (
                    transcript_text + 
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

    # 5. Display Report
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
        
        if st.button("Start New Patient"):
            st.session_state.clear()
            st.rerun()

# --- SIDEBAR (Shared) ---
with st.sidebar:
    st.title("🍃 Controls")
    if st.button("🔄 Reset App", type="secondary"):
        st.session_state.clear()
        st.rerun()
