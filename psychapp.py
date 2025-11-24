import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
import io
import asyncio
import edge_tts
from dotenv import dotenv_values
from gtts import gTTS

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MindfulAI Screening",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #FDFBF7 !important; color: #31333F !important; }
    .stApp p, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp label { color: #31333F !important; }
    .stChatMessage { background-color: #FFFFFF !important; border: 1px solid #E5E5E5; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .stChatMessage * { color: #31333F !important; }
    .stButton > button { background-color: #E0F2F1 !important; color: #004D40 !important; border: 1px solid #B2DFDB !important; border-radius: 8px; padding: 10px 20px; font-weight: 500; }
    .stButton > button:hover { background-color: #B2DFDB !important; border-color: #004D40 !important; }
    .stTextInput > div > div > input { color: #31333F !important; background-color: #FFFFFF !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Voice Mode Animation */
    @keyframes breathe {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(100, 181, 246, 0.2); }
        50% { transform: scale(1.1); box-shadow: 0 0 50px rgba(100, 181, 246, 0.5); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(100, 181, 246, 0.2); }
    }
    .voice-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; margin-top: 50px; }
    .voice-orb { width: 150px; height: 150px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 50%; animation: breathe 4s ease-in-out infinite; margin-bottom: 40px; }
    .voice-status { font-size: 18px; color: #546E7A !important; font-weight: 500; text-align: center; }
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

if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# --- 5. ADVANCED AUDIO FUNCTIONS (FIXED) ---

async def generate_neural_voice(text):
    """Generates Human-Like Audio using Edge TTS (Free Neural Voice)"""
    voice = "en-US-AriaNeural" 
    communicate = edge_tts.Communicate(text, voice)
    audio_fp = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_fp.write(chunk["data"])
    audio_fp.seek(0)
    return audio_fp

def get_audio_bytes(text):
    """
    Robust Wrapper for Async TTS. 
    Uses asyncio.run() which creates a fresh loop for the operation,
    avoiding 'Loop is closed' or 'Task attached to different loop' errors.
    """
    try:
        # Attempt 1: Human-Like Voice
        return asyncio.run(generate_neural_voice(text))
    except Exception as e:
        print(f"⚠️ Edge TTS Failed: {e} | Falling back to Standard Voice.")
        # Attempt 2: Robotic Fallback
        try:
            tts = gTTS(text=text, lang='en')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp
        except:
            return None

def transcribe_audio(audio_bytes):
    try:
        transcribe_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = transcribe_model.generate_content([
            "Transcribe this audio exactly. Output ONLY the text.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text.strip()
    except Exception:
        return "[Unintelligible Audio]"

def process_ai_response():
    try:
        response = model.generate_content(st.session_state.chat_history)
        ai_text = response.text
        
        # Generate Audio
        audio_bytes = get_audio_bytes(ai_text)
        
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
# VIEW 1: VOICE MODE
# ==========================================
if st.session_state.mode == "voice":
    
    st.markdown("""
        <div class="voice-container">
            <div class="voice-orb"></div>
            <div class="voice-status">Dr. Gemini is listening...</div>
        </div>
    """, unsafe_allow_html=True)
    
    audio_val = st.audio_input("🎙️ Tap to Speak")
    
    if audio_val and audio_val != st.session_state.last_processed_audio:
        st.session_state.last_processed_audio = audio_val
        
        with st.spinner("Thinking..."):
            audio_bytes = audio_val.getvalue()
            transcript = transcribe_audio(audio_bytes)
            
            st.session_state.messages.append({"role": "user", "content": f"🎙️ {transcript}"})
            st.session_state.chat_history.append({"role": "user", "parts": [transcript]})
            
            if process_ai_response():
                st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if "audio" in last_msg and last_msg["audio"]:
             st.audio(last_msg["audio"], format="audio/mp3", autoplay=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("❌ Exit Voice Mode", use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()

# ==========================================
# VIEW 2: CHAT MODE
# ==========================================
else: 
    st.title("🍃 MindfulAI Screener")
    
    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🎙️ Voice Mode", type="primary"):
            st.session_state.mode = "voice"
            st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🌿"): 
                st.write(msg["content"])
                if "audio" in msg and msg["audio"]:
                     st.audio(msg["audio"], format="audio/mp3", start_time=0)
        else:
            with st.chat_message("user", avatar="👤"): 
                st.write(msg["content"])

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

    if not st.session_state.report_generated:
        st.markdown("---")
        if st.button("📋 End Session & Generate Report", type="secondary", use_container_width=True):
            with st.spinner("Analyzing session..."):
                transcript_text = "\n".join([f"{m['role'].upper()}: {m.get('content','')}" for m in st.session_state.messages])
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

    if st.session_state.report_generated and "final_report_json" in st.session_state:
        try:
            data = json.loads(st.session_state.final_report_json)
            st.success("Analysis Complete")
            st.info(f"**Clinical Summary:** {data.get('clinical_summary', 'N/A')}")
            
            st.markdown("### 📊 Risk Matrix")
            if "risk_assessment" in data:
                df = pd.DataFrame(data["risk_assessment"])
                # Updated to use use_container_width to fix squashing
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download CSV", csv, "assessment.csv", "text/csv")
        except:
            st.error("Could not parse report.")
        
        if st.button("Start New Patient"):
            st.session_state.clear()
            st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🍃 Controls")
    if st.button("🔄 Reset App", type="secondary"):
        st.session_state.clear()
        st.rerun()
