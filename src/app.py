import streamlit as st
import json
import os
import uuid
import io
import shutil
from contextlib import redirect_stdout
from agent import FinancialAgent

st.set_page_config(
    page_title="Finansal AI Asistanı",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Sohbet Baloncukları */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #333; /* Hafif çerçeve */
    }
    /* Butonlar */
    .stButton button {
        border-radius: 8px;
    }
    /* Dosya Yükleyiciyi Güzelleştir */
    [data-testid="stFileUploader"] {
        padding: 10px;
        border: 1px dashed #555;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

HISTORY_FILE = "chat_history.json"
TEMP_DIR = "temp_data"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@st.cache_resource
def load_agent():
    return FinancialAgent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"⚠️ Agent hatası: {e}")
    st.stop()

def load_history():
    if not os.path.exists(HISTORY_FILE): return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return {}

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def save_uploaded_file(uploaded_file):
    if uploaded_file is None: return None
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.history[new_id] = {"title": "Yeni Sohbet", "messages": []}

with st.sidebar:
    st.title("📊 Finans Paneli")

    st.markdown("### 📎 Veri Yükle")
    uploaded_file = st.file_uploader(
        "CSV Dosyası Seç", 
        type=["csv"], 
        help="Analiz edilecek dosyayı buraya bırakın."
    )
    
    current_file_path = None
    if uploaded_file:
        current_file_path = save_uploaded_file(uploaded_file)
        st.success(f"✅ Dosya Aktif: {uploaded_file.name}")
    
    st.markdown("---")
    
    if st.button("➕ Yeni Sohbet", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.history[new_id] = {"title": "Yeni Sohbet", "messages": []}
        save_history(st.session_state.history)
        st.rerun()

    sessions = list(st.session_state.history.keys())[::-1]
    if sessions:
        st.caption("Geçmiş Sohbetler")
        for session_id in sessions:
            chat = st.session_state.history[session_id]
            title = chat.get("title", "Adsız")

            if session_id == st.session_state.current_session_id:
                st.button(f"🟢 {title}", key=session_id, use_container_width=True, disabled=True)
            else:
                if st.button(f"⚪ {title}", key=session_id, use_container_width=True):
                    st.session_state.current_session_id = session_id
                    st.rerun()
    
    if st.button("🗑️ Geçmişi Sil", type="primary"):
        st.session_state.history = {}
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR); os.makedirs(TEMP_DIR)
        st.rerun()

current_id = st.session_state.current_session_id
current_chat = st.session_state.history.get(current_id, {"title": "Yeni Sohbet", "messages": []})

st.header("🤖 Finansal Analist Asistanı")

for msg in current_chat["messages"]:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if "thoughts" in msg and msg["thoughts"]:
            with st.expander("🛠️ Teknik Detaylar"):
                st.code(msg["thoughts"], language="text")

if prompt := st.chat_input("Bir soru sorun (Örn: Aselsan durumu ne? veya Dosyayı yorumla)"):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    current_chat["messages"].append({"role": "user", "content": prompt})

    if len(current_chat["messages"]) <= 2:
        current_chat["title"] = prompt[:20] + "..." if len(prompt) > 20 else prompt
        save_history(st.session_state.history)

    final_prompt = prompt
    if current_file_path:
        final_prompt += f"\n\n[SİSTEM: Kullanıcı '{uploaded_file.name}' dosyasını yükledi. Dosya Yolu: '{current_file_path}'. Eğer dosya analizi istenirse 'analyze_full_csv' aracını kullan.]"

    with st.chat_message("assistant", avatar="🤖"):
        thought_buffer = io.StringIO()
        
        with st.status("🧠 Analiz yapılıyor...", expanded=True) as status:
            st.write("Veriler taranıyor...")
            with redirect_stdout(thought_buffer):
                try:
                    response_text = agent.run(final_prompt)
                except Exception as e:
                    response_text = f"Hata: {e}"
            status.update(label="Tamamlandı", state="complete", expanded=False)

        st.markdown(response_text)
        
        captured_logs = thought_buffer.getvalue()
        if captured_logs.strip():
            with st.expander("🛠️ Ajanın Düşünce Adımları"):
                st.code(captured_logs, language="text")

    current_chat["messages"].append({
        "role": "assistant",
        "content": response_text,
        "thoughts": captured_logs
    })
    save_history(st.session_state.history)