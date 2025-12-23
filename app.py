import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import os
import PIL.Image
from pypdf import PdfReader

# 1. Page Configuration
st.set_page_config(page_title="Gemini Code Architect", page_icon="👨‍💻", layout="wide")
st.title("Gemini Code Architect 👨‍💻")
st.caption("Expert Coding Help: Web, Mobile, Python, AI, and Software Engineering")

# 2. Sidebar Setup
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter Google API Key", type="password")
    
    # --- MODE SELECTOR ---
    st.divider()
    st.header("🧠 Select Your Expert")
    mode = st.selectbox(
        "Choose the AI's Personality:",
        [
            "General Assistant", 
            "Web Developer (HTML/CSS/JS)", 
            "Mobile App Developer (Flutter/iOS/Android)",
            "Python & AI Engineer", 
            "Software Architect"
        ]
    )
    st.info(f"Active Mode: **{mode}**")
    # ----------------------------------

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    enable_voice = st.toggle("Enable Voice Response")
    
    st.divider()
    
    # Image Analysis
    st.write("📷 **Image Analysis**")
    uploaded_image = st.file_uploader("Upload UI designs...", type=["jpg", "jpeg", "png"])
    image_data = None
    if uploaded_image:
        image_data = PIL.Image.open(uploaded_image)
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # PDF Analysis
    st.write("📄 **Document Analysis**")
    uploaded_pdf = st.file_uploader("Upload specs or docs...", type=["pdf"])
    pdf_text = ""
    if uploaded_pdf:
        reader = PdfReader(uploaded_pdf)
        for page in reader.pages:
            pdf_text += page.extract_text()
        st.success(f"PDF Loaded: {len(reader.pages)} pages")

# 3. Dynamic System Instructions
# This logic sets the "Brain" of the AI based on your dropdown selection
if mode == "Web Developer (HTML/CSS/JS)":
    system_instruction = """You are an expert Senior Web Developer. 
    - Expert in HTML5, CSS3, JavaScript, React, and Node.js.
    - Focus on responsive design, accessibility, and modern frameworks.
    - Always provide clean, copy-pasteable code blocks."""

elif mode == "Mobile App Developer (Flutter/iOS/Android)":
    system_instruction = """You are an expert Mobile App Developer.
    - Expert in React Native, Flutter, Swift (iOS), and Kotlin (Android).
    - When asked for code, ask the user which platform they prefer if they didn't specify.
    - Explain the difference between native (Swift/Kotlin) and cross-platform (React Native).
    - Provide code for mobile UI components (buttons, nav bars, lists)."""

elif mode == "Python & AI Engineer":
    system_instruction = """You are an expert Python and AI Engineer.
    - Specializing in Python, Streamlit, TensorFlow, PyTorch, and Data Science.
    - Explain complex AI concepts simply.
    - Provide complete, runnable Python scripts."""

elif mode == "Software Architect":
    system_instruction = """You are a Senior Systems Architect.
    - Focus on high-level structure, databases, scalability, and security.
    - Do not just write code; explain the 'Big Picture' design.
    - Help the user choose the right tech stack for their startup ideas."""

else:
    system_instruction = "You are a helpful AI assistant. Answer questions clearly and concisely."

# 4. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Main Logic
if api_key:
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        'gemini-flash-latest',
        system_instruction=system_instruction
    )

    if prompt := st.chat_input("Ask for code, designs, or advice..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                def stream_generator():
                    # Scenario 1: Image Analysis
                    if image_data:
                        response = model.generate_content([prompt, image_data], stream=True)
                        for chunk in response:
                            yield chunk.text
                            
                    # Scenario 2: PDF Analysis
                    elif pdf_text:
                        combined_prompt = f"Context from PDF:\n{pdf_text}\n\nUser Question: {prompt}"
                        chat = model.start_chat(history=[
                            {"role": m["role"], "parts": m["content"]}
                            for m in st.session_state.messages
                            if m["role"] in ["user", "model"]
                        ])
                        response = chat.send_message(combined_prompt, stream=True)
                        for chunk in response:
                            yield chunk.text

                    # Scenario 3: Regular Chat
                    else:
                        chat = model.start_chat(history=[
                            {"role": m["role"], "parts": m["content"]}
                            for m in st.session_state.messages
                            if m["role"] in ["user", "model"]
                        ])
                        response = chat.send_message(prompt, stream=True)
                        for chunk in response:
                            yield chunk.text

                full_response = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "model", "content": full_response})

                if enable_voice:
                    tts = gTTS(text=full_response, lang='en')
                    tts.save("response.mp3")
                    st.audio("response.mp3")
                    os.remove("response.mp3")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("Please enter your API Key in the sidebar.")