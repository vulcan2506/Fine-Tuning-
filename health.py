import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Set page config FIRST (only once)
st.set_page_config(page_title="Medical LLM Chat", layout="wide")

# ----------------------------
# 🔧 Load model and tokenizer
# ----------------------------
MODEL_PATH = "/home/vulcan/Documents/Projects/Fine Tuning /health_aligned_dpo/"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    )
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# 🌐 Streamlit UI
# ----------------------------
st.title("🩺 MedChat: Ask Me Anything Health-related!")

chat_history = st.session_state.get("chat_history", [])

# Input prompt
user_input = st.text_area("💬 Your question:", height=120)

# Submit button
if st.button("🔍 Get Answer"):
    if user_input.strip():
        chat_history.append({"role": "user", "content": user_input})

        # Build full prompt from history
        full_prompt = ""
        for msg in chat_history:
            prefix = "User:" if msg["role"] == "user" else "Assistant:"
            full_prompt += f"{prefix} {msg['content']}\n"
        full_prompt += "Assistant:"

        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)

        with st.spinner("Generating response..."):
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            response = generated.split("Assistant:")[-1].strip()

            chat_history.append({"role": "assistant", "content": response})
            st.session_state.chat_history = chat_history

# Display full conversation
if chat_history:
    st.subheader("📜 Conversation History")
    for msg in chat_history:
        role = "🧑‍⚕️ Assistant" if msg["role"] == "assistant" else "🙋 User"
        st.markdown(f"**{role}:** {msg['content']}")
