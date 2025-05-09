import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Streamlit UI setup
st.set_page_config(page_title="🧠 Friendly Python Assistant", layout="centered")
st.title("🧠 Friendly Python Assistant")
st.markdown("Generate Python code by describing your need in plain English.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("AhsanFarabi/python-assistant")
    tokenizer = AutoTokenizer.from_pretrained("AhsanFarabi/python-assistant")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cpu")  # Safe for Streamlit Cloud (no GPU assumption)
model.to(device)

# UI input
prompt = st.text_area("💬 Describe what you want the code to do:", height=100)

# Run generation
if st.button("🚀 Generate Code") and prompt.strip():
    with st.spinner("Generating Python code..."):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        st.subheader("🧾 Generated Code:")
        st.code(code, language="python")
