import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="🧠 Python Code Assistant", layout="centered")
st.title("🧠 Python Code Assistant")
st.markdown("Generate Python code by describing your need in plain English.")

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "AhsanFarabi/python-assistant",
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("AhsanFarabi/python-assistant") # Hugging face model link
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

prompt = st.text_area("💬 Describe what you want the code to do:", height=100)

if st.button("🚀 Generate Code") and prompt.strip():
    with st.spinner("Generating Python code..."):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

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
