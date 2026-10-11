import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "Ahmed-1511/distilbart-samsum"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    return tokenizer, model


tokenizer, model = load_model()


st.title("💬 Dialogue Summarization")

st.caption(
    "A fine-tuned DistilBART model for automatic dialogue summarization."
)


dialogue = st.text_area(
    "Enter your dialogue:",
    height=250,
    placeholder="Paste your dialogue here..."
)


if st.button("Summarize"):
    if dialogue.strip():

        inputs = tokenizer(
            dialogue,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        with st.spinner("Generating summary..."):
            with torch.no_grad():
                summary_ids = model.generate(
                    **inputs,
                    max_length=100,
                    min_length=10,
                    num_beams=4,
                    early_stopping=True
                )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        st.subheader("Summary")
        st.info(summary)

    else:
        st.warning("Please enter a dialogue.")

