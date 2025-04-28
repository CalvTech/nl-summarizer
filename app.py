import streamlit as st
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline

# Pagina instellingen
st.set_page_config(page_title="📝 Slimme Samenvatter (NL)", page_icon="🧠", layout="centered")

st.title("🧠 Slimme Samenvatter (Nederlands)")
st.write("Voer een Nederlandse tekst in en ontvang een samenvatting! (Maximaal ongeveer **300 woorden**).")

# Model laden (nu correct T5Tokenizer)
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("KBLab/summ-nl")
    model = AutoModelForSeq2SeqLM.from_pretrained("KBLab/summ-nl")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_model()

# 🔥 Text input buiten de form
user_input = st.text_area("📄 Voer hier je tekst in:", height=250)

# Woordenteller LIVE
word_count = len(user_input.split())
st.write(f"✏️ Aantal ingevoerde woorden: **{word_count} woorden** (advies: max 300)")

# Andere settings
summary_length = st.slider("Maximale lengte samenvatting (in tokens):", 30, 300, 100)

# Submit button
if st.button("📝 Vat samen!"):
    if user_input.strip():
        if word_count > 300:
            st.warning(f"⚠️ Je tekst bevat {word_count} woorden. Probeer onder de 300 woorden te blijven voor de beste resultaten!")

        with st.spinner("✍️ Samenvatten bezig..."):
            summary = summarizer(user_input, max_length=summary_length, min_length=30, do_sample=False)
        
        st.success("✅ Samenvatting:")
        st.write(summary[0]["summary_text"])
    else:
        st.error("⚠️ Je hebt nog geen tekst ingevoerd.")
