import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Pagina instellingen
st.set_page_config(page_title="📝 Slimme Samenvatter (NL)", page_icon="🧠", layout="centered")

# Titel en uitleg
st.title("🧠 Slimme Samenvatter (Nederlands)")
st.write("Voer een Nederlandse tekst in en ontvang een samenvatting! (Maximaal ongeveer **300 woorden**).")

# Laad het model één keer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_model()

# Formulier voor invoer
with st.form("summarize_form"):
    user_input = st.text_area("📄 Voer hier je tekst in:", height=250)
    
    # Toon direct hoeveel woorden zijn ingevoerd
    word_count = len(user_input.split())
    st.write(f"✏️ Aantal ingevoerde woorden: **{word_count} woorden** (advies: max 300)")

    summary_length = st.slider("Maximale lengte samenvatting (in tokens):", 30, 300, 100)
    submitted = st.form_submit_button("📝 Vat samen!")

if submitted:
    if user_input.strip():
        if word_count > 300:
            st.warning(f"⚠️ Je tekst bevat {word_count} woorden. Probeer onder de 300 woorden te blijven voor beste resultaten!")

        with st.spinner("✍️ Samenvatten bezig..."):
            summary = summarizer(user_input, max_length=summary_length, min_length=30, do_sample=False)
        
        st.success("✅ Samenvatting:")
        st.write(summary[0]["summary_text"])
    else:
        st.error("⚠️ Je hebt nog geen tekst ingevoerd.")
