import streamlit as st
from services import generate_fake_reviews, run_analysis

#fixed

st.set_page_config(page_title="Анализ отзывов", layout="wide")
st.title("Анализ отзывов (демо)")

uploaded_file = st.file_uploader(
    "Загрузите файл с отзывами (CSV/TSV/XLSX)", type=["csv", "tsv", "xlsx"]
)

if uploaded_file is not None:
    st.success(f"Файл `{uploaded_file.name}` загружен (имитация анализа)")

    # instead of real parsing we just fake reviews
    reviews = generate_fake_reviews(10)
    analysis = run_analysis(reviews)

    st.subheader("📊 Статистика тональностей")
    st.bar_chart(analysis.stats)

    st.subheader("💬 Примеры отзывов")
    for r in analysis.reviews:
        st.write(f"**{r.sentiment.capitalize()}** — {r.text} ({r.topic})")

else:
    st.info("Пока файл не выбран. Для демонстрации используйте любой CSV.")
