import streamlit as st
import pandas as pd
import plotly.express as px

from services import generate_fake_reviews, run_analysis
from file_processing import load_reviews_from_file
from sentiment_analysis import get_hf_logs
from models import Review
from export_utils import render_export_buttons

st.set_page_config(page_title="Анализ отзывов", layout="wide")
st.title("Анализ отзывов")

with st.sidebar:
    st.header("Данные")
    uploaded_file = st.file_uploader(
        "Загрузите файл (CSV/TSV/XLSX)", type=["csv", "tsv", "xlsx"]
    )
    st.caption("Столбец с текстом будет определён автоматически (например, `text`, `review`, `Отзыв`).")
    st.divider()
    st.subheader("Или сгенерируйте тестовые")
    num_reviews = st.slider("Количество отзывов", 5, 200, 50, step=5)
    generate_btn = st.button("Сгенерировать отзывы")
    st.divider()
    st.header("Тональность (ИИ)")
    use_ai = st.checkbox("Определять тональность через ИИ (Hugging Face API)", value=False)
    model_choice = st.selectbox(
        "Модель",
        options=[
            "auto (recommended)",
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "custom...",
        ],
        index=0,
    )
    if model_choice == "custom...":
        hf_model = st.text_input("Укажите модель HF", value="nlptown/bert-base-multilingual-uncased-sentiment")
    else:
        hf_model = model_choice
    hf_token = st.text_input("HF API Token", type="password", help="Создайте токен на huggingface.co/settings/tokens")

reviews: list[Review] | None = None

if uploaded_file is not None and st.button("Проанализировать загруженный файл"):
    file_bytes = uploaded_file.getvalue()
    try:
        reviews = load_reviews_from_file(file_bytes, uploaded_file.name)
        st.success(f"Загружено отзывов: {len(reviews)}")
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")

if reviews is None and generate_btn:
    reviews = generate_fake_reviews(num_reviews)
    st.info(f"Сгенерировано отзывов: {len(reviews)}")

if reviews:
    # Determine if AI can actually be used (token and model present)
    effective_use_ai = bool(use_ai and hf_token and hf_model)
    if use_ai and not effective_use_ai:
        st.warning("ИИ-анализ не был запущен: укажите валидный HF API Token и модель.")

    # Уведомление о времени выполнения
    st.info("🔄 Запуск анализа... Процесс займет 1-2 минуты. Пожалуйста, подождите.")
    
    analysis = run_analysis(
        reviews,
        use_ai=effective_use_ai,
        hf_model=hf_model if effective_use_ai else None,
        hf_token=hf_token if effective_use_ai else None,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Позитивные", analysis.stats["sentiment_counts"].get("positive", 0))
    with col2:
        st.metric("Нейтральные", analysis.stats["sentiment_counts"].get("neutral", 0))
    with col3:
        st.metric("Негативные", analysis.stats["sentiment_counts"].get("negative", 0))

    st.subheader("Тональность")
    sent_df = (
        pd.DataFrame(
            [
                {"sentiment": k, "count": v}
                for k, v in analysis.stats["sentiment_counts"].items()
            ]
        )
        .sort_values("count", ascending=False)
    )
    fig_sent = px.bar(sent_df, x="sentiment", y="count", color="sentiment", title="Распределение тональности")
    st.plotly_chart(fig_sent, config={"displayModeBar": True})

    st.subheader("Таблица отзывов")
    table_df = pd.DataFrame(
        [
            {
                "text": (r.text or ""),
                "sentiment": (r.sentiment or ""),
                "score": (r.sentiment_score or 0.0),
                "language": (r.language or ""),
            }
            for r in analysis.reviews
        ]
    )
    table_df = table_df.fillna("")
    # Ensure score is shown as float with 3 decimals in the UI
    try:
        table_df["score"] = pd.to_numeric(table_df["score"], errors="coerce").fillna(0.0)
    except Exception:
        pass
    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(label="score", format="%.3f"),
        },
    )

    # Добавляем кнопки экспорта
    render_export_buttons(analysis)

    st.subheader("Ключевые проблемы и достоинства (ИИ)")
    insights = analysis.stats.get("insights", {"problems": [], "strengths": []})
    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("**Проблемы**")
        if insights.get("problems"):
            for it in insights["problems"]:
                st.markdown(f"- {it}")
        else:
            st.caption("Нет обнаруженных проблем")
    with col_s:
        st.markdown("**Достоинства**")
        if insights.get("strengths"):
            for it in insights["strengths"]:
                st.markdown(f"- {it}")
        else:
            st.caption("Нет обнаруженных достоинств")

    # Heuristic notice if AI expected but scores look zero-ish
    if use_ai:
        any_nonzero = any(abs(getattr(r, "sentiment_score", 0.0)) > 1e-6 for r in analysis.reviews)
        if not any_nonzero:
            st.info(
                "Похоже, что оценки ИИ равны 0. Проверьте токен HF, модель и сетевое подключение."
            )

    with st.expander("Логи запросов к HF API"):
        logs = get_hf_logs(10)
        if logs:
            st.json(logs)
        else:
            st.caption("Пока нет логов. Запустите ИИ-анализ для появления записей.")

    with st.expander("Технические детали анализа"):
        st.json({"analysis_id": analysis.id, "created_at": str(analysis.created_at), "stats": analysis.stats})
else:
    st.info("Загрузите файл или сгенерируйте тестовые данные в сайдбаре, затем запустите анализ.")
