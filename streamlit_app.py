import streamlit as st
import pandas as pd
import io
import plotly.express as px
from src.predictor_engine import QuestionPredictorEngine

# Page configuration
st.set_page_config(
    page_title="AI Question Paper Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
    }
    .question-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return QuestionPredictorEngine(model_name="all-MiniLM-L6-v2")

engine = load_engine()

# --- Sidebar ---
st.sidebar.title("📁 Upload & Settings")
st.sidebar.write("Upload a `.zip` archive containing your previous years' question papers (PDF format).")

uploaded_zip = st.sidebar.file_uploader("Select ZIP Archive", type=["zip"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Prediction Tuning")
similarity_thresh = st.sidebar.slider(
    "Semantic Similarity Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.72,
    step=0.02,
    help="Higher values require questions to be phrased almost identically. Lower values group similar concepts together."
)

min_repetition = st.sidebar.slider(
    "Filter by Minimum Repetition",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Show only questions that repeated at least this many times."
)

# --- Header ---
st.title("🎓 Question Paper Prediction System")
st.caption("Extracts questions from past PDFs, groups them with Sentence Transformers, and ranks them by repetition and exam weightage.")

if not uploaded_zip:
    st.info("👋 Upload a `.zip` folder of past question paper PDFs from the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. 📦 Upload ZIP")
        st.write("Put all your university or school PDF question papers in one zip archive.")
    with col2:
        st.markdown("### 2. 🤖 Extract & Convert")
        st.write("PDF text is parsed into clean text and structured questions with marks.")
    with col3:
        st.markdown("### 3. 🎯 Semantic Prediction")
        st.write("Sentence Transformers detect repeated questions regardless of wording variations.")

else:
    # Process ZIP
    with st.spinner("Extracting PDF files from ZIP..."):
        pdf_dict = engine.extract_zip(uploaded_zip.getvalue())

    if not pdf_dict:
        st.error("❌ No valid `.pdf` files found inside the uploaded ZIP archive. Please verify its contents.")
    else:
        st.success(f"✅ Found **{len(pdf_dict)} PDF papers** inside the archive: {', '.join(pdf_dict.keys())}")

        predict_btn = st.button("🚀 Predict Questions Based on Repetition & Weightage", type="primary", use_container_width=True)

        if predict_btn or "predicted_df" in st.session_state:
            if predict_btn or "raw_questions_df" not in st.session_state:
                with st.spinner("Extracting text and identifying questions from all PDFs..."):
                    raw_texts, raw_questions_df = engine.process_all_papers(pdf_dict)
                    st.session_state["raw_texts"] = raw_texts
                    st.session_state["raw_questions_df"] = raw_questions_df

                with st.spinner("Running Sentence Transformer semantic grouping & weightage calculation..."):
                    predicted_df = engine.cluster_and_predict(raw_questions_df, similarity_threshold=similarity_thresh)
                    st.session_state["predicted_df"] = predicted_df
            else:
                raw_questions_df = st.session_state["raw_questions_df"]
                predicted_df = engine.cluster_and_predict(raw_questions_df, similarity_threshold=similarity_thresh)
                st.session_state["predicted_df"] = predicted_df

            raw_df = st.session_state.get("raw_questions_df", pd.DataFrame())
            pred_df = st.session_state.get("predicted_df", pd.DataFrame())

            if pred_df.empty:
                st.warning("Could not extract any questions. Check if the PDFs are scanned images or have non-standard layouts.")
            else:
                # Apply repetition filter
                filtered_pred_df = pred_df[pred_df["Times Repeated"] >= min_repetition].copy()

                # --- Metrics Overview ---
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📄 Papers Analyzed", len(pdf_dict))
                m2.metric("❓ Total Extracted Questions", len(raw_df))
                m3.metric("🎯 Unique Question Topics", len(pred_df))
                repeated_count = len(pred_df[pred_df["Times Repeated"] > 1])
                m4.metric("🔁 Repeated Questions", repeated_count)

                # --- Visual Charts ---
                st.subheader("📊 Repetition & Marks Distribution")
                c1, c2 = st.columns(2)
                with c1:
                    top_10 = filtered_pred_df.head(10).copy()
                    top_10['Short Question'] = top_10['Predicted Question'].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
                    fig_bar = px.bar(
                        top_10,
                        x='Importance Score',
                        y='Short Question',
                        orientation='h',
                        color='Times Repeated',
                        title="Top 10 High-Probability Questions (By Importance Score)",
                        labels={'Short Question': 'Question'}
                    )
                    fig_bar.update_layout(yaxis={'autorange': 'reversed'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                with c2:
                    fig_pie = px.pie(
                        filtered_pred_df,
                        names='Times Repeated',
                        title="Questions by Repetition Frequency",
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # --- Main Results Table ---
                st.subheader(f"📋 Ranked Predicted Questions ({len(filtered_pred_df)} questions)")

                tab1, tab2, tab3 = st.tabs(["🔥 Ranked Question List", "📊 Full Data Table", "📄 Extracted Raw Texts"])

                with tab1:
                    for idx, row in filtered_pred_df.iterrows():
                        badge_color = "#d9534f" if row["Times Repeated"] >= 3 else ("#f0ad4e" if row["Times Repeated"] == 2 else "#5bc0de")
                        st.markdown(f"""
                        <div class="question-box">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="background:{badge_color}; color:white; padding:3px 8px; border-radius:12px; font-weight:bold; font-size:12px;">
                                    Repeated {row['Times Repeated']}x ({row['Paper Coverage (%)']}% Papers)
                                </span>
                                <span style="font-size:13px; color:#555;">
                                    Avg Marks: <b>{row['Avg Marks']}</b> | Max Marks: <b>{row['Max Marks']}</b> | Importance Score: <b>{row['Importance Score']}</b>
                                </span>
                            </div>
                            <div style="font-size:16px; font-weight:500; color:#1f2937;">
                                {row['Predicted Question']}
                            </div>
                            <div style="font-size:12px; color:#777; margin-top:6px;">
                                📁 <i>Found in: {row['Found In Papers']}</i>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with tab2:
                    st.dataframe(
                        filtered_pred_df[[
                            "Importance Score",
                            "Predicted Question",
                            "Times Repeated",
                            "Paper Coverage (%)",
                            "Avg Marks",
                            "Max Marks",
                            "Found In Papers"
                        ]],
                        use_container_width=True
                    )

                    # Export to CSV
                    csv_data = filtered_pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predicted Questions as CSV",
                        data=csv_data,
                        file_name="predicted_questions.csv",
                        mime="text/csv"
                    )

                with tab3:
                    raw_texts = st.session_state.get("raw_texts", {})
                    selected_doc = st.selectbox("Select document to view extracted text:", list(raw_texts.keys()))
                    if selected_doc:
                        st.text_area("Converted Text", raw_texts[selected_doc], height=350)
