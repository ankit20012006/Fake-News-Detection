import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime
from time import sleep

#  LOAD MODEL 
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

#  PAGE CONFIG 
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

#  CLEAN UI 
st.markdown("""
<style>
:root {
    --sunset-1: #ffd86f;
    --sunset-2: #fc6262;
    --aqua-1: #59f0ff;
    --aqua-2: #7a8bff;
    --shell: #fff6f1;
    --glass: rgba(255, 255, 255, 0.24);
    --glass-strong: rgba(255, 255, 255, 0.38);
    --line-soft: rgba(255, 255, 255, 0.55);
    --ink: #3a225f;
    --ink-soft: #5d3d88;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 12%, rgba(255, 234, 148, 0.88) 0%, rgba(255, 234, 148, 0) 34%),
        radial-gradient(circle at 88% 18%, rgba(89, 240, 255, 0.72) 0%, rgba(89, 240, 255, 0) 42%),
        linear-gradient(130deg, #fff8e8 0%, #ffe4ec 36%, #e8efff 100%);
    background-size: 130% 130%;
    animation: skyShift 17s ease-in-out infinite alternate;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::before {
    width: 300px;
    height: 300px;
    left: -90px;
    top: 80px;
    background: rgba(255, 172, 202, 0.55);
}

[data-testid="stAppViewContainer"]::after {
    width: 340px;
    height: 340px;
    right: -110px;
    bottom: 90px;
    background: rgba(122, 139, 255, 0.5);
}

@keyframes skyShift {
    0% { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

.block-container {
    position: relative;
    z-index: 1;
    margin-top: 1.2rem;
    padding: 1.8rem 1.6rem;
    border-radius: 26px;
    border: 1px solid var(--line-soft);
    background: linear-gradient(140deg, var(--glass-strong), var(--glass));
    backdrop-filter: blur(22px) saturate(170%);
    box-shadow:
        0 20px 42px rgba(255, 148, 196, 0.32),
        0 14px 30px rgba(111, 154, 255, 0.24),
        inset 0 1px 0 rgba(255, 255, 255, 0.82);
}

h1 {
    text-align: center;
    font-size: clamp(2rem, 4vw, 3rem) !important;
    letter-spacing: 0.7px;
    color: var(--ink) !important;
    text-shadow: 0 6px 18px rgba(255, 227, 133, 0.6);
}

h3, p, .stMarkdown, label, .stCaption {
    color: var(--ink-soft) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, rgba(255, 255, 255, 0.52), rgba(237, 244, 255, 0.42));
    border-right: 1px solid rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(20px) saturate(170%);
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--ink-soft) !important;
}

[data-testid="stTextArea"] textarea {
    min-height: 210px;
    border: 1px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 18px !important;
    color: var(--ink) !important;
    background: linear-gradient(150deg, rgba(255, 255, 255, 0.8), rgba(255, 238, 244, 0.62)) !important;
    box-shadow:
        inset 0 1px 1px rgba(255, 255, 255, 0.85),
        0 14px 28px rgba(255, 166, 200, 0.24);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}

[data-testid="stTextArea"] textarea:focus {
    transform: translateY(-2px);
    box-shadow:
        0 0 0 3px rgba(122, 139, 255, 0.25),
        0 18px 32px rgba(122, 139, 255, 0.26);
}

.stButton > button {
    border: 0;
    border-radius: 16px;
    padding: 0.72rem 1.4rem;
    color: #fffaf4;
    font-weight: 700;
    letter-spacing: 0.3px;
    background: linear-gradient(135deg, var(--sunset-1) 0%, var(--sunset-2) 52%, #ff4f8b 100%);
    box-shadow:
        0 12px 24px rgba(252, 98, 98, 0.36),
        0 8px 18px rgba(255, 216, 111, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    filter: saturate(1.08);
    box-shadow:
        0 16px 30px rgba(252, 98, 98, 0.42),
        0 10px 22px rgba(255, 216, 111, 0.34);
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.01);
}

[data-testid="stMetricValue"],
.stAlert,
[data-testid="stDataFrame"],
.stPlotlyChart,
[data-baseweb="tab-list"] {
    border: 1px solid rgba(255, 255, 255, 0.62);
    border-radius: 18px;
    background: linear-gradient(140deg, rgba(255, 255, 255, 0.5), rgba(255, 242, 247, 0.34));
    backdrop-filter: blur(14px);
    box-shadow: 0 14px 26px rgba(145, 166, 255, 0.2);
}

[data-testid="stDataFrame"] {
    overflow: hidden;
}

[data-testid="stDataFrame"] table {
    border-collapse: collapse !important;
}

[data-testid="stDataFrame"] thead tr th {
    background: linear-gradient(135deg, rgba(255, 214, 236, 0.85), rgba(213, 236, 255, 0.88)) !important;
    color: #4b2d78 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.8) !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background: rgba(255, 255, 255, 0.55) !important;
}

[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: rgba(248, 238, 255, 0.5) !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(255, 225, 240, 0.72) !important;
}

[data-testid="stDataFrame"] tbody td {
    color: #5b3888 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.65) !important;
}

.result-banner {
    margin: 0.7rem 0 1rem 0;
    padding: 1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.82);
    background: linear-gradient(130deg, rgba(255, 255, 255, 0.58), rgba(255, 240, 248, 0.45));
    box-shadow:
        0 14px 30px rgba(255, 167, 202, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.9);
    animation: liftIn 0.5s ease;
}

.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    border-radius: 999px;
    padding: 0.4rem 0.95rem;
    color: #fff9f1;
    font-weight: 700;
    letter-spacing: 0.25px;
}

.result-pill.real {
    background: linear-gradient(135deg, #2ddadf, #6296ff);
    box-shadow: 0 8px 18px rgba(67, 217, 255, 0.4);
}

.result-pill.fake {
    background: linear-gradient(135deg, #ff6ea8, #ff8a6b);
    box-shadow: 0 8px 18px rgba(255, 110, 168, 0.42);
}

.result-meta {
    margin-top: 0.55rem;
    color: #5d3d88;
    font-size: 0.98rem;
    font-weight: 600;
}

[data-testid="stMetric"] {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.78);
    padding: 0.65rem 0.8rem;
    background: linear-gradient(140deg, rgba(255, 255, 255, 0.56), rgba(238, 246, 255, 0.46));
    box-shadow:
        0 12px 22px rgba(135, 172, 255, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 0.88);
    animation: popIn 0.45s ease;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #5d3d88 !important;
}

@keyframes popIn {
    0% { transform: translateY(12px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

@keyframes liftIn {
    0% { transform: translateY(14px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

.status-timeline {
    margin: 0.4rem 0 1rem 0;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.58), rgba(231, 246, 255, 0.45));
    box-shadow:
        0 12px 26px rgba(118, 161, 255, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 0.9);
    animation: popIn 0.45s ease;
}

.status-top {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #4f2f80;
    font-weight: 700;
}

.pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #43d9ff, #7a8bff);
    box-shadow: 0 0 12px rgba(67, 217, 255, 0.8);
    animation: beat 1.1s ease-in-out infinite;
}

.status-line {
    margin-top: 0.5rem;
    color: #6b4999;
    font-size: 0.94rem;
}

.status-track {
    margin-top: 0.6rem;
    height: 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.6);
    overflow: hidden;
}

.status-fill {
    height: 100%;
    width: 48%;
    border-radius: 999px;
    background: linear-gradient(90deg, #43d9ff, #7a8bff, #ff86b8);
    background-size: 200% 100%;
    animation: stream 1.4s linear infinite;
}

@keyframes stream {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

@keyframes beat {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.25); opacity: 0.75; }
}

.stProgress > div > div {
    background-color: rgba(255, 255, 255, 0.46);
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--aqua-1), var(--aqua-2));
    box-shadow: 0 0 16px rgba(89, 240, 255, 0.55);
    border-radius: 999px;
}

button[data-baseweb="tab"] {
    color: var(--ink-soft) !important;
    font-weight: 650;
    border-radius: 12px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255, 242, 160, 0.6), rgba(255, 207, 228, 0.6)) !important;
    color: var(--ink) !important;
}

footer,
#MainMenu,
header[data-testid="stHeader"] > div:nth-child(2) {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.9rem;
        border-radius: 20px;
    }

    h1 {
        font-size: 1.75rem !important;
    }

    .stButton > button {
        width: 100%;
    }
}

</style>
""", unsafe_allow_html=True)

#  TITLE 
st.title("📰 Fake News Detection App")
st.markdown("### 🔍 Analyze whether a news article is **Real or Fake**")

#  SESSION STATE 
if "history" not in st.session_state:
    st.session_state.history = []

#  SIDEBAR 
with st.sidebar:
    st.markdown("## ✨ Dashboard")
    st.caption("Bright glassmorphism analytics panel")
    st.metric("Predictions Made", len(st.session_state.history))
    st.markdown("### Quick Tips")
    st.markdown("- Use full news paragraph for better accuracy")
    st.markdown("- Check both chart and confidence bars")
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("History cleared")
        st.rerun()

#  INPUT 
user_input = st.text_area("✍️ Enter News Article", height=200)

#  BUTTON 
if st.button("🚀 Check News"):
    if user_input:

        status_box = st.empty()

        status_box.markdown(
            """
            <div class="status-timeline">
                <div class="status-top"><span class="pulse-dot"></span>Analyzing Article</div>
                <div class="status-line">Tokenizing and cleaning text input...</div>
                <div class="status-track"><div class="status-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        sleep(0.25)

        input_vector = vectorizer.transform([user_input])

        status_box.markdown(
            """
            <div class="status-timeline">
                <div class="status-top"><span class="pulse-dot"></span>Running Model Inference</div>
                <div class="status-line">Comparing patterns with trained classifier...</div>
                <div class="status-track"><div class="status-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        sleep(0.25)

        prediction = model.predict(input_vector)

        try:
            prob = model.predict_proba(input_vector)[0]
            fake_prob = prob[0]
            real_prob = prob[1]
        except:
            fake_prob = 1 - prediction[0]
            real_prob = prediction[0]

        status_box.markdown(
            """
            <div class="status-timeline">
                <div class="status-top"><span class="pulse-dot"></span>Finalizing Output</div>
                <div class="status-line">Confidence scores prepared successfully.</div>
                <div class="status-track"><div class="status-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        sleep(0.2)
        status_box.empty()

        #  RESULT 
        result = "REAL" if prediction[0] == 1 else "FAKE"

        if result == "REAL":
            st.markdown(
                f"""
                <div class="result-banner">
                    <span class="result-pill real">✅ REAL News</span>
                    <div class="result-meta">Model confidence: {real_prob*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                f"""
                <div class="result-banner">
                    <span class="result-pill fake">⚠️ FAKE News</span>
                    <div class="result-meta">Model confidence: {fake_prob*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col_real, col_fake = st.columns(2)
        with col_real:
            st.metric("REAL Confidence", f"{real_prob*100:.2f}%")
        with col_fake:
            st.metric("FAKE Confidence", f"{fake_prob*100:.2f}%")

        #  PROGRESS BARS 
        st.subheader("📊 Confidence Level")
        st.write("Real News Confidence")
        st.progress(real_prob)

        st.write("Fake News Confidence")
        st.progress(fake_prob)

        #  DATA 
        df = pd.DataFrame({
            "Category": ["Fake", "Real"],
            "Probability": [fake_prob, real_prob]
        })

        chart_colors = {
            "Fake": "#ff6ea8",
            "Real": "#43d9ff"
        }

        #  PLOTLY BAR 
        fig_bar = px.bar(
            df,
            x="Category",
            y="Probability",
            color="Category",
            color_discrete_map=chart_colors,
            text="Probability",
            title="Confidence Score"
        )
        fig_bar.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside',
            marker_line_width=2,
            marker_line_color='rgba(255,255,255,0.85)',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}<extra></extra>'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.2)',
            font=dict(color='#5d3d88'),
            title_font=dict(color='#3a225f', size=22),
            legend=dict(
                title='',
                orientation='h',
                y=1.12,
                x=0.5,
                xanchor='center',
                font=dict(color='#5d3d88')
            ),
            xaxis=dict(
                title='',
                showgrid=False,
                tickfont=dict(color='#5d3d88')
            ),
            yaxis=dict(
                title='Probability',
                range=[0, 1],
                tickformat='.0%',
                gridcolor='rgba(122, 139, 255, 0.28)',
                zeroline=False,
                tickfont=dict(color='#5d3d88')
            ),
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.92)',
                font=dict(color='#3a225f')
            ),
            margin=dict(t=65, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        #  PLOTLY PIE 
        fig_pie = px.pie(
            df,
            names="Category",
            values="Probability",
            hole=0.4,
            color="Category",
            color_discrete_map=chart_colors,
            title="Distribution"
        )
        fig_pie.update_traces(
            textinfo='percent+label',
            textfont=dict(color='#fff8f2', size=14),
            marker=dict(line=dict(color='rgba(255,255,255,0.85)', width=2)),
            hovertemplate='<b>%{label}</b><br>Share: %{percent}<br>Probability: %{value:.2f}<extra></extra>'
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#5d3d88'),
            title_font=dict(color='#3a225f', size=22),
            legend=dict(
                orientation='h',
                y=-0.08,
                x=0.5,
                xanchor='center',
                font=dict(color='#5d3d88')
            ),
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.92)',
                font=dict(color='#3a225f')
            ),
            margin=dict(t=65, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        #  SAVE HISTORY 
        st.session_state.history.append({
            "Text": user_input[:100],
            "Result": result,
            "Real %": round(real_prob*100, 2),
            "Fake %": round(fake_prob*100, 2),
            "Time": datetime.now().strftime("%H:%M:%S")
        })

    else:
        st.warning("⚠️ Please enter a news article")

#  TABS 
tab1, tab2 = st.tabs(["📜 History", "📥 Download"])

#  HISTORY 
with tab1:
    st.subheader("Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No history yet")

# DOWNLOAD 
with tab2:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download History", csv, "history.csv", "text/csv")
    else:
        st.info("Nothing to download yet")

#  FOOTER 
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit + Plotly")