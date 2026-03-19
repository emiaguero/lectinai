import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import cv2
import numpy as np
from morphometry import MorphometryAnalyzer

# --- Page Config ---
st.set_page_config(
    page_title="LectinAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 960px; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Header */
    .hero-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #26A69A 0%, #00897B 50%, #00695C 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; letter-spacing: -0.03em; line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 0.85rem; color: #80CBC4; font-weight: 400;
        letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.2rem;
    }
    .teal-divider {
        height: 1px; border: none; margin: 1.2rem 0;
        background: linear-gradient(90deg, transparent, rgba(38,166,154,0.3), transparent);
    }

    /* Result hero — the big diagnosis panel */
    .result-panel {
        background: linear-gradient(145deg, #1A1F2E 0%, #232838 100%);
        border: 1px solid rgba(38, 166, 154, 0.2);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-section-label {
        font-size: 0.7rem; color: #607D8B; text-transform: uppercase;
        letter-spacing: 0.15em; font-weight: 600; margin-bottom: 0.6rem;
    }
    .result-score {
        font-size: 4rem; font-weight: 800; line-height: 1;
        margin: 0.2rem 0;
    }
    .result-label {
        font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem;
    }
    .result-bar-track {
        width: 100%; height: 6px; background: rgba(255,255,255,0.06);
        border-radius: 3px; margin-top: 0.8rem; overflow: hidden;
    }
    .result-bar-fill {
        height: 100%; border-radius: 3px;
        transition: width 0.6s ease;
    }

    /* Secondary metric */
    .secondary-metric {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        display: flex; align-items: center; gap: 0.8rem;
    }
    .secondary-metric-value {
        font-size: 1.3rem; font-weight: 700; color: #80CBC4;
    }
    .secondary-metric-label {
        font-size: 0.78rem; color: #78909C; line-height: 1.3;
    }

    /* Legend pills */
    .legend-row {
        display: flex; gap: 0.4rem; flex-wrap: wrap;
        justify-content: center; margin-top: 0.8rem;
    }
    .legend-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 100px; padding: 0.25rem 0.7rem;
        font-size: 0.7rem; color: #90A4AE;
    }
    .legend-dot {
        width: 7px; height: 7px; border-radius: 50%; display: inline-block;
    }

    /* Image panel inside expander */
    .image-panel {
        background: linear-gradient(145deg, #1A1F2E 0%, #232838 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 0.8rem; overflow: hidden;
    }
    .panel-label {
        font-size: 0.7rem; color: #80CBC4; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 600;
        margin-bottom: 0.5rem; padding-left: 0.25rem;
    }

    /* Empty state */
    .empty-state {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 340px; color: #546E7A;
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 20px; background: rgba(255,255,255,0.02);
    }
    .empty-icon { font-size: 3.5rem; margin-bottom: 0.8rem; opacity: 0.35; }
    .empty-text { font-size: 0.95rem; color: #607D8B; }
    .empty-hint { color: #455A64; font-size: 0.78rem; margin-top: 0.3rem; }

    /* Streamlit overrides */
    div[data-testid="stFileUploader"] { border-radius: 16px; }
    div[data-testid="stFileUploader"] section { padding: 1.2rem !important; }
    .stSpinner > div { border-top-color: #26A69A !important; }

    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_analyzer():
    analyzer = MorphometryAnalyzer()
    model_path = os.path.join(os.path.dirname(__file__), "assets", "lectin_model.pth")
    analyzer.load_ai_model(model_path)
    return analyzer


analyzer = get_analyzer()


def load_image_from_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


SCORE_LABELS = ["Nada", "Leve", "Moderado", "Alto"]
SCORE_COLORS = ["#546E7A", "#FDD835", "#FB8C00", "#E53935"]


def render_score_panel(score, section_label):
    """Render a large hero-style score card."""
    label = SCORE_LABELS[score]
    color = SCORE_COLORS[score]
    bar_pct = int((score / 3) * 100)
    return f"""
    <div class="result-panel">
        <div class="result-section-label">{section_label}</div>
        <div class="result-score" style="color:{color}">{score}</div>
        <div class="result-label" style="color:{color}">{label}</div>
        <div class="result-bar-track">
            <div class="result-bar-fill" style="width:{bar_pct}%; background:{color}"></div>
        </div>
    </div>
    """


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="hero-title">LectinAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Clasificación de Intensidad de Tinción por IA</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.file_uploader(
    "Subir micrografía",
    type=["jpg", "png", "jpeg", "tif", "tiff"],
    label_visibility="collapsed",
    help="Formatos: JPG, PNG, TIFF",
)

if uploaded_file is not None:
    image = load_image_from_upload(uploaded_file)

    with st.spinner("Analizando muestra..."):
        tissue_mask = analyzer.segment_tissue(image)
        h_channel, dab_channel = analyzer.separate_stains(image)

        if h_channel is None:
            st.error("Fallo en separación de tinciones")
            st.stop()

        positive_mask = analyzer.segment_positive_area(dab_channel, threshold=0.25)
        ratio = analyzer.calculate_ratio(tissue_mask, positive_mask)
        zonal_results = analyzer.analyze_zonal_intensity(
            tissue_mask, dab_channel, positive_mask
        )
        ai_results = analyzer.predict_intensity_ai(image)
        border_mask_viz = zonal_results["masks"]["border"]
        overlay = analyzer.generate_overlay(
            image, positive_mask, border_mask=border_mask_viz
        )

    # ============================================================
    # PRIMARY RESULT — AI INTENSITY SCORES (hero)
    # ============================================================
    col_border, col_inner = st.columns(2)

    if ai_results:
        with col_border:
            st.markdown(
                render_score_panel(ai_results["border_score"], "Intensidad del Borde"),
                unsafe_allow_html=True,
            )
        with col_inner:
            st.markdown(
                render_score_panel(ai_results["inner_score"], "Intensidad del Interior"),
                unsafe_allow_html=True,
            )
    else:
        st.warning("Modelo de IA no disponible. Verifica que `lectin_model.pth` esté en `src/assets/`.")

    # Scale legend
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-pill"><span class="legend-dot" style="background:#546E7A"></span> 0 · Nada</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#FDD835"></span> 1 · Leve</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#FB8C00"></span> 2 · Moderado</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#E53935"></span> 3 · Alto</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # SECONDARY — Tissue coverage (clearer name)
    # ============================================================
    st.markdown(
        f"""
        <div class="secondary-metric">
            <div class="secondary-metric-value">{ratio:.1f}%</div>
            <div class="secondary-metric-label">
                Cobertura de tejido teñido<br>
                <span style="font-size:0.7rem; color:#546E7A">Porcentaje del área tisular con tinción DAB positiva (OD &gt; 0.25)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # DETAIL — Image comparison
    # ============================================================
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown(
            '<div class="image-panel"><div class="panel-label">Imagen Original</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_result:
        st.markdown(
            '<div class="image-panel"><div class="panel-label">Áreas Detectadas</div>',
            unsafe_allow_html=True,
        )
        st.image(overlay, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # ============================================================
    # EMPTY STATE
    # ============================================================
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-text">Sube una micrografía para obtener la clasificación de intensidad</div>
            <div class="empty-hint">JPG, PNG o TIFF · Tinción H&amp;E DAB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
