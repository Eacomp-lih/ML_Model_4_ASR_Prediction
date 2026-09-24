from pathlib import Path

import streamlit as st

from prediction_core import ASRPredictor, ELECTROLYTES, MODEL_NAMES


st.set_page_config(
    page_title="钙钛矿型SOFC阴极材料650℃下ASR预测",
    page_icon="⚡",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { background: #f7f9fc; }
    .block-container { max-width: 980px; padding-top: 2.2rem; }
    .title { color: #17324d; font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
    .subtitle { color: #617386; margin-bottom: 1.5rem; }
    .result-card { background: white; border: 1px solid #dce5ee; border-radius: 12px;
                   padding: 1rem 1.2rem; margin-top: 1rem; box-shadow: 0 2px 10px #17324d12; }
    .label { color: #617386; font-size: .85rem; margin-bottom: .2rem; }
    .value { color: #17324d; font-size: 1.15rem; font-weight: 650; }
    .score { color: #147d67; font-size: 1.35rem; font-weight: 750; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">钙钛矿型SOFC阴极材料650℃下ASR预测</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">输入阴极材料化学式与电解质类型，调用训练好的机器学习模型进行预测，并评估其是否位于训练数据覆盖区域。</div>',
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="正在加载模型和PCA适用域分析器…")
def get_predictor():
    return ASRPredictor()


with st.container(border=True):
    st.subheader("预测输入")
    left, right = st.columns([1.35, 1])
    with left:
        formula = st.text_input("化学式", value="Pr0.3Sr0.7CoO3", help="支持 ABO3 或 A2B2O5 形式，例如 SmBaCo2O5")
    with right:
        model = st.selectbox("模型", [name.upper() for name in MODEL_NAMES], index=1)
    electrolyte = st.selectbox("电解质类型", list(ELECTROLYTES), index=1)
    run = st.button("运行预测", type="primary", use_container_width=True)

if run:
    try:
        predictor = get_predictor()
        with st.spinner("正在生成特征并计算预测…"):
            result = predictor.predict(formula, electrolyte, model.lower(), verbose=False)

        st.success("预测完成")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        cols = st.columns(3)
        fields = [
            ("使用模型", result["model"].upper()),
            ("材料化学式", result["formula"]),
            ("电解质类型", result["electrolyte"]),
            ("结构类型", result["structure_type"]),
            ("Log_ASR", f'{result["Log_ASR"]:.5f}'),
            ("ASR（Ω·cm²）", f'{result["ASR"]:.5f}'),
            ("PCA可靠性得分（%）", f'{result["reliability_score"]:.5f}'),
            ("PCA适用域判断", result["pca_domain"]),
            ("PCA-kNN距离", f'{result["pca_distance"]:.5f}'),
        ]
        for i, (label, value) in enumerate(fields):
            with cols[i % 3]:
                css = "score" if "可靠性" in label else "value"
                st.markdown(f'<div class="label">{label}</div><div class="{css}">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if result["pca_domain"] == "outside":
            st.warning("该材料位于训练数据覆盖区域之外，模型预测可靠性较低，建议结合实验或更大范围数据谨慎使用。")
        if result["audit"].get("warning"):
            st.info(f'化学计量/价态提示：{result["audit"]["warning"]}')
    except Exception as exc:
        st.error(f"无法完成预测：{exc}")

st.caption("说明：ASR 为模型根据 Log_ASR 计算得到的 10^Log_ASR，单位为 Ω·cm²；PCA可靠性得分越高，表示输入材料与训练数据越相似。")
