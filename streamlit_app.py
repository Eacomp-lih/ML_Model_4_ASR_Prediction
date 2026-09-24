from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from prediction_core import ASRPredictor, ELECTROLYTES, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "data" / "data_923K_2026_09_09_v2.xlsx"
LOGO_FILE = ROOT / "assets" / "eacomp-logo.png"

st.set_page_config(page_title="钙钛矿型SOFC阴极材料650℃下ASR预测", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f5f8fc}[data-testid="stSidebar"]{background:linear-gradient(180deg,#f7fbff,#eef5ff);border-right:1px solid #d9e6f5}
[data-testid="stSidebar"] img{max-width:220px;margin:.4rem auto 1.2rem}.block-container{max-width:1180px;padding-top:1.7rem;padding-bottom:2rem}
.page-title{color:#173c67;font-size:2rem;font-weight:760;margin-bottom:.3rem}.page-subtitle{color:#63778f;margin-bottom:1.4rem}
.result-card{background:white;border:1px solid #d9e6f5;border-radius:14px;padding:1rem 1.2rem;margin-top:.8rem;box-shadow:0 5px 18px rgba(20,73,125,.07)}
.label{color:#6b7d90;font-size:.86rem;margin-bottom:.15rem}.value{color:#173c67;font-size:1.14rem;font-weight:680}.score{color:#117a68;font-size:1.28rem;font-weight:760}
.note{background:#edf5ff;border-left:4px solid #237ef5;padding:.75rem 1rem;border-radius:7px;color:#36516d}div.stButton>button{border-radius:9px;font-weight:650}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在加载模型与适用域分析器…")
def get_predictor():
    return ASRPredictor()


@st.cache_data(show_spinner=False)
def get_training_data():
    df = pd.read_excel(TRAINING_FILE, sheet_name="features")
    one_hot = [f"electrolyte_{name}" for name in ELECTROLYTES]
    df["电解质"] = df[one_hot].idxmax(axis=1).str.replace("electrolyte_", "", regex=False)
    df["ASR（Ω·cm²）"] = np.power(10.0, pd.to_numeric(df["Log_ASR"], errors="coerce"))
    return df


def heading(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def read_table(upload):
    return pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)


def normalize_rows(df, require_target=True):
    missing = {"Composition", "electrolyte"} - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{sorted(missing)}")
    out = df.copy()
    out["Composition"] = out["Composition"].astype(str).str.strip()
    out["electrolyte"] = out["electrolyte"].astype(str).str.strip()
    bad = sorted(set(out["electrolyte"]) - set(ELECTROLYTES))
    if bad:
        raise ValueError(f"不支持的电解质：{bad}")
    if "Log_ASR" in out:
        out["Log_ASR"] = pd.to_numeric(out["Log_ASR"], errors="coerce")
    elif "ASR" in out:
        out["ASR"] = pd.to_numeric(out["ASR"], errors="coerce")
        if (out["ASR"] <= 0).any():
            raise ValueError("ASR 必须为正数")
        out["Log_ASR"] = np.log10(out["ASR"])
    elif require_target:
        raise ValueError("需要 ASR 或 Log_ASR 列")
    if require_target and out["Log_ASR"].isna().any():
        raise ValueError("ASR/Log_ASR 中存在缺失值或非数值")
    if len(out) > 2000:
        raise ValueError("单次最多处理 2000 行")
    errors, predictor = [], get_predictor()
    for idx, row in out.iterrows():
        try:
            predictor.build_features(row["Composition"], row["electrolyte"])
        except Exception as exc:
            errors.append(f"第 {idx + 2} 行：{exc}")
    if errors:
        raise ValueError("；".join(errors[:8]) + ("……" if len(errors) > 8 else ""))
    return out


def training_matrix(rows):
    predictor = get_predictor()
    X = pd.concat([predictor.build_features(r.Composition, r.electrolyte)[0] for r in rows.itertuples()], ignore_index=True)
    return X, rows["Log_ASR"].astype(float).to_numpy()


def new_model(name, params):
    features = get_predictor().feature_columns
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                     ("scaler", StandardScaler())]), features)])
    if name == "RF":
        estimator = RandomForestRegressor(**params, random_state=33, n_jobs=-1)
    elif name == "SVR":
        estimator = SVR(**params, kernel="rbf")
    else:
        estimator = MLPRegressor(**params, activation="relu", solver="adam", early_stopping=True,
                                 validation_fraction=.15, n_iter_no_change=50, random_state=42)
    pipeline = Pipeline([("preprocessor", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) if name in {"SVR", "ANN"} else pipeline


def prediction_page():
    heading("ASR预测", "输入材料与电解质，输出650℃下预测结果和PCA适用域可靠性。")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.45, .8, .9])
        formula = c1.text_input("化学式", "Pr0.3Sr0.7CoO3", help="仅支持 ABO3 或 A2B2O5")
        model = c2.selectbox("模型", [n.upper() for n in MODEL_NAMES], index=1)
        electrolyte = c3.selectbox("电解质类型", list(ELECTROLYTES), index=1)
        run = st.button("运行预测", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("正在生成特征并计算预测…"):
                r = get_predictor().predict(formula, electrolyte, model.lower(), verbose=False)
            st.success("预测完成")
            fields = [("使用模型", r["model"].upper()), ("材料化学式", r["formula"]), ("电解质类型", r["electrolyte"]),
                      ("结构类型", r["structure_type"]), ("Log_ASR", f'{r["Log_ASR"]:.5f}'),
                      ("ASR（Ω·cm²）", f'{r["ASR"]:.5f}'), ("PCA可靠性得分（%）", f'{r["reliability_score"]:.5f}'),
                      ("PCA适用域判断", r["pca_domain"]), ("PCA-kNN距离", f'{r["pca_distance"]:.5f}')]
            cols = st.columns(3)
            for i, (label, value) in enumerate(fields):
                css = "score" if "可靠性" in label else "value"
                cols[i % 3].markdown(f'<div class="result-card"><div class="label">{label}</div><div class="{css}">{value}</div></div>', unsafe_allow_html=True)
            if r["pca_domain"] == "outside":
                st.warning("该材料位于训练数据覆盖区域之外，请谨慎使用预测结果。")
            if r["audit"].get("warning"):
                st.info(f'化学计量/价态提示：{r["audit"]["warning"]}')
        except Exception as exc:
            st.error(f"无法完成预测：{exc}")


def training_page():
    heading("模型训练", "上传自定义数据集、调整模型参数，并生成可下载的新模型。")
    st.markdown('<div class="note">文件需包含 Composition、electrolyte，以及 ASR 或 Log_ASR。训练仅在当前会话执行，不写入网站内置数据。</div>', unsafe_allow_html=True)
    upload = st.file_uploader("上传训练数据（CSV/XLSX）", type=["csv", "xlsx"], key="train")
    name = st.selectbox("训练模型", ["RF", "SVR", "ANN"])
    params = {}
    if name == "RF":
        a, b, c = st.columns(3); params["n_estimators"] = a.slider("决策树数量", 50, 500, 300, 50)
        params["max_depth"] = b.slider("最大深度", 2, 30, 15); params["min_samples_leaf"] = c.slider("叶节点最小样本", 1, 10, 1)
    elif name == "SVR":
        a, b, c = st.columns(3); params["C"] = a.number_input("C", .01, 100., 1.)
        params["epsilon"] = b.number_input("epsilon", .001, 1., .05); params["gamma"] = c.selectbox("gamma", ["scale", "auto"])
    else:
        a, b, c = st.columns(3); layer_text = a.text_input("隐藏层，例如 32,64,32", "32,64,32")
        params["alpha"] = b.number_input("L2正则化 alpha", .00001, 1., .01, format="%.5f")
        params["max_iter"] = c.slider("最大迭代次数", 200, 1500, 800, 100)
        try: params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in layer_text.split(",") if x.strip())
        except ValueError: params["hidden_layer_sizes"] = ()
    if st.button("开始训练", type="primary", disabled=upload is None, use_container_width=True):
        try:
            rows = normalize_rows(read_table(upload))
            if len(rows) < 20: raise ValueError("至少需要20条有效数据")
            if name == "ANN" and not params["hidden_layer_sizes"]: raise ValueError("隐藏层格式无效")
            X, y = training_matrix(rows); rng = np.random.default_rng(1234)
            test_idx = rng.choice(len(rows), max(4, round(len(rows) * .2)), replace=False)
            train_mask = np.ones(len(rows), dtype=bool); train_mask[test_idx] = False
            model = new_model(name, params)
            with st.spinner("模型训练中…"):
                model.fit(X.loc[train_mask], y[train_mask]); pred = model.predict(X.iloc[test_idx])
            st.success(f"训练完成：训练集 {train_mask.sum()} 条，测试集 {len(test_idx)} 条")
            c1, c2, c3 = st.columns(3); c1.metric("测试集 R²", f"{r2_score(y[test_idx], pred):.5f}")
            c2.metric("测试集 MAE", f"{mean_absolute_error(y[test_idx], pred):.5f}")
            c3.metric("测试集 RMSE", f"{np.sqrt(mean_squared_error(y[test_idx], pred)):.5f}")
            buf = io.BytesIO(); joblib.dump(model, buf)
            st.download_button("下载训练后的模型", buf.getvalue(), f"{name.lower()}_custom.joblib", "application/octet-stream")
        except Exception as exc: st.error(f"训练失败：{exc}")


def upload_page():
    heading("数据上传", "校验并整理材料实验数据，支持 ABO3 和 A2B2O5 两种结构。")
    st.markdown('<div class="note">必需列：Composition、electrolyte、ASR（Ω·cm²）。数据只保存在当前会话，请及时下载备份。</div>', unsafe_allow_html=True)
    upload = st.file_uploader("上传数据（CSV/XLSX）", type=["csv", "xlsx"], key="data")
    if upload:
        try:
            rows = normalize_rows(read_table(upload)); view = rows[["Composition", "electrolyte", "Log_ASR"]].copy()
            view["ASR"] = np.power(10., view["Log_ASR"]); st.session_state["uploaded_data"] = view
            st.success(f"通过校验：{len(view)} 条数据"); st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("下载校验后的CSV", view.to_csv(index=False).encode("utf-8-sig"), "validated_asr_data.csv", "text/csv")
        except Exception as exc: st.error(f"数据校验失败：{exc}")


def query_page():
    heading("数据查询", "浏览内置训练数据，并按化学式、电解质和650℃下ASR范围筛选。")
    data = get_training_data(); a, b = st.columns([1.5, 1])
    keyword = a.text_input("搜索化学式", placeholder="输入全部或部分化学式")
    electrolyte = b.multiselect("电解质类型", list(ELECTROLYTES), default=list(ELECTROLYTES))
    values = data["ASR（Ω·cm²）"].replace([np.inf, -np.inf], np.nan).dropna(); low, high = float(values.min()), float(values.max())
    span = st.slider("650℃下ASR范围（Ω·cm²）", low, high, (low, high))
    mask = data["电解质"].isin(electrolyte) & data["ASR（Ω·cm²）"].between(*span)
    if keyword: mask &= data["Composition"].astype(str).str.contains(keyword, case=False, regex=False)
    result = data.loc[mask, ["Composition", "电解质", "Log_ASR", "ASR（Ω·cm²）"]].copy().round(5)
    st.caption(f"共找到 {len(result)} 条记录"); st.dataframe(result, use_container_width=True, hide_index=True, height=520)


with st.sidebar:
    st.image(str(LOGO_FILE), use_container_width=True)
    st.caption("SOFC 阴极材料智能分析平台")
    page = st.radio("功能导航", ["ASR预测", "模型训练", "数据上传", "数据查询"], label_visibility="collapsed")
    st.divider(); st.caption("网页运行需要联网；计算过程在私有云端后端完成。")

{"ASR预测": prediction_page, "模型训练": training_page, "数据上传": upload_page, "数据查询": query_page}[page]()
