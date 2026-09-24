from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from prediction_core import ASRPredictor, ELECTROLYTES, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "data" / "data_923K_2026_09_09_v2.xlsx"
LOGO_FILE = ROOT / "assets" / "eacomp-logo.png"
APP_VERSION = "v0.3.0"

st.set_page_config(page_title="钙钛矿型SOFC阴极材料650℃下ASR预测", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f5f8fc}[data-testid="stSidebar"]{background:linear-gradient(180deg,#f7fbff,#eef5ff);border-right:1px solid #d9e6f5}
[data-testid="stHeader"]{background:rgba(245,248,252,.94)}
[data-testid="stSidebar"] img{max-width:220px;margin:.7rem auto 1.5rem}.block-container{max-width:1180px;padding-top:5.25rem!important;padding-bottom:2rem}
.page-title{color:#173c67;font-size:2rem;font-weight:760;margin-bottom:.3rem}.page-subtitle{color:#63778f;margin-bottom:1.4rem}
.result-card{background:white;border:1px solid #d9e6f5;border-radius:14px;padding:1rem 1.2rem;margin-top:.8rem;box-shadow:0 5px 18px rgba(20,73,125,.07)}
.label{color:#6b7d90;font-size:.86rem;margin-bottom:.15rem}.value{color:#173c67;font-size:1.14rem;font-weight:680}.score{color:#117a68;font-size:1.28rem;font-weight:760}
.note{background:#edf5ff;border-left:4px solid #237ef5;padding:.75rem 1rem;border-radius:7px;color:#36516d}div.stButton>button{border-radius:9px;font-weight:650}
.version{position:fixed;bottom:18px;left:24px;color:#94a3b8;font-size:.78rem;z-index:999}
.model-banner{background:linear-gradient(90deg,#173c67,#237ef5);color:#fff;border-radius:11px;padding:.78rem 1rem;margin:.25rem 0 1rem;font-weight:750;font-size:1.04rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.3rem}
[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent;border-radius:9px;padding:.58rem .65rem;color:#32465d;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#f0f5fb;color:#1768e5}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#eaf2ff;color:#1768e5;font-weight:700;box-shadow:inset -4px 0 #237ef5}
[data-testid="stSidebar"] [role="radiogroup"] input{display:none!important}
[data-testid="stSidebar"] [role="radiogroup"] label>div>div:first-child{display:none!important}
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
    if (out["Composition"] == "").any():
        raise ValueError("Composition 中存在空化学式")
    return out


def training_matrix(rows):
    predictor = get_predictor()
    X = pd.concat([predictor.build_features(r.Composition, r.electrolyte)[0] for r in rows.itertuples()], ignore_index=True)
    return X, rows["Log_ASR"].astype(float).to_numpy()


def new_model(name, params, random_seed):
    features = get_predictor().feature_columns
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                     ("scaler", StandardScaler())]), features)])
    if name == "RF":
        estimator = RandomForestRegressor(**params, random_state=random_seed, n_jobs=-1)
    elif name == "SVR":
        estimator = SVR(**params)
    else:
        estimator = MLPRegressor(**params, random_state=random_seed)
    pipeline = Pipeline([("preprocessor", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) if name in {"SVR", "ANN"} else pipeline


def custom_model(name, params, numeric_features, categorical_features, random_seed):
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                    ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))
    prep = ColumnTransformer(transformers)
    if name == "RF":
        estimator = RandomForestRegressor(**params, random_state=random_seed, n_jobs=-1)
    elif name == "SVR":
        estimator = SVR(**params)
    else:
        estimator = MLPRegressor(**params, random_state=random_seed)
    pipeline = Pipeline([("preprocessor", prep), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) if name in {"SVR", "ANN"} else pipeline


def parameter_controls(name, prefix):
    params = {}
    if name == "RF":
        a, b, c, d = st.columns(4)
        params["n_estimators"] = a.slider("决策树数量", 50, 1000, 300, 50, key=f"{prefix}_trees")
        depth = b.slider("最大深度（0=不限）", 0, 50, 15, key=f"{prefix}_depth")
        params["max_depth"] = None if depth == 0 else depth
        params["min_samples_split"] = c.slider("节点拆分最小样本", 2, 20, 2, key=f"{prefix}_split")
        params["min_samples_leaf"] = d.slider("叶节点最小样本", 1, 20, 1, key=f"{prefix}_leaf")
        e, f = st.columns(2)
        params["max_features"] = e.selectbox("每次拆分使用的特征", ["sqrt", "log2", 1.0], key=f"{prefix}_features")
        params["bootstrap"] = f.toggle("Bootstrap采样", value=True, key=f"{prefix}_bootstrap")
    elif name == "SVR":
        a, b, c, d = st.columns(4)
        params["kernel"] = a.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], key=f"{prefix}_kernel")
        params["C"] = b.number_input("惩罚参数 C", .001, 1000., 1., format="%.3f", key=f"{prefix}_c")
        params["epsilon"] = c.number_input("epsilon", .0001, 2., .05, format="%.4f", key=f"{prefix}_eps")
        params["gamma"] = d.selectbox("gamma", ["scale", "auto"], key=f"{prefix}_gamma")
        if params["kernel"] == "poly":
            params["degree"] = st.slider("多项式阶数", 2, 6, 3, key=f"{prefix}_degree")
    else:
        a, b, c, d = st.columns(4)
        layers = a.text_input("隐藏层，例如 32,64,32", "32,64,32", key=f"{prefix}_layers")
        params["activation"] = b.selectbox("激活函数", ["relu", "tanh", "logistic", "identity"], key=f"{prefix}_activation")
        params["solver"] = c.selectbox("优化器", ["adam", "sgd"], key=f"{prefix}_solver")
        params["alpha"] = d.number_input("L2正则化 alpha", .00001, 1., .01, format="%.5f", key=f"{prefix}_alpha")
        e, f, g = st.columns(3)
        params["learning_rate_init"] = e.number_input("初始学习率", .00001, 1., .001, format="%.5f", key=f"{prefix}_lr")
        params["learning_rate"] = f.selectbox("学习率策略", ["constant", "adaptive", "invscaling"], key=f"{prefix}_lr_type")
        params["max_iter"] = g.slider("最大迭代次数", 200, 3000, 800, 100, key=f"{prefix}_iter")
        params["early_stopping"] = st.toggle("启用早停", value=True, key=f"{prefix}_early")
        params["validation_fraction"] = .15
        params["n_iter_no_change"] = 50
        try:
            params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in layers.split(",") if x.strip())
        except ValueError:
            params["hidden_layer_sizes"] = ()
    return params


def evaluation_controls(prefix):
    st.markdown("##### 数据划分与评估")
    a, b, c, d = st.columns(4)
    train_pct = a.number_input("训练集（%）", 10, 90, 70, 5, key=f"{prefix}_train_pct")
    val_pct = b.number_input("验证集（%）", 5, 80, 15, 5, key=f"{prefix}_val_pct")
    test_pct = c.number_input("测试集（%）", 5, 80, 15, 5, key=f"{prefix}_test_pct")
    random_seed = d.number_input("随机种子", 0, 999999, 42, 1, key=f"{prefix}_seed")
    use_cv = st.toggle("启用 5-fold 交叉验证", value=False, key=f"{prefix}_cv",
                       help="在训练集与验证集组成的开发集上执行5折交叉验证；测试集始终保持独立。")
    if train_pct + val_pct + test_pct != 100:
        st.error("训练集、验证集和测试集比例之和必须等于 100%。")
    return int(train_pct), int(val_pct), int(test_pct), int(random_seed), use_cv


def metric_row(title, y_true, prediction):
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_true, prediction):.5f}")
    c2.metric("MAE", f"{mean_absolute_error(y_true, prediction):.5f}")
    c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, prediction)):.5f}")


def fit_and_report(X, y, model, filename, split_config):
    if len(X) < 10:
        raise ValueError("至少需要10条有效数据")
    if len(X) < 50:
        st.warning("当前数据量较少，测试指标波动可能较大，训练结果仅建议用于初步探索。")
    train_pct, val_pct, test_pct, random_seed, use_cv = split_config
    if train_pct + val_pct + test_pct != 100:
        raise ValueError("训练集、验证集和测试集比例之和必须等于100%")
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_pct / 100, random_state=random_seed)
    val_share = val_pct / (train_pct + val_pct)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=val_share, random_state=random_seed)
    if min(len(X_train), len(X_val), len(X_test)) < 2:
        raise ValueError("当前数据量与划分比例导致某个子集少于2条，请增加数据或调整比例")
    if use_cv and len(X_dev) < 10:
        raise ValueError("启用5-fold时，训练集与验证集合计至少需要10条数据")
    with st.spinner("模型训练中…"):
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        if use_cv:
            cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            scores = cross_validate(clone(model), X_dev, y_dev, cv=cv,
                                    scoring={"r2":"r2", "mae":"neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error"})
        model.fit(X_dev, y_dev)
        test_pred = model.predict(X_test)
    st.success(f"训练完成：训练集 {len(X_train)} 条，验证集 {len(X_val)} 条，测试集 {len(X_test)} 条")
    metric_row("验证集表现（仅用训练集拟合）", y_val, val_pred)
    if use_cv:
        st.markdown("**5-fold 交叉验证（开发集）**")
        c1, c2, c3 = st.columns(3)
        c1.metric("平均 R²", f'{scores["test_r2"].mean():.5f}', f'±{scores["test_r2"].std():.5f}')
        c2.metric("平均 MAE", f'{-scores["test_mae"].mean():.5f}', f'±{scores["test_mae"].std():.5f}')
        c3.metric("平均 RMSE", f'{-scores["test_rmse"].mean():.5f}', f'±{scores["test_rmse"].std():.5f}')
    metric_row("独立测试集表现（训练集+验证集重新拟合）", y_test, test_pred)
    buf = io.BytesIO(); joblib.dump(model, buf)
    st.download_button("下载训练后的模型", buf.getvalue(), filename, "application/octet-stream")


def prediction_page():
    heading("ASR预测", "输入材料与电解质，输出650℃下预测结果和PCA适用域可靠性。")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.45, .8, .9])
        formula = c1.text_input("化学式", "Pr0.3Sr0.7CoO3", help="输入可由当前描述符库解析的氧化物化学式")
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
                st.warning(f'可靠性说明：该样本位于训练数据适用域外，得分为 {r["reliability_score"]:.5f}%。这表示它与训练数据特征分布差异较大，请谨慎使用预测结果；该分数不是预测准确率。')
            else:
                st.info(f'可靠性说明：该样本位于训练数据适用域内，得分为 {r["reliability_score"]:.5f}%。分数表示它与训练数据特征分布的相似程度，越高通常越值得参考；该分数不是预测准确率。')
            if r["audit"].get("warning"):
                st.info(f'化学计量/价态提示：{r["audit"]["warning"]}')
        except Exception as exc:
            st.error(f"无法完成预测：{exc}")


MODEL_OPTIONS = {
    "RandomForestRegressor（随机森林回归）": "RF",
    "SVR（支持向量回归）": "SVR",
    "MLPRegressor（人工神经网络回归）": "ANN",
}


def training_page(mode="overview"):
    heading("模型训练", "配置数据划分、评估策略和模型专属超参数。")
    if mode == "overview":
        st.markdown('<div class="note">请从左侧缩进的子菜单选择“内置数据集训练”或“自定义数据集训练”。</div>', unsafe_allow_html=True)
        st.markdown("- **内置数据集训练**：使用网站既定特征，通过调整参数重新训练。\n- **自定义数据集训练**：上传自己的特征表，系统自动识别数值与分类特征。")
        return
    if mode == "builtin":
        st.markdown('<div class="note">使用网站内置训练集及既定特征，仅允许选择模型和调整超参数，不改变原始数据。</div>', unsafe_allow_html=True)
        selected_name = st.selectbox("选择回归模型", list(MODEL_OPTIONS), key="builtin_name")
        name = MODEL_OPTIONS[selected_name]
        st.markdown(f'<div class="model-banner">当前模型：{selected_name}</div>', unsafe_allow_html=True)
        params = parameter_controls(name, "builtin")
        split_config = evaluation_controls("builtin")
        if st.button("使用内置数据开始训练", type="primary", use_container_width=True):
            try:
                data = get_training_data(); features = get_predictor().feature_columns
                if name == "ANN" and not params.get("hidden_layer_sizes"): raise ValueError("隐藏层格式无效")
                model = new_model(name, params, split_config[3])
                fit_and_report(data[features], data["Log_ASR"].to_numpy(), model, f"builtin_{name.lower()}.joblib", split_config)
            except Exception as exc: st.error(f"训练失败：{exc}")
    else:
        st.markdown('<div class="note">默认将 Composition 和 ASR 之外的全部列作为训练特征。字符列会自动独热编码，数值列会自动补全并标准化。数据量过少时，结果可能不准确。</div>', unsafe_allow_html=True)
        example = pd.DataFrame({"Composition":["La0.6Sr0.4CoO3","Pr0.5Ba0.5CoO3","La0.8Sr0.2FeO3"],
                                "electrolyte":["GDC","SDC","zirconia"],"feature_1":[1.12,1.35,.98],
                                "feature_2":[25.4,18.2,31.7],"ASR":[.12,.21,.47]})
        with st.expander("查看上传格式示例"):
            st.dataframe(example, hide_index=True, use_container_width=True)
            st.download_button("下载示例 CSV", example.to_csv(index=False).encode("utf-8-sig"), "custom_training_example.csv", "text/csv")
        upload = st.file_uploader("上传训练数据（CSV/XLSX）", type=["csv", "xlsx"], key="custom_train")
        selected_name = st.selectbox("选择回归模型", list(MODEL_OPTIONS), key="custom_name")
        name = MODEL_OPTIONS[selected_name]
        st.markdown(f'<div class="model-banner">当前模型：{selected_name}</div>', unsafe_allow_html=True)
        params = parameter_controls(name, "custom")
        split_config = evaluation_controls("custom")
        if st.button("使用上传数据开始训练", type="primary", disabled=upload is None, use_container_width=True):
            try:
                rows = read_table(upload)
                if "Composition" not in rows or "ASR" not in rows: raise ValueError("自定义训练表必须包含 Composition 和 ASR 列")
                rows = rows.copy(); rows["ASR"] = pd.to_numeric(rows["ASR"], errors="coerce")
                if rows["ASR"].isna().any() or (rows["ASR"] <= 0).any(): raise ValueError("ASR 列必须全部为正数")
                feature_cols = [c for c in rows.columns if c not in {"Composition", "ASR", "Log_ASR"}]
                if not feature_cols: raise ValueError("除 Composition 和 ASR 外，至少需要一列训练特征")
                X = rows[feature_cols]; numeric = X.select_dtypes(include=np.number).columns.tolist()
                categorical = [c for c in feature_cols if c not in numeric]
                if name == "ANN" and not params.get("hidden_layer_sizes"): raise ValueError("隐藏层格式无效")
                model = custom_model(name, params, numeric, categorical, split_config[3])
                fit_and_report(X, np.log10(rows["ASR"].to_numpy()), model, f"custom_{name.lower()}.joblib", split_config)
            except Exception as exc: st.error(f"训练失败：{exc}")


def upload_page():
    heading("数据上传", "校验并整理材料实验数据。")
    st.markdown(f'<div class="note">必需列：Composition、electrolyte、ASR（Ω·cm²）。电解质仅支持 {", ".join(ELECTROLYTES)}。数据只保存在当前会话，请及时下载备份。</div>', unsafe_allow_html=True)
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
    axis_options = {
        "Goldschmidt 容忍因子": "tolerance_factor",
        "B位平均氧化态": "B_site_oxidation",
        "O/B 化学计量比": "oxygen_per_B",
        "组成熵": "composition_entropy",
        "Log_ASR": "Log_ASR",
    }
    axis_label = st.selectbox("交互图横轴", list(axis_options), help="选择具有材料学或预测意义的横轴变量")
    x_col = axis_options[axis_label]
    result = data.loc[mask, ["Composition", "电解质", "Log_ASR", "ASR（Ω·cm²）", x_col]].copy()
    result = result.loc[:, ~result.columns.duplicated()]
    st.caption(f"共找到 {len(result)} 条记录")
    if not result.empty:
        fig = px.scatter(result, x=x_col, y="ASR（Ω·cm²）", color="电解质", hover_name="Composition",
                         hover_data={"Log_ASR": ":.5f", "ASR（Ω·cm²）": ":.5f", x_col: ":.5f"},
                         labels={x_col: axis_label}, color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(marker={"size": 9, "opacity": .78, "line": {"width": .5, "color": "white"}})
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10), hovermode="closest", legend_title_text="电解质")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(result.round(5), use_container_width=True, hide_index=True, height=440)


with st.sidebar:
    st.image(str(LOGO_FILE), width=220)
    st.caption("SOFC 阴极材料智能分析平台")
    nav = {
        "⌁  ASR预测":"ASR预测",
        "⚙  模型训练":"模型训练",
        "　　↳ 内置数据集训练":"内置数据集训练",
        "　　↳ 自定义数据集训练":"自定义数据集训练",
        "⇧  数据上传":"数据上传",
        "▦  数据查询":"数据查询",
    }
    selected = st.radio("功能导航", list(nav), label_visibility="collapsed")
    page = nav[selected]
    st.markdown(f'<div class="version">当前版本：{APP_VERSION}</div>', unsafe_allow_html=True)

if page == "模型训练":
    training_page("overview")
elif page == "内置数据集训练":
    training_page("builtin")
elif page == "自定义数据集训练":
    training_page("custom")
else:
    {"ASR预测": prediction_page, "数据上传": upload_page, "数据查询": query_page}[page]()
