from __future__ import annotations

import io
import base64
import importlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import extra_streamlit_components as stx
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
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

import prediction_core
from prediction_core import ELECTROLYTES, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "data" / "data_923K_2026_09_09_v2.xlsx"
LOGO_FILE = ROOT / "assets" / "sanhuan-logo.png"
APP_VERSION = "v0.9.5"

st.set_page_config(page_title="钙钛矿型SOFC阴极材料650℃下ASR预测", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
if "ui_theme_saved" not in st.session_state:
    st.session_state.ui_theme_saved = "亮色"
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
.st-key-sidebar_utility{position:fixed;left:1rem;right:1rem;bottom:2.65rem;width:auto!important;z-index:998}
.st-key-sidebar_utility .stButton{width:auto!important}.st-key-sidebar_utility .stButton>button{width:auto!important;min-height:2.05rem!important;padding:.28rem .58rem!important;font-size:.84rem!important}
.st-key-sidebar_utility [data-testid="stCaptionContainer"]{padding:0 .58rem;color:#64748b;overflow-wrap:anywhere}
.st-key-sidebar_utility [data-testid="stBaseButton-secondary"]{background:transparent!important;border-color:transparent!important}
.st-key-sidebar_utility [data-testid="stBaseButton-secondary"]:hover{background:#e8f1ff!important;border-color:#c9dcf7!important}
.model-banner{background:linear-gradient(90deg,#173c67,#237ef5);color:#fff;border-radius:11px;padding:.78rem 1rem;margin:.25rem 0 1rem;font-weight:750;font-size:1.04rem}
.auth-title{text-align:center;color:#173c67;font-size:2rem;font-weight:760;margin:.35rem 0 .3rem}.auth-subtitle{text-align:center;color:#63778f;margin-bottom:1.5rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.3rem}
[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent;border-radius:9px;padding:.58rem .65rem;color:#32465d;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#f0f5fb;color:#1768e5}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#eaf2ff;color:#1768e5;font-weight:700;box-shadow:inset -4px 0 #237ef5}
[data-testid="stSidebar"] [role="radiogroup"] input{display:none!important}
[data-testid="stSidebar"] [role="radiogroup"] label>div>div:first-child{display:none!important}
[data-testid="stSidebar"] .stButton,[data-testid="stSidebar"] .stButton>button{width:100%}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;text-align:left;padding:.55rem .7rem}
</style>""", unsafe_allow_html=True)

if st.session_state.get("ui_theme_saved") == "暗色":
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:#070b14;color:#f1f5f9}
    [data-testid="stHeader"]{background:rgba(7,11,20,.97)}
    [data-testid="stSidebar"]{background:#0b1220;border-right:1px solid #334155}
    .page-title,.value,.auth-title{color:#f8fafc}.page-subtitle,.label,.auth-subtitle{color:#b6c2d2}
    .result-card,[data-testid="stForm"],[data-testid="stExpander"]{background:#111c2e;border:1px solid #3b4c65;box-shadow:0 6px 20px rgba(0,0,0,.28)}
    .note{background:#172554;color:#eff6ff;border-left-color:#60a5fa}
    [data-testid="stWidgetLabel"],p,li,h1,h2,h3,h4,h5,h6,label{color:#f1f5f9!important}
    input,textarea,[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#18243a!important;color:#fff!important;border-color:#64748b!important}
    [data-baseweb="tab-list"]{background:#111c2e;border:1px solid #334155;border-radius:9px}[data-baseweb="tab"]{color:#dbeafe}
    [data-baseweb="popover"],[role="listbox"]{background:#18243a!important;color:#fff!important}
    [data-testid="stDataFrame"],iframe{background:#111c2e!important;border:1px solid #3b4c65}
    [data-testid="stMetric"]{background:#111c2e;border:1px solid #3b4c65;border-radius:10px;padding:.55rem}
    [data-testid="stBaseButton-secondary"]{background:#18243a;color:#f8fafc;border-color:#64748b}
    .st-key-sidebar_utility [data-testid="stBaseButton-secondary"]:hover{background:#243552!important;border-color:#7c93b3!important}
    </style>
    """, unsafe_allow_html=True)

pio.templates.default = "plotly_dark" if st.session_state.get("ui_theme_saved") == "暗色" else "plotly_white"

if st.session_state.get("ui_font_size") == "大":
    st.markdown("<style>html,body,[class*=css]{font-size:17px}.page-title{font-size:2.15rem}</style>", unsafe_allow_html=True)
if st.session_state.get("ui_density") == "紧凑":
    st.markdown("<style>.block-container{padding-top:4.3rem!important}.result-card{padding:.72rem .9rem;margin-top:.55rem}[data-testid=stVerticalBlock]{gap:.65rem}</style>", unsafe_allow_html=True)


def get_auth_config():
    try:
        auth = dict(st.secrets.get("auth", {}))
    except Exception:
        auth = {}
    url = str(auth.get("supabase_url", "")).strip().strip('"\'').rstrip("/")
    key = str(auth.get("supabase_anon_key", "")).strip().strip('"\'')
    return auth, url, key


def service_request(method, url, *, attempts=3, timeout=(5, 15), **kwargs):
    """Retry temporary network/cold-start failures without retrying user input errors."""
    last_exception = None
    response = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code != 429 and response.status_code < 500:
                return response
        except requests.RequestException as exc:
            last_exception = exc
        if attempt < attempts - 1:
            time.sleep(1.5 * (2 ** attempt))
    if response is not None:
        return response
    raise last_exception or requests.ConnectionError("账号服务连接失败")


COOKIE_MANAGER = stx.CookieManager(key="asr_auth_cookies")


def get_cookie_manager():
    return COOKIE_MANAGER


def remember_login(payload):
    st.session_state.auth_access_token = payload.get("access_token")
    st.session_state.auth_email = payload.get("user", {}).get("email", "")
    st.session_state.auth_username = payload.get("user", {}).get("user_metadata", {}).get("username", "")
    refresh_token = payload.get("refresh_token")
    if refresh_token:
        get_cookie_manager().set("asr_refresh_token", refresh_token,
                                 expires_at=datetime.now() + timedelta(days=30), key="save_refresh_token")


def clear_login():
    for item in ["auth_access_token", "auth_email", "auth_username"]:
        st.session_state.pop(item, None)
    try:
        get_cookie_manager().delete("asr_refresh_token", key="delete_refresh_token")
    except Exception:
        pass


def password_issues(password):
    issues = []
    if len(password) < 8:
        issues.append("至少8位")
    if not any(char.isupper() for char in password):
        issues.append("至少1个大写字母")
    if not any(char.islower() for char in password):
        issues.append("至少1个小写字母")
    if not any(char.isdigit() for char in password):
        issues.append("至少1个数字")
    return issues


def save_browser_ui_state():
    manager = get_cookie_manager()
    expires_at = datetime.now() + timedelta(days=30)
    manager.set("asr_active_page", st.session_state.get("active_page", "ASR预测"),
                expires_at=expires_at, key="save_active_page")
    manager.set("asr_active_module", st.session_state.get("active_module", "ASR预测"),
                expires_at=expires_at, key="save_active_module")
    manager.set("asr_training_menu", "1" if st.session_state.get("training_menu_open", False) else "0",
                expires_at=expires_at, key="save_training_menu")


def restore_browser_ui_state():
    if st.session_state.get("browser_ui_restored"):
        return
    manager = get_cookie_manager()
    valid_pages = {"ASR预测", "内置数据集训练", "自定义数据集训练", "数据上传", "数据查询", "设置"}
    valid_modules = {"ASR预测", "模型训练", "数据上传", "数据查询", "设置"}
    saved_page = manager.get("asr_active_page")
    saved_module = manager.get("asr_active_module")
    saved_theme = manager.get("asr_ui_theme")
    if saved_page in valid_pages:
        st.session_state.active_page = saved_page
    if saved_module in valid_modules:
        st.session_state.active_module = saved_module
    st.session_state.training_menu_open = manager.get("asr_training_menu") == "1"
    theme_changed = saved_theme in {"亮色", "暗色"} and saved_theme != st.session_state.get("ui_theme_saved")
    if saved_theme in {"亮色", "暗色"}:
        st.session_state.ui_theme_saved = saved_theme
    st.session_state.browser_ui_restored = True
    if theme_changed:
        st.rerun()


def authentication_gate():
    auth, url, key = get_auth_config()
    if not auth.get("required", False):
        return
    if not url or not key:
        st.error("账号系统已启用，但尚未配置 Supabase URL 或匿名公钥。请联系管理员。")
        st.stop()
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc or not parsed_url.netloc.endswith(".supabase.co"):
        st.error("账号系统配置错误：supabase_url 必须是 https://<项目ID>.supabase.co 格式的 Project URL，不能使用 Supabase 控制台页面地址。")
        st.stop()
    if not st.session_state.get("auth_access_token"):
        refresh_token = get_cookie_manager().get("asr_refresh_token")
        if refresh_token:
            try:
                refresh_response = service_request("POST",
                    f"{url}/auth/v1/token?grant_type=refresh_token",
                    headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"refresh_token": refresh_token})
                if refresh_response.ok:
                    remember_login(refresh_response.json())
                    st.rerun()
                elif refresh_response.status_code in {400, 401, 403}:
                    clear_login()
                else:
                    st.error("账号服务正在恢复或暂时繁忙，已保留登录凭据。请稍候刷新页面重试。")
                    st.stop()
            except requests.RequestException:
                st.error("账号服务可能正在从休眠状态启动，已保留登录凭据。请等待约30秒后刷新页面，无需重新登录。")
                st.stop()
        elif st.session_state.get("cookie_load_attempts", 0) < 3:
            st.session_state.cookie_load_attempts = st.session_state.get("cookie_load_attempts", 0) + 1
            time.sleep(.35)
            st.rerun()
    if st.session_state.get("auth_access_token"):
        return
    _, auth_col, _ = st.columns([1, 1.15, 1])
    with auth_col:
        logo_left, logo_col, logo_right = st.columns([.2, .6, .2])
        with logo_col:
            st.image(str(LOGO_FILE), use_container_width=True)
        st.markdown('<div class="auth-title">用户登录</div><div class="auth-subtitle">注册账号并登录后方可使用材料预测平台</div>', unsafe_allow_html=True)
        if st.session_state.pop("account_deleted", False):
            st.success("账号已永久注销。")
        login_tab, register_tab = st.tabs(["登录", "注册"])
        headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        with login_tab:
            email = st.text_input("邮箱", key="login_email")
            password = st.text_input("密码", type="password", key="login_password")
            locked_until = st.session_state.get("login_locked_until")
            locked = bool(locked_until and datetime.now() < locked_until)
            if locked:
                remaining = max(1, int((locked_until - datetime.now()).total_seconds() // 60) + 1)
                st.error(f"登录失败次数过多，请在约 {remaining} 分钟后重试。")
            if st.button("登录", type="primary", use_container_width=True, disabled=locked):
                try:
                    response = service_request("POST", f"{url}/auth/v1/token?grant_type=password", headers=headers,
                                               json={"email": email.strip(), "password": password})
                    if response.ok:
                        st.session_state.login_failed_attempts = 0
                        st.session_state.pop("login_locked_until", None)
                        payload = response.json()
                        remember_login(payload)
                        st.rerun()
                    elif response.status_code == 429 or response.status_code >= 500:
                        st.error("账号服务正在启动或暂时繁忙，请等待约30秒后再次点击登录。此次不会计入登录失败次数。")
                    else:
                        attempts = st.session_state.get("login_failed_attempts", 0) + 1
                        st.session_state.login_failed_attempts = attempts
                        if attempts >= 5:
                            st.session_state.login_locked_until = datetime.now() + timedelta(minutes=15)
                            st.error("连续登录失败5次，当前会话已锁定15分钟。你也可以使用“忘记密码”。")
                        else:
                            st.error(f"登录失败，请检查邮箱、密码或邮箱验证状态。还可尝试 {5 - attempts} 次。")
                except requests.RequestException:
                    st.error("账号服务可能正在从休眠状态启动。请等待约30秒后重试；此次不会计入登录失败次数。")
            with st.expander("忘记密码"):
                reset_email = st.text_input("注册邮箱", key="reset_email")
                if st.button("发送密码重置邮件", disabled=not reset_email.strip(), use_container_width=True):
                    try:
                        response = service_request("POST", f"{url}/auth/v1/recover", headers=headers,
                                                   json={"email": reset_email.strip()})
                        if response.ok:
                            st.success("若该邮箱已注册，密码重置邮件将很快发出。请检查收件箱和垃圾邮件。")
                        else:
                            st.error("暂时无法发送重置邮件。请确认邮箱格式，稍后重试。")
                    except requests.RequestException:
                        st.error("账号服务暂时不可用。请检查网络后重试。")
        with register_tab:
            new_email = st.text_input("注册邮箱", key="register_email")
            new_password = st.text_input("设置密码", type="password", key="register_password",
                                         help="至少8位，并包含大写字母、小写字母和数字。")
            confirm_password = st.text_input("确认密码", type="password", key="register_password_confirm")
            if st.button("创建账号", use_container_width=True):
                issues = password_issues(new_password)
                if issues:
                    st.error("密码强度不足：" + "、".join(issues) + "。")
                elif new_password != confirm_password:
                    st.error("两次输入的密码不一致。")
                else:
                    try:
                        response = service_request("POST", f"{url}/auth/v1/signup", headers=headers,
                                                   json={"email": new_email.strip(), "password": new_password})
                        if response.ok:
                            payload = response.json()
                            if payload.get("access_token"):
                                remember_login(payload)
                                st.rerun()
                            st.success("注册成功。若已开启邮箱验证，请先查收验证邮件，然后返回登录。")
                        else:
                            content_type = response.headers.get("content-type", "")
                            message = response.json().get("msg", "注册失败") if content_type.startswith("application/json") else "注册失败"
                            st.error(message)
                    except requests.RequestException:
                        st.error("暂时无法连接账号认证服务，请检查 Supabase Project URL 和网络状态。")
    st.stop()


authentication_gate()
restore_browser_ui_state()


@st.cache_resource(show_spinner="正在加载模型与适用域分析器…")
def _load_predictor(version):
    # Streamlit can rerun this file while retaining the old imported module.
    # Reload it when the app version changes so new predictor methods are present.
    return importlib.reload(prediction_core).ASRPredictor()


def get_predictor():
    return _load_predictor(APP_VERSION)


@st.cache_data(show_spinner=False)
def get_training_data():
    df = pd.read_excel(TRAINING_FILE, sheet_name="features")
    one_hot = [f"electrolyte_{name}" for name in ELECTROLYTES]
    df["电解质"] = df[one_hot].idxmax(axis=1).str.replace("electrolyte_", "", regex=False)
    df["ASR（Ω·cm²）"] = np.power(10.0, pd.to_numeric(df["Log_ASR"], errors="coerce"))
    return df


def t(chinese, english):
    return chinese


def heading(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def add_pca_applicability_domain(figure, target_scores):
    """Overlay the notebook's kNN/conformal 85% PCA applicability domain."""
    grid = get_predictor().pca_domain_grid(target_scores)
    p_values = np.asarray(grid["p"], dtype=float)
    # Plot only the inside mask. Using a constraint contour directly causes
    # Plotly to fill the complement of the domain in some versions.
    inside_mask = np.where(p_values > grid["alpha"], 1.0, np.nan)
    boundary = go.Contour(
        x=grid["x"], y=grid["y"], z=inside_mask,
        contours={"start": 0.5, "end": 1.0, "size": 0.5, "coloring": "fill", "showlines": False},
        colorscale=[[0.0, "rgba(35,126,245,0.10)"], [1.0, "rgba(35,126,245,0.10)" ]],
        connectgaps=False, line={"color": "#237ef5", "width": 2.2},
        fillcolor="rgba(35,126,245,0.10)",
        name="85%适用域", showlegend=True, showscale=False, hoverinfo="skip",
    )
    outline = go.Contour(
        x=grid["x"], y=grid["y"], z=p_values,
        contours={"start": grid["alpha"], "end": grid["alpha"], "size": 1,
                   "coloring": "lines", "showlines": True},
        line={"color": "#1768c4", "width": 2.4}, showscale=False,
        showlegend=False, hoverinfo="skip",
    )
    figure.add_trace(boundary)
    figure.add_trace(outline)
    figure.data = (figure.data[-2], figure.data[-1]) + figure.data[:-2]
    figure.update_layout(
        title={"text": figure.layout.title.text + "（85%适用域）"},
        legend_title_text="样本与适用域",
        height=560,
        margin=dict(l=42, r=18, t=64, b=46),
    )
    return grid


def add_operation(category, summary, status="成功"):
    history = st.session_state.setdefault("operation_history", [])
    history.insert(0, {"时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "类型": category, "内容": summary, "状态": status})
    del history[100:]


def friendly_error(title, exc, suggestions):
    st.error(f"{title}：{exc}")
    with st.expander("查看可能的解决方法"):
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def prediction_exports(result):
    export_fields = {
        "模型": result["model"].upper(), "材料": result["formula"], "电解质": result["electrolyte"],
        "结构类型": result["structure_type"], "Log_ASR": result["Log_ASR"],
        "ASR_Ohm_cm2": result["ASR"], "PCA可靠性得分_pct": result["reliability_score"],
        "可靠性等级": result["reliability_level"], "适用域": result["pca_domain"],
        "PCA_kNN距离": result["pca_distance"],
    }
    frame = pd.DataFrame([export_fields])
    csv_data = frame.to_csv(index=False).encode("utf-8-sig")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="预测结果")
    pdf_buffer = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    pdf.setTitle("ASR Prediction Report")
    pdf.setFont("STSong-Light", 16)
    pdf.drawString(55, 800, "钙钛矿型SOFC阴极材料650℃下ASR预测报告")
    pdf.setFont("STSong-Light", 10)
    y = 765
    for label, value in export_fields.items():
        display = f"{value:.5f}" if isinstance(value, float) else str(value)
        pdf.drawString(60, y, f"{label}：{display}")
        y -= 24
    pdf.drawString(60, y - 8, "免责声明：本结果由机器学习模型生成，仅供科研参考，不替代实验验证。")
    pdf.save()
    name = f'{result["formula"]}_{result["model"].lower()}_asr_prediction'
    c1, c2, c3 = st.columns(3)
    c1.download_button("下载 CSV", csv_data, f"{name}.csv", "text/csv", use_container_width=True)
    c2.download_button("下载 Excel", excel_buffer.getvalue(), f"{name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    c3.download_button("下载 PDF 报告", pdf_buffer.getvalue(), f"{name}.pdf", "application/pdf", use_container_width=True)


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


def metric_values(y_true, prediction):
    return {
        "R²": float(r2_score(y_true, prediction)),
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
    }


def training_visualizations(y_true, prediction, title):
    chart = pd.DataFrame({"真实值": y_true, "预测值": prediction})
    chart["残差"] = chart["真实值"] - chart["预测值"]
    left, right = st.columns(2)
    parity = px.scatter(chart, x="真实值", y="预测值", title=f"{title}：真实值—预测值")
    low = float(min(chart["真实值"].min(), chart["预测值"].min()))
    high = float(max(chart["真实值"].max(), chart["预测值"].max()))
    parity.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"dash":"dash", "color":"#64748b"})
    left.plotly_chart(parity, use_container_width=True, config={"displaylogo": False})
    residual = px.scatter(chart, x="预测值", y="残差", title=f"{title}：残差图")
    residual.add_hline(y=0, line_dash="dash", line_color="#64748b")
    right.plotly_chart(residual, use_container_width=True, config={"displaylogo": False})
    hist = px.histogram(chart, x="残差", nbins=min(30, max(8, len(chart) // 3)), title=f"{title}：误差分布")
    st.plotly_chart(hist, use_container_width=True, config={"displaylogo": False})


def save_training_history(record):
    history = st.session_state.setdefault("training_history", [])
    history.insert(0, record)
    del history[5:]


def render_training_history():
    history = st.session_state.get("training_history", [])
    if not history:
        return
    st.markdown("### 最近训练任务")
    table = pd.DataFrame([{k: v for k, v in item.items() if k not in {"model_bytes", "parameters", "cv_scores"}} for item in history])
    st.dataframe(table, use_container_width=True, hide_index=True)
    metric_table = table[["模型", "测试R²", "测试MAE", "测试RMSE"]].copy()
    metric_long = metric_table.melt(id_vars="模型", var_name="指标", value_name="数值")
    st.plotly_chart(px.bar(metric_long, x="模型", y="数值", color="指标", barmode="group", title="最近训练模型指标对比"),
                    use_container_width=True, config={"displaylogo": False})
    best = max(history, key=lambda item: item["测试R²"])
    st.download_button(f'下载当前最佳模型（{best["模型"]}，测试R²={best["测试R²"]:.5f}）',
                       best["model_bytes"], best["文件名"], "application/octet-stream")


def fit_and_report(X, y, model, filename, split_config, *, model_name, parameters, source):
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
    training_visualizations(y_test, test_pred, "独立测试集")
    if use_cv:
        cv_frame = pd.DataFrame({
            "折次": np.arange(1, 6), "R²": scores["test_r2"],
            "MAE": -scores["test_mae"], "RMSE": -scores["test_rmse"],
        })
        cv_long = cv_frame.melt(id_vars="折次", var_name="指标", value_name="数值")
        st.plotly_chart(px.line(cv_long, x="折次", y="数值", color="指标", markers=True, title="5-fold 各折评估结果"),
                        use_container_width=True, config={"displaylogo": False})
    buf = io.BytesIO(); joblib.dump(model, buf)
    model_bytes = buf.getvalue()
    test_metrics = metric_values(y_test, test_pred)
    save_training_history({
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "来源": source, "模型": model_name,
        "数据量": len(X), "随机种子": random_seed, "训练/验证/测试": f"{train_pct}/{val_pct}/{test_pct}",
        "5-fold": "是" if use_cv else "否", "测试R²": test_metrics["R²"],
        "测试MAE": test_metrics["MAE"], "测试RMSE": test_metrics["RMSE"],
        "参数": str(parameters), "parameters": parameters, "model_bytes": model_bytes, "文件名": filename,
    })
    add_operation("模型训练", f"{source} / {model_name} / {len(X)}条 / 测试R²={test_metrics['R²']:.5f}")
    st.download_button("下载训练后的模型", model_bytes, filename, "application/octet-stream")


def prediction_page():
    heading(t("ASR预测", "ASR Prediction"), t("输入材料与电解质，输出650℃下预测结果和PCA适用域可靠性。",
                                               "Enter a material and electrolyte to predict ASR at 650°C with PCA reliability."))
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.45, .8, .9])
        formula = c1.text_input("化学式", "Pr0.3Sr0.7CoO3", help="输入可由当前描述符库解析的氧化物化学式")
        model_options = [n.upper() for n in MODEL_NAMES]
        default_model = st.session_state.get("default_prediction_model", "RF")
        default_electrolyte = st.session_state.get("default_electrolyte", "GDC")
        model = c2.selectbox("模型", model_options, index=model_options.index(default_model) if default_model in model_options else 1)
        electrolyte = c3.selectbox("电解质类型", list(ELECTROLYTES),
                                   index=list(ELECTROLYTES).index(default_electrolyte) if default_electrolyte in ELECTROLYTES else 1)
        run = st.button("运行预测", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("正在生成特征并计算预测…"):
                r = get_predictor().predict(formula, electrolyte, model.lower(), verbose=False)
            st.session_state.last_prediction = r
            add_operation("ASR预测", f'{formula} / {electrolyte} / {model} / ASR={r["ASR"]:.5f}')
            st.success("预测完成")
            fields = [("使用模型", r["model"].upper()), ("材料化学式", r["formula"]), ("电解质类型", r["electrolyte"]),
                      ("结构类型", r["structure_type"]), ("Log_ASR", f'{r["Log_ASR"]:.5f}'),
                      ("ASR（Ω·cm²）", f'{r["ASR"]:.5f}'), ("PCA可靠性得分（%）", f'{r["reliability_score"]:.5f}'),
                      ("可靠性等级", r["reliability_level"]), ("PCA适用域判断", r["pca_domain"]),
                      ("PCA-kNN距离", f'{r["pca_distance"]:.5f}')]
            cols = st.columns(3)
            for i, (label, value) in enumerate(fields):
                css = "score" if "可靠性" in label else "value"
                cols[i % 3].markdown(f'<div class="result-card"><div class="label">{label}</div><div class="{css}">{value}</div></div>', unsafe_allow_html=True)
            if r["pca_domain"] == "outside":
                st.warning(f'可靠性说明：该样本位于训练数据适用域外，得分为 {r["reliability_score"]:.5f}%。这表示它与训练数据特征分布差异较大，请谨慎使用预测结果；该分数不是预测准确率。')
            else:
                st.info(f'可靠性说明：该样本位于训练数据适用域内，得分为 {r["reliability_score"]:.5f}%。分数表示它与训练数据特征分布的相似程度，越高通常越值得参考；该分数不是预测准确率。')
            level_explanation = {
                "高": "输入材料与训练集特征分布高度相似，预测通常具有较好的参考价值。",
                "中": "输入材料与训练集存在一定相似性，但部分特征可能处于样本稀疏区域，建议结合实验或其他模型复核。",
                "低": "输入材料远离主要训练分布，属于外推预测，应谨慎解释并优先进行实验验证。",
            }
            st.markdown(f'**{r["reliability_level"]}可靠性：** {level_explanation[r["reliability_level"]]}')
            st.markdown("### PCA适用域分析")
            details = r["pca_details"]
            pca_frame = pd.DataFrame(details["training_scores"], columns=["PC1", "PC2"])
            if len(pca_frame) > 1500:
                pca_frame = pca_frame.sample(1500, random_state=42)
            pca_frame["类型"] = "训练数据"
            target_type = "输入材料：域内" if r["pca_domain"] == "inside" else "输入材料：域外"
            target_frame = pd.DataFrame([[*details["target_scores"], target_type]], columns=["PC1", "PC2", "类型"])
            pca_plot = pd.concat([pca_frame, target_frame], ignore_index=True)
            fig = px.scatter(pca_plot, x="PC1", y="PC2", color="类型", symbol="类型",
                             color_discrete_map={"训练数据":"#94a3b8", "输入材料：域内":"#ff8c33",
                                                 "输入材料：域外":"#c51b7d"},
                             title="输入材料在PCA空间中的位置")
            fig.update_traces(marker={"size": 7, "opacity": .65})
            fig.update_traces(selector={"name": target_type}, marker={"size": 16, "opacity": 1,
                                                                     "symbol": "triangle-up" if r["pca_domain"] == "inside" else "x",
                                                                     "line":{"width":2,"color":"white"}})
            add_pca_applicability_domain(fig, [details["target_scores"]])
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.caption("绿色区域为基于第5近邻距离和校准集绘制的85% PCA适用域；输入材料落在区域内/外与上方判断一致。二维投影仅反映前两个主成分。")
            nearest = pd.DataFrame(details["nearest_materials"]).rename(columns={
                "electrolyte":"电解质", "Log_ASR":"Log_ASR", "distance":"PCA标准化距离"
            })
            st.markdown("#### 最近的3个训练材料")
            st.dataframe(nearest.round(5), use_container_width=True, hide_index=True)
            st.markdown("### 预测可解释性")
            st.caption("下图表示将单个特征替换为训练集中位数后，当前预测值的变化。正值表示该特征使预测Log_ASR升高，负值表示使其降低；结果反映局部模型敏感度，不代表因果关系。")
            impacts = pd.DataFrame(r["feature_impacts"]).sort_values("impact")
            impact_fig = px.bar(impacts, x="impact", y="feature", orientation="h", color="impact",
                                color_continuous_scale="RdBu_r", labels={"impact":"对Log_ASR的局部影响", "feature":"特征"})
            st.plotly_chart(impact_fig, use_container_width=True, config={"displaylogo": False})
            st.dataframe(impacts[["feature", "value", "reference_median", "impact"]].round(5),
                         use_container_width=True, hide_index=True)
            if r["audit"].get("warning"):
                st.info(f'化学计量/价态提示：{r["audit"]["warning"]}')
            st.markdown("### 导出预测结果")
            prediction_exports(r)
            st.caption("免责声明：预测结果由机器学习模型生成，仅供科研筛选与分析参考，不能替代实验测试或工程验证。")
        except Exception as exc:
            add_operation("ASR预测", f"{formula} / {electrolyte} / {model}", "失败")
            friendly_error("无法完成预测", exc, [
                "检查化学式是否书写完整，并使用标准元素符号和数字。",
                "确认所选电解质属于平台支持的五种类型。",
                "若模型或特征加载失败，请刷新页面后重试，并在持续失败时提交问题反馈。",
            ])
    batch_prediction_section()


def batch_prediction_section():
    st.divider()
    st.markdown("### 批量预测")
    st.caption("上传 CSV 或 XLSX 文件，必须包含 Composition 和 electrolyte 两列；单次最多预测100条材料。")
    example = pd.DataFrame({
        "Composition": ["Pr0.3Sr0.7CoO3", "La0.6Sr0.4CoO3"],
        "electrolyte": ["GDC", "SDC"],
    })
    with st.expander("查看批量预测文件示例"):
        st.dataframe(example, use_container_width=True, hide_index=True)
        st.download_button("下载示例 CSV", example.to_csv(index=False).encode("utf-8-sig"),
                           "batch_prediction_example.csv", "text/csv")
    left, right = st.columns([1.5, 1])
    upload = left.file_uploader("上传批量预测文件", type=["csv", "xlsx"], key="batch_prediction_file")
    model_options = [name.upper() for name in MODEL_NAMES]
    default_model = st.session_state.get("default_prediction_model", "RF")
    model = right.selectbox("批量预测模型", model_options,
                            index=model_options.index(default_model) if default_model in model_options else 1,
                            key="batch_prediction_model")
    if st.button("运行批量预测", type="primary", disabled=upload is None, use_container_width=True):
        try:
            rows = read_table(upload)
            missing = {"Composition", "electrolyte"} - set(rows.columns)
            if missing:
                raise ValueError(f"缺少必要列：{sorted(missing)}")
            if not 1 <= len(rows) <= 100:
                raise ValueError("批量预测文件必须包含1至100条数据")
            rows = rows[["Composition", "electrolyte"]].copy()
            rows["Composition"] = rows["Composition"].astype(str).str.strip()
            rows["electrolyte"] = rows["electrolyte"].astype(str).str.strip()
            invalid = sorted(set(rows["electrolyte"]) - set(ELECTROLYTES))
            if invalid:
                raise ValueError(f"存在不支持的电解质：{invalid}")
            output = []
            pca_targets = []
            nearest_rows = []
            training_scores = None
            explained_variance = None
            progress = st.progress(0, text="正在进行批量预测…")
            predictor = get_predictor()
            for index, row in enumerate(rows.itertuples(index=False), start=1):
                try:
                    result = predictor.predict(row.Composition, row.electrolyte, model.lower(),
                                               verbose=False, include_details=True, include_impacts=False)
                    output.append({
                        "Composition": row.Composition, "electrolyte": row.electrolyte, "模型": model,
                        "Log_ASR": result["Log_ASR"], "ASR（Ω·cm²）": result["ASR"],
                        "PCA可靠性得分（%）": result["reliability_score"],
                        "可靠性等级": result["reliability_level"], "适用域": result["pca_domain"], "错误": "",
                    })
                    details = result["pca_details"]
                    if training_scores is None:
                        training_scores = details["training_scores"]
                        explained_variance = details["explained_variance_ratio"]
                    pca_targets.append({
                        "输入序号": index, "Composition": row.Composition, "electrolyte": row.electrolyte,
                        "PC1": details["target_scores"][0], "PC2": details["target_scores"][1],
                        "PCA可靠性得分（%）": result["reliability_score"],
                        "可靠性等级": result["reliability_level"], "适用域": result["pca_domain"],
                    })
                    for rank, neighbor in enumerate(details["nearest_materials"], start=1):
                        nearest_rows.append({
                            "输入序号": index, "输入材料": row.Composition, "输入电解质": row.electrolyte,
                            "排名": rank, "最近训练材料": neighbor["Composition"],
                            "训练材料电解质": neighbor["electrolyte"],
                            "训练材料Log_ASR": neighbor["Log_ASR"],
                            "PCA标准化距离": neighbor["distance"],
                        })
                except Exception as exc:
                    output.append({"Composition": row.Composition, "electrolyte": row.electrolyte,
                                   "模型": model, "错误": str(exc)})
                progress.progress(index / len(rows), text=f"已完成 {index}/{len(rows)}")
            progress.empty()
            result_frame = pd.DataFrame(output)
            success_count = int((result_frame["错误"] == "").sum())
            failed_count = len(result_frame) - success_count
            st.session_state.batch_prediction_result = result_frame
            nearest_frame = pd.DataFrame(nearest_rows)
            st.session_state.batch_nearest_materials = nearest_frame
            st.session_state.batch_pca_payload = {
                "training_scores": training_scores or [], "targets": pca_targets,
                "explained_variance_ratio": explained_variance or [],
            }
            st.session_state.batch_prediction_csv = result_frame.to_csv(index=False).encode("utf-8-sig")
            batch_excel = io.BytesIO()
            with pd.ExcelWriter(batch_excel, engine="openpyxl") as writer:
                result_frame.to_excel(writer, index=False, sheet_name="批量预测结果")
                if not nearest_frame.empty:
                    nearest_frame.to_excel(writer, index=False, sheet_name="最近训练材料")
            st.session_state.batch_prediction_excel = batch_excel.getvalue()
            add_operation("ASR预测", f"批量预测 / {model} / 成功{success_count}条 / 失败{failed_count}条",
                          "成功" if failed_count == 0 else "部分成功")
            if failed_count:
                st.warning(f"批量预测完成：成功 {success_count} 条，失败 {failed_count} 条。失败原因已写入结果表。")
            else:
                st.success(f"批量预测完成：共 {success_count} 条。")
        except Exception as exc:
            add_operation("ASR预测", f"批量预测 / {model}", "失败")
            friendly_error("无法完成批量预测", exc, [
                "确认文件包含 Composition 和 electrolyte 两列。",
                "确认数据不超过100条，且电解质名称属于平台支持列表。",
                "检查文件是否为有效的CSV或XLSX格式。",
            ])
    result_frame = st.session_state.get("batch_prediction_result")
    if isinstance(result_frame, pd.DataFrame) and not result_frame.empty:
        st.dataframe(result_frame.round(5), use_container_width=True, hide_index=True)
        render_batch_pca_analysis()
        csv_bytes = st.session_state.get("batch_prediction_csv")
        excel_bytes = st.session_state.get("batch_prediction_excel")
        if not csv_bytes or not excel_bytes:
            csv_bytes = result_frame.to_csv(index=False).encode("utf-8-sig")
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                result_frame.to_excel(writer, index=False, sheet_name="批量预测结果")
                nearest_frame = st.session_state.get("batch_nearest_materials")
                if isinstance(nearest_frame, pd.DataFrame) and not nearest_frame.empty:
                    nearest_frame.to_excel(writer, index=False, sheet_name="最近训练材料")
            excel_bytes = excel_buffer.getvalue()
            st.session_state.batch_prediction_csv = csv_bytes
            st.session_state.batch_prediction_excel = excel_bytes
        render_batch_downloads(csv_bytes, excel_bytes)
        st.caption("免责声明：批量预测结果仅供科研筛选参考，失败行不会生成预测值，所有结果均应结合实验验证。")


def render_batch_pca_analysis():
    payload = st.session_state.get("batch_pca_payload", {})
    targets = pd.DataFrame(payload.get("targets", []))
    training_scores = payload.get("training_scores", [])
    if targets.empty or not training_scores:
        return
    st.markdown("### 批量PCA适用域分析")
    training = pd.DataFrame(training_scores, columns=["PC1", "PC2"])
    if len(training) > 1500:
        training = training.sample(1500, random_state=42)
    training["类型"] = "训练数据"
    training["Composition"] = ""
    training["electrolyte"] = ""
    training["PCA可靠性得分（%）"] = np.nan
    plotted_targets = targets.copy()
    plotted_targets["类型"] = np.where(plotted_targets["适用域"] == "inside", "输入材料：域内", "输入材料：域外")
    plot_frame = pd.concat([
        training[["PC1", "PC2", "类型", "Composition", "electrolyte", "PCA可靠性得分（%）"]],
        plotted_targets[["PC1", "PC2", "类型", "Composition", "electrolyte", "PCA可靠性得分（%）"]],
    ], ignore_index=True)
    figure = px.scatter(
        plot_frame, x="PC1", y="PC2", color="类型", symbol="类型", hover_name="Composition",
        hover_data={"electrolyte": True, "PCA可靠性得分（%）": ":.5f"},
        color_discrete_map={"训练数据": "#94a3b8", "输入材料：域内": "#ff8c33",
                            "输入材料：域外": "#c51b7d"},
        title="批量输入材料在PCA空间中的位置",
    )
    figure.update_traces(selector={"name": "训练数据"}, marker={"size": 6, "opacity": .42})
    figure.update_traces(
        selector={"name": "训练数据"},
        hovertemplate="类型=训练数据<br>PC1=%{x:.5f}<br>PC2=%{y:.5f}<extra></extra>",
    )
    for name in ["输入材料：域内", "输入材料：域外"]:
        target_subset = plotted_targets[plotted_targets["类型"] == name]
        figure.update_traces(selector={"name": name}, marker={"size": 12, "opacity": .95,
                                                              "symbol": "triangle-up" if name.endswith("域内") else "x",
                                                              "line": {"width": 1.5, "color": "white"},
                                                              "sizeref": 1})
        figure.update_traces(
            selector={"name": name},
            hovertemplate=("类型=%{fullData.name}<br>化学式=%{customdata[0]}<br>"
                           "电解质=%{customdata[1]}<br>PC1=%{x:.5f}<br>PC2=%{y:.5f}<br>"
                           "PCA可靠性得分（%%）=%{customdata[2]:.5f}<extra></extra>"),
            customdata=target_subset[["Composition", "electrolyte", "PCA可靠性得分（%）"]].values,
        )
    variance = payload.get("explained_variance_ratio", [])
    if len(variance) >= 2:
        figure.update_xaxes(title=f"PC1（解释方差 {variance[0] * 100:.2f}%）")
        figure.update_yaxes(title=f"PC2（解释方差 {variance[1] * 100:.2f}%）")
    add_pca_applicability_domain(figure, targets[["PC1", "PC2"]].values.tolist())
    figure.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="样本类型")
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    st.caption("绿色区域为基于第5近邻距离和校准集绘制的85% PCA适用域；橙色表示域内，紫色表示域外。")

    nearest = st.session_state.get("batch_nearest_materials")
    if isinstance(nearest, pd.DataFrame) and not nearest.empty:
        st.markdown("#### 最近的3个训练材料")
        labels = nearest[["输入序号", "输入材料", "输入电解质"]].drop_duplicates().copy()
        label_map = {int(row.输入序号): f'{int(row.输入序号)}. {row.输入材料} | {row.输入电解质}'
                     for row in labels.itertuples()}
        selected_index = st.selectbox("选择输入材料", list(label_map),
                                      format_func=lambda value: label_map[value], key="batch_nearest_selector")
        shown = nearest[nearest["输入序号"] == selected_index].drop(
            columns=["输入序号", "输入材料", "输入电解质"])
        st.dataframe(shown.round(5), use_container_width=True, hide_index=True)
        st.caption("PCA标准化距离越小，表示该训练材料在PCA空间中与输入材料越接近。Excel下载文件包含全部输入材料的最近邻结果。")


@st.fragment
def render_batch_downloads(csv_bytes, excel_bytes):
    st.markdown("### 下载批量预测结果")
    c1, c2 = st.columns(2)
    c1.download_button("下载批量结果 CSV", data=csv_bytes, file_name="batch_asr_predictions.csv",
                       mime="text/csv", key="download_batch_csv", on_click="ignore",
                       type="primary", use_container_width=True)
    c2.download_button("下载批量结果 Excel", data=excel_bytes, file_name="batch_asr_predictions.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="download_batch_excel", on_click="ignore", type="primary", use_container_width=True)
    csv_b64 = base64.b64encode(csv_bytes).decode("ascii")
    excel_b64 = base64.b64encode(excel_bytes).decode("ascii")
    st.markdown(
        f'<div style="text-align:center;margin-top:.45rem;color:#64748b;font-size:.86rem">'
        f'若上方按钮被浏览器拦截，可使用备用链接：'
        f'<a download="batch_asr_predictions.csv" href="data:text/csv;base64,{csv_b64}">CSV</a>'
        f' &nbsp;|&nbsp; '
        f'<a download="batch_asr_predictions.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}">Excel</a>'
        f'</div>', unsafe_allow_html=True,
    )


MODEL_OPTIONS = {
    "RandomForestRegressor（随机森林回归）": "RF",
    "SVR（支持向量回归）": "SVR",
    "MLPRegressor（人工神经网络回归）": "ANN",
}


def training_page(mode="overview"):
    heading(t("模型训练", "Model Training"), t("配置数据划分、评估策略和模型专属超参数。",
                                                 "Configure data splits, evaluation and model-specific hyperparameters."))
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
                fit_and_report(data[features], data["Log_ASR"].to_numpy(), model,
                               f"builtin_{name.lower()}.joblib", split_config,
                               model_name=name, parameters=params, source="内置数据集")
            except Exception as exc:
                add_operation("模型训练", f"内置数据集 / {name}", "失败")
                friendly_error("训练失败", exc, ["检查数据划分比例之和是否为100%。", "减少模型复杂度或关闭5-fold后重试。", "刷新页面仍失败时，请提交问题反馈。"])
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
                fit_and_report(X, np.log10(rows["ASR"].to_numpy()), model,
                               f"custom_{name.lower()}.joblib", split_config,
                               model_name=name, parameters=params, source="用户数据集")
            except Exception as exc:
                add_operation("模型训练", f"用户数据集 / {name}", "失败")
                friendly_error("训练失败", exc, ["确认文件包含 Composition、ASR 以及至少一列特征。", "确认ASR均为正数，且数据量满足划分要求。", "检查文本与数值特征列的数据格式是否一致。"])
    render_training_history()


def inspect_uploaded_data(raw):
    issues = []
    duplicate_columns = raw.columns[raw.columns.duplicated()].tolist()
    if duplicate_columns:
        issues.append(("错误", "重复列名", f"发现重复列：{duplicate_columns}"))
    required = {"Composition", "electrolyte", "ASR"}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        issues.append(("错误", "缺少必要列", f"缺少：{missing_columns}"))
        return issues, None
    work = raw.copy()
    missing_counts = work[["Composition", "electrolyte", "ASR"]].isna().sum()
    for column, count in missing_counts.items():
        if count:
            issues.append(("错误", "缺失值", f"{column} 列存在 {int(count)} 个缺失值"))
    numeric_asr = pd.to_numeric(work["ASR"], errors="coerce")
    invalid_numeric = int((numeric_asr.isna() & work["ASR"].notna()).sum())
    if invalid_numeric:
        issues.append(("错误", "数值格式", f"ASR 列存在 {invalid_numeric} 个无法转换为数值的单元格"))
    nonpositive = int((numeric_asr <= 0).fillna(False).sum())
    if nonpositive:
        issues.append(("错误", "ASR范围", f"发现 {nonpositive} 个非正ASR值"))
    invalid_electrolytes = sorted(set(work["electrolyte"].dropna().astype(str).str.strip()) - set(ELECTROLYTES))
    if invalid_electrolytes:
        issues.append(("错误", "非法电解质", f"不支持：{invalid_electrolytes}"))
    duplicate_mask = work.duplicated(subset=["Composition", "electrolyte", "ASR"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        issues.append(("警告", "重复样本", f"发现 {duplicate_count} 行重复记录；整理结果将保留第一条"))
    valid_values = numeric_asr[(numeric_asr > 0) & numeric_asr.notna()]
    if len(valid_values) >= 4:
        logs = np.log10(valid_values)
        q1, q3 = logs.quantile([.25, .75]); iqr = q3 - q1
        outlier_mask = (np.log10(numeric_asr.where(numeric_asr > 0)) < q1 - 1.5 * iqr) | (np.log10(numeric_asr.where(numeric_asr > 0)) > q3 + 1.5 * iqr)
        outlier_count = int(outlier_mask.fillna(False).sum())
        if outlier_count:
            issues.append(("警告", "异常ASR", f"按Log_ASR的1.5×IQR规则识别到 {outlier_count} 个潜在异常值，请人工复核"))
    if not issues:
        issues.append(("通过", "完整校验", "未发现重复、缺失、异常格式或非法类别"))
    has_error = any(level == "错误" for level, _, _ in issues)
    if has_error:
        return issues, None
    work["ASR"] = numeric_asr
    work = work.drop_duplicates(subset=["Composition", "electrolyte", "ASR"], keep="first")
    return issues, normalize_rows(work)


def upload_page():
    heading(t("数据上传", "Data Upload"), t("校验并整理材料实验数据。", "Validate and organize experimental material data."))
    st.markdown(f'<div class="note">必需列：Composition、electrolyte、ASR（Ω·cm²）。电解质仅支持 {", ".join(ELECTROLYTES)}。数据只保存在当前会话，请及时下载备份。</div>', unsafe_allow_html=True)
    upload = st.file_uploader("上传数据（CSV/XLSX）", type=["csv", "xlsx"], key="data")
    if upload:
        try:
            issues, rows = inspect_uploaded_data(read_table(upload))
            report = pd.DataFrame(issues, columns=["级别", "检查项目", "结果"])
            st.markdown("### 数据质量校验报告")
            st.dataframe(report, use_container_width=True, hide_index=True)
            if rows is None:
                signature = (upload.name, getattr(upload, "size", None), "failed")
                if st.session_state.get("last_upload_signature") != signature:
                    add_operation("数据上传", f"{upload.name} / 数据校验未通过", "失败")
                    st.session_state.last_upload_signature = signature
                st.error("数据存在必须修正的错误，暂不生成整理结果。")
                return
            view = rows[["Composition", "electrolyte", "Log_ASR"]].copy()
            view["ASR"] = np.power(10., view["Log_ASR"]); st.session_state["uploaded_data"] = view
            signature = (upload.name, getattr(upload, "size", None))
            if st.session_state.get("last_upload_signature") != signature:
                add_operation("数据上传", f"{upload.name} / 校验通过{len(view)}条")
                st.session_state.last_upload_signature = signature
            st.success(f"通过校验：{len(view)} 条数据"); st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("下载校验后的CSV", view.to_csv(index=False).encode("utf-8-sig"), "validated_asr_data.csv", "text/csv")
        except Exception as exc:
            add_operation("数据上传", getattr(upload, "name", "未知文件"), "失败")
            friendly_error("数据校验失败", exc, ["确认文件是有效的CSV或XLSX格式。", "检查必要列名是否为 Composition、electrolyte、ASR。", "确认ASR为正数，电解质名称属于支持列表。"])


def query_page():
    heading(t("数据查询", "Data Query"), t("浏览内置训练数据，并按化学式、电解质和650℃下ASR范围筛选。",
                                             "Browse and filter the built-in training data."))
    data = get_training_data(); a, b = st.columns([1.5, 1])
    keyword = a.text_input("搜索化学式", placeholder="输入全部或部分化学式")
    electrolyte = b.multiselect("电解质类型", list(ELECTROLYTES), default=list(ELECTROLYTES))
    values = data["ASR（Ω·cm²）"].replace([np.inf, -np.inf], np.nan).dropna(); low, high = float(values.min()), float(values.max())
    span = st.slider("650℃下ASR范围（Ω·cm²）", low, high, (low, high))
    mask = data["电解质"].isin(electrolyte) & data["ASR（Ω·cm²）"].between(*span)
    if keyword: mask &= data["Composition"].astype(str).str.contains(keyword, case=False, regex=False)
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    axis_labels = {
        "tolerance_factor": "Goldschmidt 容忍因子",
        "B_site_oxidation": "B位平均氧化态",
        "oxygen_per_B": "O/B 化学计量比",
        "composition_entropy": "组成熵",
        "ASR（Ω·cm²）": "ASR（Ω·cm²）",
        "Log_ASR": "Log_ASR",
    }
    label_to_column = {axis_labels.get(col, col): col for col in numeric_columns}
    axis_names = list(label_to_column)
    default_x = axis_names.index("B_site_Lewis_acid_strength") if "B_site_Lewis_acid_strength" in axis_names else 0
    default_y = axis_names.index("Log_ASR") if "Log_ASR" in axis_names else min(1, len(axis_names) - 1)
    x_box, y_box = st.columns(2)
    x_label = x_box.selectbox("交互图横轴", axis_names, index=default_x)
    y_label = y_box.selectbox("交互图纵轴", axis_names, index=default_y)
    x_col, y_col = label_to_column[x_label], label_to_column[y_label]
    result = data.loc[mask].copy()
    st.caption(f"共找到 {len(result)} 条记录")
    if x_col == y_col:
        st.info("当前横纵轴选择了同一变量，可选择另一变量以观察相关关系。")
    if not result.empty:
        hover_columns = [c for c in ["Log_ASR", "ASR（Ω·cm²）", x_col, y_col] if c in result]
        hover_data = {c: ":.5f" for c in dict.fromkeys(hover_columns)}
        fig = px.scatter(result, x=x_col, y=y_col, color="电解质", hover_name="Composition",
                         hover_data=hover_data, labels={x_col: x_label, y_col: y_label},
                         color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(marker={"size": 9, "opacity": .78, "line": {"width": .5, "color": "white"}})
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10), hovermode="closest", legend_title_text="电解质")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    default_columns = ["Composition", "电解质", "ASR（Ω·cm²）", "Log_ASR"]
    shown_columns = st.multiselect("表格显示列", list(data.columns), default=default_columns,
                                   help="增删选项即可决定下方表格显示哪些字段")
    if shown_columns:
        st.dataframe(result[shown_columns].round(5), use_container_width=True, hide_index=True, height=440)
    else:
        st.info("请至少选择一个表格显示列。")


def reset_display_settings():
    defaults = {"ui_theme_saved":"亮色", "_ui_theme_control":"亮色", "ui_font_size":"标准", "ui_density":"舒适",
                "default_prediction_model":"RF", "default_electrolyte":"GDC"}
    st.session_state.update(defaults)
    get_cookie_manager().set("asr_ui_theme", "亮色", expires_at=datetime.now() + timedelta(days=30),
                             key="reset_ui_theme")


def save_theme_setting():
    st.session_state.ui_theme_saved = st.session_state._ui_theme_control
    get_cookie_manager().set("asr_ui_theme", st.session_state.ui_theme_saved,
                             expires_at=datetime.now() + timedelta(days=30), key="save_ui_theme")


def settings_page():
    heading(t("设置", "Settings"), t("管理账号资料、登录安全、界面显示和问题反馈。",
                                      "Manage your account, security, appearance and feedback."))
    account_tab, display_tab, history_tab, service_tab, feedback_tab, help_tab = st.tabs([
        t("账号管理", "Account"), t("显示设置", "Appearance"),
        "操作记录", "服务状态", t("问题反馈", "Feedback"), t("帮助与关于", "Help & About")])
    auth, url, key = get_auth_config()
    token = st.session_state.get("auth_access_token", "")
    with account_tab:
        if not auth.get("required", False) or not token:
            st.info("当前未启用账号认证，账号资料与密码管理暂不可用。")
        else:
            st.markdown("#### 账号信息")
            st.text_input("登录邮箱", value=st.session_state.get("auth_email", ""), disabled=True)
            username = st.text_input("用户名", value=st.session_state.get("auth_username", ""),
                                     placeholder="设置用于平台内显示的用户名")
            if st.button("保存用户名", type="primary"):
                headers = {"apikey": key, "Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                try:
                    response = service_request("PUT", f"{url}/auth/v1/user", headers=headers,
                                               json={"data": {"username": username.strip()}})
                    if response.ok:
                        st.session_state.auth_username = username.strip()
                        st.success("用户名已更新。")
                    else:
                        st.error("用户名更新失败，请重新登录后再试。")
                except requests.RequestException:
                    st.error("暂时无法连接账号认证服务。")
            st.divider()
            st.markdown("#### 更改密码")
            current_password = st.text_input("当前密码", type="password", key="settings_current_password")
            new_password = st.text_input("新密码", type="password", key="settings_new_password",
                                         help="至少8位，并包含大写字母、小写字母和数字。")
            confirm_password = st.text_input("确认新密码", type="password", key="settings_confirm_password")
            if st.button("更新密码"):
                if not current_password:
                    st.error("请输入当前密码。")
                elif password_issues(new_password):
                    st.error("密码强度不足：" + "、".join(password_issues(new_password)) + "。")
                elif new_password != confirm_password:
                    st.error("两次输入的新密码不一致。")
                elif current_password == new_password:
                    st.error("新密码不能与当前密码相同。")
                else:
                    try:
                        verify_response = service_request("POST",
                            f"{url}/auth/v1/token?grant_type=password",
                            headers={"apikey": key, "Authorization": f"Bearer {key}",
                                     "Content-Type": "application/json"},
                            json={"email": st.session_state.get("auth_email", ""),
                                  "password": current_password})
                        if not verify_response.ok:
                            st.error("当前密码不正确，无法更新密码。")
                            st.stop()
                        verified_payload = verify_response.json()
                        verified_token = verified_payload.get("access_token", "")
                        headers = {"apikey": key, "Authorization": f"Bearer {verified_token}",
                                   "Content-Type": "application/json"}
                        response = service_request("PUT", f"{url}/auth/v1/user", headers=headers,
                                                   json={"password": new_password})
                        if response.ok:
                            remember_login(verified_payload)
                            st.success("密码已更新。下次登录请使用新密码。")
                        else:
                            st.error("密码更新失败，请重新登录后再试。")
                    except requests.RequestException:
                        st.error("暂时无法连接账号认证服务。")
            st.divider()
            if st.button("退出当前账号", use_container_width=True):
                headers = {"apikey": key, "Authorization": f"Bearer {token}"}
                try:
                    service_request("POST", f"{url}/auth/v1/logout", headers=headers, attempts=2)
                except requests.RequestException:
                    pass
                clear_login()
                st.rerun()
            with st.expander("危险操作：注销账号"):
                st.warning("注销后账号将被永久删除，且无法恢复。与该账号关联的云端记录也可能被删除。")
                delete_password = st.text_input("输入当前密码", type="password", key="delete_account_password")
                delete_confirmation = st.text_input("输入“永久注销”以确认", key="delete_account_confirmation")
                if st.button("永久注销账号", disabled=delete_confirmation != "永久注销" or not delete_password,
                             use_container_width=True):
                    try:
                        verify_response = service_request("POST",
                            f"{url}/auth/v1/token?grant_type=password",
                            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json={"email": st.session_state.get("auth_email", ""), "password": delete_password})
                        if not verify_response.ok:
                            st.error("当前密码不正确，账号未注销。")
                            st.stop()
                        headers = {"apikey": key, "Authorization": f'Bearer {verify_response.json().get("access_token", token)}',
                                   "Content-Type": "application/json"}
                        response = service_request("POST", f"{url}/rest/v1/rpc/delete_own_account", headers=headers,
                                                   json={})
                        if response.ok:
                            clear_login()
                            st.session_state.account_deleted = True
                            st.rerun()
                        else:
                            st.error("账号注销接口尚未配置，请联系管理员创建 delete_own_account 数据库函数。")
                    except requests.RequestException:
                        st.error("暂时无法连接账号服务，请稍后重试。")
    with display_tab:
        st.markdown("#### 外观与使用偏好")
        theme_options = ["亮色", "暗色"]
        if "_ui_theme_control" not in st.session_state:
            st.session_state._ui_theme_control = st.session_state.get("ui_theme_saved", "亮色")
        st.radio("显示模式", theme_options, horizontal=True, key="_ui_theme_control",
                 on_change=save_theme_setting,
                 help="暗色模式会同时调整页面、输入框、卡片、表格和图表配色。")
        a, b = st.columns(2)
        a.radio("字体大小", ["标准", "大"], horizontal=True, key="ui_font_size")
        b.radio("内容密度", ["舒适", "紧凑"], horizontal=True, key="ui_density")
        st.markdown("#### 预测默认值")
        c, d = st.columns(2)
        c.selectbox("默认模型", [name.upper() for name in MODEL_NAMES], index=1, key="default_prediction_model")
        d.selectbox("默认电解质", list(ELECTROLYTES), index=1, key="default_electrolyte")
        st.caption("显示与默认值保存在当前浏览器会话中，不会影响其他用户。")
        st.button("恢复默认显示设置", on_click=reset_display_settings)
    with history_tab:
        st.markdown("#### 最近操作记录")
        st.caption("记录当前会话中的预测、训练和数据上传操作，最多保留100条；退出或服务重启后不会保留。")
        history = st.session_state.get("operation_history", [])
        if history:
            history_frame = pd.DataFrame(history)
            categories = st.multiselect("筛选类型", ["ASR预测", "模型训练", "数据上传"],
                                        default=["ASR预测", "模型训练", "数据上传"])
            shown_history = history_frame[history_frame["类型"].isin(categories)]
            st.dataframe(shown_history, use_container_width=True, hide_index=True)
            st.download_button("下载操作记录 CSV", shown_history.to_csv(index=False).encode("utf-8-sig"),
                               "asr_operation_history.csv", "text/csv")
            confirm_clear = st.checkbox("我确认清空当前会话的操作记录", key="confirm_clear_history")
            if st.button("清空操作记录", disabled=not confirm_clear):
                st.session_state.operation_history = []
                st.rerun()
        else:
            st.info("当前会话尚无预测、训练或数据上传记录。")
    with service_tab:
        st.markdown("#### 服务状态")
        checks = [
            ("训练数据", TRAINING_FILE.exists(), "内置训练数据文件可用" if TRAINING_FILE.exists() else "训练数据文件缺失"),
            ("账号配置", bool(url and key), "Supabase配置已加载" if url and key else "Supabase配置不完整"),
        ]
        try:
            predictor_ready = bool(get_predictor().models)
            predictor_message = "预测模型已加载" if predictor_ready else "未找到预测模型"
        except Exception as exc:
            predictor_ready = False
            predictor_message = f"模型加载失败：{exc}"
        checks.insert(0, ("预测服务", predictor_ready, predictor_message))
        status_frame = pd.DataFrame([{"服务": name, "状态": "正常" if ok else "异常", "说明": message}
                                     for name, ok, message in checks])
        st.dataframe(status_frame, use_container_width=True, hide_index=True)
        if st.button("检查云端账号服务", use_container_width=True):
            try:
                response = service_request("GET", f"{url}/auth/v1/health", headers={"apikey": key}, attempts=2)
                if response.ok:
                    st.success("云端账号服务连接正常。")
                else:
                    st.error(f"云端账号服务返回异常状态（HTTP {response.status_code}）。请稍后重试或联系管理员。")
            except requests.RequestException:
                st.error("无法连接云端账号服务。请检查网络后重试；预测模型本身可能仍可正常使用。")
    with feedback_tab:
        st.markdown("#### 提交问题或建议")
        category = st.selectbox("反馈类型", ["功能建议", "预测问题", "数据问题", "账号问题", "界面问题", "其他"])
        message = st.text_area("反馈内容", height=160, placeholder="请描述遇到的问题、复现步骤或希望增加的功能。")
        if st.button("提交反馈", type="primary", disabled=not message.strip(), use_container_width=True):
            if not token:
                st.error("请登录后再提交反馈。")
            else:
                headers = {"apikey": key, "Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "return=minimal"}
                payload = {"email": st.session_state.get("auth_email", ""),
                           "username": st.session_state.get("auth_username", ""),
                           "category": category, "message": message.strip(), "app_version": APP_VERSION}
                try:
                    response = service_request("POST", f"{url}/rest/v1/feedback", headers=headers, json=payload)
                    if response.ok:
                        st.success("反馈已提交，感谢你的建议。")
                    else:
                        st.error("反馈入口尚未完成数据库配置，请联系管理员创建 feedback 表及对应访问策略。")
                except requests.RequestException:
                    st.error("暂时无法连接反馈服务，请稍后重试。")
    with help_tab:
        st.markdown("#### 使用说明")
        with st.expander("ASR预测与可靠性", expanded=True):
            st.markdown("输入材料化学式、模型和电解质后运行预测。Log_ASR 与 ASR 为650℃下的模型输出；PCA可靠性衡量输入特征与训练数据分布的相似程度，并不是预测准确率。")
        with st.expander("模型训练与训练记录"):
            st.markdown("内置数据训练使用平台特征；自定义训练默认将 Composition、ASR 之外的列作为特征。系统保留当前会话最近5次训练，并支持下载表现最佳的模型。")
        with st.expander("数据上传与查询"):
            st.markdown("上传模块会检查缺失值、重复样本、非法电解质、ASR格式与异常值。查询模块支持筛选数据、自定义坐标轴和显示列。")
        st.markdown("#### 隐私政策")
        st.markdown("- 账号密码由 Supabase Auth 管理，本项目代码和 GitHub 仓库不保存用户密码。\n- 上传的数据默认仅在当前应用会话中处理，除非页面明确提示将数据写入云端。\n- 反馈内容会连同账号邮箱、用户名和应用版本写入受访问策略保护的反馈表。\n- 请勿上传含有个人敏感信息、商业机密或无权处理的数据。")
        st.markdown("#### 服务条款与免责声明")
        st.markdown("- 本平台用于科研辅助、材料筛选和方法探索，不构成产品性能承诺、工程设计依据或商业决策建议。\n- 预测值、可靠性得分和解释结果均来源于有限训练数据与统计模型，可能存在偏差、外推误差或数据质量影响。\n- 任何关键结论均应通过独立实验和专业判断验证；因直接使用平台输出造成的损失，平台不承担相应责任。\n- 用户应确保上传数据来源合法，并拥有必要的使用和处理权限。\n- 禁止利用平台实施违法活动、攻击服务或绕过访问控制。")


def set_active_page(page_name, module_name, keep_training_menu=False):
    """Update both navigation levels before Streamlit redraws the sidebar."""
    st.session_state.active_page = page_name
    st.session_state.active_module = module_name
    if not keep_training_menu:
        st.session_state.training_menu_open = False
    save_browser_ui_state()


def toggle_training_menu():
    st.session_state.active_module = "模型训练"
    st.session_state.training_menu_open = not st.session_state.training_menu_open
    save_browser_ui_state()


with st.sidebar:
    st.image(str(LOGO_FILE), width=220)
    st.caption("SOFC 阴极材料智能分析平台")
    if "active_page" not in st.session_state:
        st.session_state.active_page = "ASR预测"
    if "active_module" not in st.session_state:
        st.session_state.active_module = (
            "模型训练" if st.session_state.active_page in {"内置数据集训练", "自定义数据集训练"}
            else st.session_state.active_page
        )
    if "training_menu_open" not in st.session_state:
        st.session_state.training_menu_open = False

    st.button(t("◇  ASR预测", "◇  ASR Prediction"), type="primary" if st.session_state.active_module == "ASR预测" else "secondary",
              on_click=set_active_page, args=("ASR预测", "ASR预测"), use_container_width=True)
    st.button(t("▦  模型训练", "▦  Model Training"), type="primary" if st.session_state.active_module == "模型训练" else "secondary",
              on_click=toggle_training_menu, use_container_width=True)
    if st.session_state.training_menu_open:
        _, child_area = st.columns([.11, .89])
        with child_area:
            st.button(t("01  内置数据集训练", "01  Built-in Dataset"),
                      type="primary" if st.session_state.active_page == "内置数据集训练" else "secondary",
                      on_click=set_active_page, args=("内置数据集训练", "模型训练", True))
            st.button(t("02  自定义数据集训练", "02  Custom Dataset"),
                      type="primary" if st.session_state.active_page == "自定义数据集训练" else "secondary",
                      on_click=set_active_page, args=("自定义数据集训练", "模型训练", True))
    st.button(t("⇧  数据上传", "⇧  Data Upload"), type="primary" if st.session_state.active_module == "数据上传" else "secondary",
              on_click=set_active_page, args=("数据上传", "数据上传"), use_container_width=True)
    st.button(t("⌕  数据查询", "⌕  Data Query"), type="primary" if st.session_state.active_module == "数据查询" else "secondary",
              on_click=set_active_page, args=("数据查询", "数据查询"), use_container_width=True)
    with st.container(key="sidebar_utility"):
        if st.session_state.get("auth_access_token"):
            st.caption(f'当前账号：{st.session_state.get("auth_email", "已登录用户")}')
        st.button(t("⚙  设置", "⚙  Settings"), type="secondary",
                  on_click=set_active_page, args=("设置", "设置"), use_container_width=True)
    page = st.session_state.active_page
    st.markdown(f'<div class="version">当前版本：{APP_VERSION}</div>', unsafe_allow_html=True)

if page == "内置数据集训练":
    training_page("builtin")
elif page == "自定义数据集训练":
    training_page("custom")
else:
    {"ASR预测": prediction_page, "数据上传": upload_page, "数据查询": query_page, "设置": settings_page}[page]()
