"""Launch the fully local ASR application in a desktop window."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
APP_FILE = APP_ROOT / "streamlit_app.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(port: int, process: subprocess.Popen) -> bool:
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=0.5) as health:
                if health.status != 200:
                    continue
            with urllib.request.urlopen(f"{base_url}/", timeout=0.5) as page:
                if page.status == 200 and b"<html" in page.read(1024).lower():
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def _serve(port: int) -> None:
    try:
        os.environ["ASR_DESKTOP_MODE"] = "1"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        os.chdir(APP_ROOT)
        import streamlit.config as st_config
        from streamlit.web import bootstrap

        # In a PyInstaller bundle, streamlit/config.py is outside site-packages.
        # Streamlit otherwise mistakes the bundle for a development checkout
        # and does not mount its production frontend, making '/' return 404.
        st_config.set_option("global.developmentMode", False)
        st_config.set_option("server.address", "127.0.0.1")
        st_config.set_option("server.port", port)
        st_config.set_option("server.headless", True)
        st_config.set_option("browser.serverAddress", "127.0.0.1")
        st_config.set_option("browser.serverPort", port)
        bootstrap.run(
            str(APP_FILE), False, [],
            flag_options={
                "server.address": "127.0.0.1", "server.port": port,
                "server.headless": True, "server.fileWatcherType": "none",
                "browser.gatherUsageStats": False,
            },
        )
    except Exception as exc:
        _show_error(f"本地服务启动失败：{exc}")
        raise


def _show_error(message: str) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "ASR预测离线版", 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        _serve(int(sys.argv[2]))
        return

    port = _free_port()
    env = os.environ.copy()
    env["ASR_DESKTOP_MODE"] = "1"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    command = [sys.executable, "--serve", str(port)] if getattr(sys, "frozen", False) else [
        sys.executable, str(Path(__file__).resolve()), "--serve", str(port),
    ]
    process = subprocess.Popen(command, cwd=APP_ROOT, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        if not _wait_for_server(port, process):
            _show_error("本地应用未能启动。请检查程序文件是否完整，然后重新打开。")
            return
        url = f"http://127.0.0.1:{port}/"
        try:
            import webview

            webview.create_window("钙钛矿型SOFC阴极材料650℃下ASR预测", url,
                                  width=1280, height=840, min_size=(960, 650))
            webview.start(gui="edgechromium", debug=False)
        except Exception:
            # Windows without WebView2 can still use the installed default browser.
            webbrowser.open(url)
            if sys.platform == "win32":
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    0, "应用已在默认浏览器中打开。使用完毕后点击“确定”关闭本地服务。",
                    "ASR预测离线版", 0x40,
                )
            else:
                input("应用已在浏览器中打开；按回车键关闭本地服务。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
