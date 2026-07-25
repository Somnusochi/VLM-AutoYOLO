#!/usr/bin/env python3
"""VLM-AutoYOLO CLI — one command to setup and launch."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = BACKEND / ".venv"
IS_WIN = sys.platform == "win32"
BIN = "Scripts" if IS_WIN else "bin"
PYTHON_EXE = "python.exe" if IS_WIN else "python"
PYTHON = VENV / BIN / PYTHON_EXE
UVICORN = VENV / BIN / ("uvicorn.exe" if IS_WIN else "uvicorn")
ALEMBIC = VENV / BIN / ("alembic.exe" if IS_WIN else "alembic")
PIP = VENV / BIN / ("pip.exe" if IS_WIN else "pip")
PIDFILE = ROOT / ".cli_pids.json"

# 8000 and 8500 are reserved by the local Windows networking stack.
# Keep the environment override for users who need a different port.
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 9000))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", 5173))

# ── Utilities ──────────────────────────────────────────

def green(s):   return f"\033[92m{s}\033[0m"
def red(s):     return f"\033[91m{s}\033[0m"
def cyan(s):    return f"\033[96m{s}\033[0m"
def yellow(s):  return f"\033[93m{s}\033[0m"

def step(msg):   print(f"  {cyan(msg)}")
def ok(msg):     print(f"  {green('✓')} {msg}")
def warn(msg):   print(f"  {yellow('!')} {msg}")
def fail(msg):   print(f"  {red('✗')} {msg}"); sys.exit(1)

def run(cmd, **kwargs):
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)

def run_output(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs).stdout.strip()

def is_port_open(port):
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except Exception:
        return False


# ── Prereq checks ──────────────────────────────────────

def check_python():
    step("Checking Python 3.12+...")
    v = sys.version_info
    if v.major == 3 and v.minor >= 12:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return
    fail(f"Python 3.12+ required, found {v.major}.{v.minor}")

def check_node():
    step("Checking Node.js 22+...")
    try:
        out = run_output(["node", "--version"])
        major = int(out.lstrip("v").split(".")[0])
        if major >= 22:
            ok(out)
            return
        fail(f"Node.js 22+ required, found {out}")
    except Exception:
        fail("Node.js not found. Install: https://nodejs.org")

def check_pnpm():
    step("Checking pnpm...")
    # Windows 全局安装会生成 pnpm.cmd
    pnpm = shutil.which("pnpm") or (shutil.which("pnpm.cmd") if IS_WIN else None)
    if pnpm:
        ok(pnpm)
        return
    warn("pnpm not found in PATH. Install: npm install -g pnpm   or   brew install pnpm")
    print("  Attempting: npm install -g pnpm...")
    try:
        # .cmd 必须经由 cmd.exe 调用
        npm_install = ["cmd", "/c", "npm", "install", "-g", "pnpm"] if IS_WIN else ["npm", "install", "-g", "pnpm"]
        run(npm_install)
        # 安装后需重新确认当前进程能找到 pnpm
        pnpm = shutil.which("pnpm") or (shutil.which("pnpm.cmd") if IS_WIN else None)
        if pnpm:
            ok(f"installed: {pnpm}")
            return
        fail("pnpm was installed but is not in PATH. Restart the terminal, then run: pnpm -v")
    except subprocess.CalledProcessError:
        fail("Failed to install pnpm. Run manually: npm install -g pnpm")

def check_ffmpeg():
    step("Checking ffmpeg...")
    if shutil.which("ffmpeg"):
        ok("found")
        return
    warn("ffmpeg not found — video features will not work")
    print("    Install: brew install ffmpeg / apt install ffmpeg")

def check_postgres():
    step("Checking PostgreSQL...")
    env_file = BACKEND / ".env"
    if env_file.exists():
        content = env_file.read_text()
        if "DATABASE_URL" in content and "postgresql" in content:
            ok("configured in .env")
            return True
    ok("will use SQLite (zero-config)")
    return False


# ── Quick pre-flight for start ─────────────────────────

def ensure_setup_done():
    """Lightweight check — warn if deps look missing, but don't fail."""
    if not VENV.exists():
        fail("Virtual env not found. Run: python cli.py setup")
    if not (FRONTEND / "node_modules").exists():
        fail("node_modules not found. Run: python cli.py setup")


# ── Setup ──────────────────────────────────────────────

def setup_venv():
    step("Setting up Python virtual environment...")
    if VENV.exists():
        ok("already exists")
        return
    run([sys.executable, "-m", "venv", str(VENV)])
    run([str(PIP), "install", "--upgrade", "pip"])
    ok("created")

def install_python_deps():
    step("Installing Python dependencies (may take a few minutes)...")
    req = BACKEND / "requirements.txt"
    run([str(PIP), "install", "-r", str(req)])
    ok("done")

def install_node_deps():
    step("Installing Node.js dependencies (may take a few minutes)...")
    if (FRONTEND / "node_modules").exists():
        ok("already exists")
        return
    # npm global installs pnpm as pnpm.cmd on Windows; cmd.exe is required to run it.
    pnpm_install = ["cmd", "/c", "pnpm", "install"] if IS_WIN else ["pnpm", "install"]
    run(pnpm_install, cwd=FRONTEND)
    ok("done")

def check_db_config():
    env_file = BACKEND / ".env"
    env_example = BACKEND / ".env.example"
    if not env_file.exists() and env_example.exists():
        step("Creating .env from .env.example...")
        shutil.copy(env_example, env_file)
        ok("done")

def run_migrations():
    step("Running database migrations...")
    result = run_output(
        [str(ALEMBIC), "upgrade", "head"],
        cwd=BACKEND,
        env={**os.environ, "PYTHONPATH": str(BACKEND)},
    )
    if "Running" in result:
        ok("migrations applied")
    else:
        ok("up to date")

def check_sam3_token():
    """Guide SAM3 setup — needs HF_TOKEN for the gated facebook/sam3 model."""
    step("Checking SAM3 model access...")

    # Check if HF_TOKEN is already set
    env_file = BACKEND / ".env"
    if env_file.exists():
        content = env_file.read_text()
        if "HF_TOKEN" in content and "HF_TOKEN=" in content:
            val = content.split("HF_TOKEN=")[-1].split("\n")[0].strip()
            if val and val != "${HF_TOKEN:-}":
                ok("HF_TOKEN configured")
                return True

    print(f"\n  {yellow('SAM3 requires a HuggingFace access token.')}")
    print("  To use SAM3 text-driven detection + segmentation:")
    print()
    print(f"  1. Visit {cyan('https://huggingface.co/facebook/sam3')}")
    print("     Click \"Agree and access repository\"")
    print(f"  2. Create a Read token at {cyan('https://huggingface.co/settings/tokens')}")
    print(f"  3. Set it in {cyan('backend/.env')}:")
    print("       HF_TOKEN=hf_your_token_here")
    print()
    print(f"  {yellow('Skip this step if you only use VLM+SAM2 mode.')}")
    return False

SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"

def download_vlm():
    step("VLM model (LocateAnything-3B, ~6GB)...")
    model_dir = BACKEND / "model"
    required_files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "modeling_locateanything.py",
        "modeling_qwen2.py",
        "modeling_vit.py",
        "preprocessor_config.json",
        "processing_locateanything.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    if (
        model_dir.exists()
        and not any(model_dir.glob("*.segments.json"))
        and all((model_dir / name).is_file() for name in required_files)
    ):
        ok("already cached")
        return
    print(f"  Downloading missing model files to {model_dir}... (~10-30 min)")
    try:
        run(
            [str(PYTHON), "-c",
             "from huggingface_hub import snapshot_download; "
             "snapshot_download('nvidia/LocateAnything-3B', local_dir='model')"],
            cwd=BACKEND,
        )
        ok("downloaded")
    except Exception as e:
        warn(f"Download failed: {e}")
        print("  Will download on first detection instead.")

def download_sam2():
    step("SAM2 model (facebook/sam2.1-hiera-base-plus, ~2.4GB)...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam2.1-hiera-base-plus"
    if cache_dir.exists() and any(cache_dir.rglob("sam2.1_hiera_base_plus.pt")):
        ok("already cached")
        return
    print("  Downloading SAM2... (~5-15 min)")
    try:
        run(
            [str(PYTHON), "-c",
             "from sam2.build_sam import build_sam2_hf; "
             f"build_sam2_hf('{SAM2_MODEL_ID}', device='cpu')"],
            cwd=BACKEND,
        )
        ok("downloaded")
    except Exception as e:
        warn(f"Download failed: {e}")
        print("  SAM2 will download on first use.")

def download_models(selection="all"):
    """Download selected models. selection: 'all' | 'vlm' | 'sam2' | 'vlm,sam2'"""
    selected = set(selection.replace(" ", "").split(","))

    if "all" in selected:
        selected = {"vlm", "sam2"}

    for m in sorted(selected):
        if m == "vlm":
            download_vlm()
        elif m == "sam2":
            download_sam2()
        else:
            warn(f"Unknown model: {m} (valid: vlm, sam2)")

    if not selected:
        print("  No models selected. Use --models vlm|sam2|all")


# ── Start / Stop ───────────────────────────────────────

def _save_pids(backend_pid, frontend_pid):
    PIDFILE.write_text(json.dumps({
        "backend": backend_pid,
        "frontend": frontend_pid,
        "backend_port": BACKEND_PORT,
        "frontend_port": FRONTEND_PORT,
    }))

def _load_pids():
    if not PIDFILE.exists():
        return None
    try:
        return json.loads(PIDFILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None

def start_backend():
    step(f"Starting backend on port {BACKEND_PORT}...")
    if is_port_open(BACKEND_PORT):
        warn(f"Port {BACKEND_PORT} already in use. Try: BACKEND_PORT={BACKEND_PORT + 1} python3 cli.py start")
        return None
    proc = subprocess.Popen(
        [str(UVICORN), "app.main:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        cwd=BACKEND,
        env={**os.environ, "PYTHONPATH": str(BACKEND)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import urllib.request
    for _ in range(30):
        time.sleep(0.5)
        if proc.poll() is not None:
            fail(f"Backend process exited with code {proc.returncode}. Check logs.")
        try:
            urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/api/health", timeout=1)
            ok("backend ready")
            return proc
        except Exception:
            continue
    proc.kill()
    fail("Backend failed to start — check backend logs for errors.")

def start_frontend():
    step(f"Starting frontend on port {FRONTEND_PORT}...")
    if is_port_open(FRONTEND_PORT):
        warn(f"Port {FRONTEND_PORT} already in use. Try: FRONTEND_PORT={FRONTEND_PORT + 1} python3 cli.py start")
        return None
    proc = subprocess.Popen(
        ["pnpm.cmd", "dev", "--port", str(FRONTEND_PORT)],
        cwd=FRONTEND,
        env={**os.environ, "VITE_BACKEND_PORT": str(BACKEND_PORT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import urllib.request
    for _ in range(30):
        time.sleep(0.5)
        if proc.poll() is not None:
            fail(f"Frontend process exited with code {proc.returncode}. Check logs.")
        try:
            urllib.request.urlopen(f"http://localhost:{FRONTEND_PORT}", timeout=1)
            ok("frontend ready")
            return proc
        except Exception:
            continue
    proc.kill()
    fail("Frontend failed to start — check frontend logs for errors.")


def cmd_start():
    print(green("VLM-AutoYOLO\n"))
    ensure_setup_done()

    backend = start_backend()
    frontend = start_frontend()
    if backend and frontend:
        _save_pids(backend.pid, frontend.pid)
    elif backend:
        backend.kill()  # rollback
    elif frontend:
        frontend.kill()

    if not backend or not frontend:
        sys.exit(1)

    print(f"\n{'━' * 46}")
    print(f"  {green('VLM-AutoYOLO')}")
    print(f"  前端:   {cyan(f'http://localhost:{FRONTEND_PORT}')}")
    print(f"  后端:   {cyan(f'http://localhost:{BACKEND_PORT}')}")
    print(f"  API:    {cyan(f'http://localhost:{BACKEND_PORT}/docs')}")
    print(f"  {red('Ctrl+C')} 停止")
    print(f"{'━' * 46}")
    print(f"  💡 {yellow('提示: VLM 和 SAM 模型采用【惰性加载】策略。')}")
    print(f"     {yellow('在网页端首次执行检测前，它们会显示为“未加载”状态。')}")
    print(f"{'━' * 46}\n")

    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cmd_stop()


def _kill_port(port):
    """Kill process listening on a port (fallback when no PID file)."""
    try:
        if IS_WIN:
            run_output(f'netstat -ano | findstr :{port}', shell=True)
        else:
            out = run_output(["lsof", "-ti", f":{port}"])
            for pid_str in out.split("\n"):
                try:
                    os.kill(int(pid_str), signal.SIGTERM)
                except OSError:
                    pass
    except Exception:
        pass

def cmd_stop():
    pids = _load_pids()
    killed = False
    if pids:
        for name in ("backend", "frontend"):
            pid = pids.get(name)
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                except OSError:
                    pass
        PIDFILE.unlink(missing_ok=True)

    # Fallback: kill by port if PID file missing or stale
    if is_port_open(BACKEND_PORT):
        _kill_port(BACKEND_PORT)
        killed = True
    if is_port_open(FRONTEND_PORT):
        _kill_port(FRONTEND_PORT)
        killed = True

    if killed:
        time.sleep(0.5)
        print(green("Services stopped."))
    else:
        print("No running services found.")


def cmd_status():
    backend_up = is_port_open(BACKEND_PORT)
    frontend_up = is_port_open(FRONTEND_PORT)

    print(f"  Backend  ({BACKEND_PORT}): {green('running') if backend_up else red('stopped')}")
    print(f"  Frontend ({FRONTEND_PORT}): {green('running') if frontend_up else red('stopped')}")

    if backend_up and frontend_up:
        print(f"\n  {cyan(f'http://localhost:{FRONTEND_PORT}')}")
        print(f"  {cyan(f'http://localhost:{BACKEND_PORT}/docs')}")

    pids = _load_pids()
    if pids:
        print(f"\n  PIDs: backend={pids.get('backend')}, frontend={pids.get('frontend')}")


# ── Main ────────────────────────────────────────────────

def cmd_setup(skip_models=False, model_selection="all"):
    print(green("VLM-AutoYOLO Setup\n"))
    check_python()
    check_node()
    check_pnpm()
    check_ffmpeg()
    check_postgres()
    setup_venv()
    install_python_deps()
    install_node_deps()
    check_db_config()
    run_migrations()
    if not skip_models:
        download_models(model_selection)
    check_sam3_token()
    print(f"\n{green('Setup complete!')} Run: {cyan('python cli.py start')}")

def print_help():
    print("""VLM-AutoYOLO CLI

Usage: python3 cli.py <command> [options]

Commands:
  setup         Install dependencies, init database, download models
  start          Launch backend + frontend (requires setup first)
  stop           Stop running services
  status         Show whether services are running
  all            Setup + start (one command)
  download       Download/re-download models only

Options:
  --models=vlm|sam2|all   Select which models to download (default: all)
                          vlm  = LocateAnything-3B (~6GB, required)
                          sam2 = SAM2.1 segmentation (~2.4GB, optional)
                          all  = both
                          Use comma for multiple: --models=vlm,sam2
  --no-models              Skip model download entirely
  --help, -h               Show this help

Environment:
  BACKEND_PORT   Backend port (default 8000)
  FRONTEND_PORT  Frontend port (default 5173)
""")

def main():
    args = sys.argv[1:]

    if "-h" in args or "--help" in args:
        print_help()
        return

    skip_models = "--no-models" in args
    model_selection = "all"
    for a in args:
        if a.startswith("--models="):
            model_selection = a.split("=", 1)[1]
    cmds = [a for a in args if not a.startswith("--")]
    cmd = cmds[0] if cmds else "start"

    if cmd == "setup":
        cmd_setup(skip_models=skip_models, model_selection=model_selection)
    elif cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    elif cmd == "all":
        cmd_setup(skip_models=skip_models, model_selection=model_selection)
        print()
        cmd_start()
    elif cmd == "download":
        check_python()
        setup_venv()
        install_python_deps()
        download_models(model_selection)
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
