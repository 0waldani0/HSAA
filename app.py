#!/usr/bin/env python3
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, "build")

SUITES = [
    ("test_unified_engine", os.path.join(BUILD_DIR, "test_unified_engine"), []),
    ("test_c_api", os.path.join(BUILD_DIR, "test_c_api"), []),
    ("test_vsa_db", os.path.join(BUILD_DIR, "test_vsa_db"), []),
    ("test_llm_text_code_gen", os.path.join(BUILD_DIR, "test_llm_text_code_gen"), []),
    ("test_hssa", os.path.join(BUILD_DIR, "test_hssa"), []),
    ("test_model_mod", os.path.join(BUILD_DIR, "test_model_mod"), []),
    ("e2e_test_runner", os.path.join(BUILD_DIR, "e2e_test_runner"), []),
]

_cached_payload = None


def ensure_compiled():
    os.makedirs(BUILD_DIR, exist_ok=True)
    if all(os.path.isfile(bp) and os.access(bp, os.X_OK) for _, bp, _ in SUITES):
        return
    cmd = r"""
    if ! command -v g++ >/dev/null 2>&1; then
        apt-get update && apt-get install -y --no-install-recommends g++ gcc libc6-dev make
    fi
    CXXFLAGS="-std=c++20 -O2 -march=native -pthread -DBUILDING_POPCNT_BRAIN_LIB -Iinclude -Isecond_brain -Ilegacy_ptq"
    gcc -O2 -march=native -Iinclude -Ilegacy_ptq -c legacy_ptq/ptq_cuda.c -o build/ptq_cuda.o &
    g++ $CXXFLAGS -c src/vsa_db.cpp -o build/vsa_db.o &
    g++ $CXXFLAGS -c src/hssa.cpp -o build/hssa.o &
    g++ $CXXFLAGS -c src/model_mod.cpp -o build/model_mod.o &
    g++ $CXXFLAGS -c src/bridge.cpp -o build/bridge.o &
    g++ $CXXFLAGS -c src/bench_runner.cpp -o build/bench_runner.o &
    g++ $CXXFLAGS -c src/unified_engine.cpp -o build/unified_engine.o &
    g++ $CXXFLAGS -c second_brain/popcnt_brain.cpp -o build/popcnt_brain.o &
    wait
    CORE_OBJS="build/vsa_db.o build/hssa.o build/model_mod.o build/bridge.o build/bench_runner.o build/unified_engine.o build/popcnt_brain.o build/ptq_cuda.o"
    g++ $CXXFLAGS tests/test_unified_engine.cpp $CORE_OBJS -o build/test_unified_engine &
    g++ $CXXFLAGS second_brain/test_c_api.cpp build/popcnt_brain.o -o build/test_c_api &
    g++ $CXXFLAGS tests/test_vsa_db.cpp $CORE_OBJS -o build/test_vsa_db &
    g++ $CXXFLAGS tests/test_llm_text_code_gen.cpp build/popcnt_brain.o -o build/test_llm_text_code_gen &
    g++ $CXXFLAGS tests/test_hssa.cpp $CORE_OBJS -o build/test_hssa &
    g++ $CXXFLAGS tests/test_model_mod.cpp $CORE_OBJS -o build/test_model_mod &
    g++ $CXXFLAGS tests/e2e_test_runner.cpp $CORE_OBJS -o build/e2e_test_runner &
    g++ $CXXFLAGS src/main_lab.cpp $CORE_OBJS -o build/upgrade_lab &
    wait
    """
    subprocess.run(["bash", "-c", cmd], cwd=ROOT_DIR, check=False)


def run_all_suites():
    global _cached_payload
    ensure_compiled()
    results = []
    for name, bin_path, args in SUITES:
        exists = os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)
        if not exists:
            results.append({
                "binary": name,
                "exists": False,
                "exitCode": -1,
                "durationMs": 0,
                "output": f"Binary not found: {bin_path}",
            })
            continue
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [bin_path] + args,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=60,
            )
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            out = (proc.stdout or "") + (proc.stderr or "")
            results.append({
                "binary": name,
                "exists": True,
                "exitCode": proc.returncode,
                "durationMs": dt_ms,
                "output": out.strip(),
            })
        except Exception as exc:
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            results.append({
                "binary": name,
                "exists": True,
                "exitCode": -1,
                "durationMs": dt_ms,
                "output": str(exc),
            })
    _cached_payload = {"suites": results, "timestamp": time.time()}
    return _cached_payload


def run_live_synthesis(mode="all", prompt=""):
    ensure_compiled()
    bin_path = os.path.join(BUILD_DIR, "upgrade_lab")
    args = [bin_path, "unified", "--json", mode]
    if prompt:
        args.append(prompt)
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = (proc.stdout or "").strip()
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            qs = parse_qs(parsed.query)
            force_refresh = "refresh" in qs or _cached_payload is None
            data = run_all_suites() if force_refresh else _cached_payload
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/synthesize":
            qs = parse_qs(parsed.query)
            mode = qs.get("mode", ["all"])[0]
            prompt = qs.get("prompt", [""])[0]
            data = run_live_synthesis(mode=mode, prompt=prompt)
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path in ("/", "/index.html"):
            index_path = os.path.join(ROOT_DIR, "index.html")
            with open(index_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", 3000), DashboardHandler)
    print("HSSA & SecondBrain Unified Engine dashboard listening on http://0.0.0.0:3000", flush=True)
    server.serve_forever()
