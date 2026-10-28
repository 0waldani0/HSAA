import http.server
import socketserver
import urllib.request
import urllib.error
import json
import sys
import re
import time
import os
import threading

PORT = 1234
CHANNEL = "waldani_qwen38_27b_session"
NTFY_URL = f"https://ntfy.sh/{CHANNEL}/raw?poll=1"
ENDPOINT_FILE = r"c:\Users\Администратор\Documents\project upgrade\kaggle_endpoint.txt"

cached_target = None
remote_model_id = None
last_check = 0

def fetch_ntfy_endpoint():
    global cached_target, last_check
    now = time.time()
    try:
        req = urllib.request.Request("https://ntfy.sh/waldani_qwen38_27b_session/json?poll=1&since=all", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as res:
            raw = res.read().decode("utf-8", errors="ignore")
            matches = re.findall(r"OPENCODE_API_URL:\s*(https://[a-zA-Z0-9-]+\.trycloudflare\.com/v1)", raw)
            if matches:
                latest = matches[-1].rstrip("/")
                if latest != cached_target:
                    print(f"[BRIDGE] Updated active Kaggle API endpoint from ntfy: {latest}", flush=True)
                    cached_target = latest
                    try:
                        with open(ENDPOINT_FILE, "w", encoding="utf-8") as f:
                            f.write(latest)
                    except:
                        pass
                last_check = now
                return cached_target
    except Exception as e:
        print(f"[BRIDGE] Error checking ntfy: {e}", flush=True)
    return cached_target

def get_remote_target():
    global cached_target, last_check
    now = time.time()
    
    # 1. Fetch from ntfy every 10 seconds or if no cached target
    if not cached_target or (now - last_check > 10):
        t = fetch_ntfy_endpoint()
        if t:
            return t

    if cached_target:
        return cached_target

    # 2. Fallback to local endpoint file
    if os.path.exists(ENDPOINT_FILE):
        try:
            with open(ENDPOINT_FILE, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t.startswith("http"):
                    cached_target = t.rstrip("/")
                    return cached_target
        except:
            pass

    return None

def get_remote_model_id(target_base):
    global remote_model_id
    if remote_model_id:
        return remote_model_id
    try:
        req = urllib.request.Request(f"{target_base}/models", headers={"User-Agent": "OpenCode-Bridge"})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode("utf-8"))
            if "data" in data and len(data["data"]) > 0:
                remote_model_id = data["data"][0].get("id")
                print(f"[BRIDGE] Discovered remote model ID: {remote_model_id}", flush=True)
                return remote_model_id
    except:
        pass
    return None

class OpenCodeProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write(f"[BRIDGE-REQ] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")
        sys.stdout.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if "models" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            models_resp = {
                "object": "list",
                "data": [
                    {"id": "qwen-27b-heretic", "object": "model", "owned_by": "kaggle"},
                    {"id": "qwen-27b-uncensored", "object": "model", "owned_by": "kaggle"},
                    {"id": "qwen3.8-27b", "object": "model", "owned_by": "kaggle"},
                    {"id": "local-llama/qwen-27b-uncensored", "object": "model", "owned_by": "kaggle"},
                    {"id": "local-llama/qwen-27b-heretic", "object": "model", "owned_by": "kaggle"}
                ]
            }
            self.wfile.write(json.dumps(models_resp).encode("utf-8"))
            return

        target_base = get_remote_target()
        if not target_base:
            self.send_error(503, "Kaggle GPU server is not active or tunnel not found.")
            return

        target_url = target_base + "/models" if "models" in self.path else target_base + self.path

        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "OpenCode-Bridge"})
            with urllib.request.urlopen(req, timeout=30) as res:
                self.send_response(res.status)
                for k, v in res.headers.items():
                    if k.lower() not in ["content-length", "transfer-encoding"]:
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(res.read())
        except Exception as e:
            self.send_error(502, f"Proxy error: {e}")

    def do_POST(self):
        target_base = get_remote_target()
        if not target_base:
            self.send_error(503, "Kaggle GPU server is not active. Please start the Kaggle kernel.")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        # Normalise payload to model on Kaggle
        try:
            body_json = json.loads(raw_body.decode("utf-8"))
            m_id = get_remote_model_id(target_base)
            if m_id:
                body_json["model"] = m_id
            body = json.dumps(body_json).encode("utf-8")
        except:
            body = raw_body

        if "chat/completions" in self.path:
            target_url = f"{target_base}/chat/completions"
        elif "completions" in self.path:
            target_url = f"{target_base}/completions"
        else:
            target_url = f"{target_base}{self.path}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenCode-Bridge",
            "Content-Length": str(len(body))
        }

        try:
            req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as res:
                self.send_response(res.status)
                for k, v in res.headers.items():
                    if k.lower() not in ["content-length"]:
                        self.send_header(k, v)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # Stream response
                while True:
                    chunk = res.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            global cached_target
            cached_target = None
            self.send_error(502, f"Bridge error connecting to Kaggle: {e}")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    print(f"=== OpenCode Kaggle Bridge Daemon ===", flush=True)
    print(f"Listening locally on: http://127.0.0.1:{PORT}/v1", flush=True)
    target = get_remote_target()
    if target:
        print(f"Active target endpoint: {target}", flush=True)
    else:
        print("Waiting for Kaggle GPU tunnel to publish endpoint...", flush=True)

    server = ThreadedHTTPServer(("127.0.0.1", PORT), OpenCodeProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping bridge daemon...", flush=True)
