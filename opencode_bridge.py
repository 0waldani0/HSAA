import http.server
import socketserver
import urllib.request
import urllib.error
import json
import sys
import re
import time
import threading

PORT = 1234
CHANNEL = "waldani_qwen38_27b_session"
NTFY_URL = f"https://ntfy.sh/{CHANNEL}/raw?poll=1"
ENDPOINT_FILE = r"c:\Users\Администратор\Documents\project upgrade\kaggle_endpoint.txt"

cached_target = None
last_check = 0

def get_remote_target():
    global cached_target, last_check
    now = time.time()
    
    # 1. Check local override file if exists
    if os.path.exists(ENDPOINT_FILE):
        try:
            with open(ENDPOINT_FILE, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t.startswith("http"):
                    return t.rstrip("/")
        except:
            pass
            
    # 2. Re-check ntfy every 10 seconds if not cached
    if cached_target and (now - last_check < 15):
        return cached_target
        
    try:
        req = urllib.request.Request(NTFY_URL, headers={"User-Agent": "Mozilla/5.0"})
        res = urllib.request.urlopen(req, timeout=5)
        raw = res.read().decode("utf-8", errors="ignore")
        matches = re.findall(r"OPENCODE_API_URL:\s*(https://[a-zA-Z0-9-]+\.trycloudflare\.com/v1)", raw)
        if matches:
            cached_target = matches[-1].rstrip("/")
            last_check = now
            print(f"[BRIDGE] Found active Kaggle API endpoint: {cached_target}", flush=True)
            return cached_target
    except Exception as e:
        print(f"[BRIDGE] Error checking ntfy: {e}", flush=True)
        
    return cached_target

class OpenCodeProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write(f"[BRIDGE-REQ] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")
        sys.stdout.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/models":
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

        target_url = target_base + self.path
        if target_url.startswith(target_base + "/v1") and target_base.endswith("/v1"):
            target_url = target_base + self.path[3:]

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
        body = self.rfile.read(content_length)

        target_url = target_base + self.path
        if target_url.startswith(target_base + "/v1") and target_base.endswith("/v1"):
            target_url = target_base + self.path[3:]

        headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "User-Agent": "OpenCode-Bridge"
        }

        try:
            req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as res:
                self.send_response(res.status)
                for k, v in res.headers.items():
                    if k.lower() not in ["content-length"]:
                        self.send_header(k, v)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # Stream response line by line / chunk by chunk
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
            self.send_error(502, f"Bridge error connecting to Kaggle: {e}")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    import os
    print(f"=== OpenCode Kaggle Bridge Daemon ===", flush=True)
    print(f"Listening locally on: http://127.0.0.1:{PORT}/v1", flush=True)
    target = get_remote_target()
    if target:
        print(f"Initial target endpoint: {target}", flush=True)
    else:
        print("Waiting for Kaggle GPU tunnel to publish endpoint...", flush=True)

    server = ThreadedHTTPServer(("127.0.0.1", PORT), OpenCodeProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping bridge daemon...", flush=True)
