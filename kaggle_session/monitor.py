import urllib.request
import time
import subprocess
import sys

channel = 'waldani_qwen38_27b_session'
url = f'https://ntfy.sh/{channel}/raw?poll=1'

print("Starting Kaggle Kernel Monitor...", flush=True)

# Fetch existing messages to only print new ones
seen_messages = set()
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req, timeout=5)
    for line in res.read().decode('utf-8', errors='ignore').split('\n'):
        if line.strip():
            seen_messages.add(line.strip())
except:
    pass
print(f"Skipped {len(seen_messages)} prior messages.", flush=True)

for i in range(60): # 10 minutes max
    # Check notifications from ntfy.sh
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=5)
        raw = res.read().decode('utf-8', errors='ignore').strip()
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        for line in lines:
            if line not in seen_messages:
                seen_messages.add(line)
                print(f"[KAGGLE UPDATE] {line}", flush=True)
                if "GRADIO_URL:" in line:
                    print(f"SUCCESS! {line}", flush=True)
                    sys.exit(0)
    except Exception as e:
        pass
    
    # Check kaggle kernels status every 30s
    if i % 3 == 0:
        try:
            status_out = subprocess.check_output(["kaggle", "kernels", "status", "waldani/notebook084829c55e"], text=True)
            print(f"[STATUS] {status_out.strip()}", flush=True)
            if "COMPLETE" in status_out or "ERROR" in status_out or "FAILED" in status_out:
                print("Kernel finished or error occurred.", flush=True)
                break
        except Exception as e:
            print(f"Status check error: {e}", flush=True)

    time.sleep(10)
