import subprocess
import sys
import time

def run_build(target):
    cmd = f'cmd.exe /c "build.bat {target}"'
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=r'c:\Users\Администратор\Documents\project upgrade')
    dt = time.time() - t0
    return res.returncode, res.stdout, res.stderr, dt

def main():
    print("Testing rapid repeated builds of model_mod and lab...", flush=True)
    for i in range(1, 4):
        print(f"\n>>> Iteration {i}: Building model_mod...", flush=True)
        rc, out, err, dt = run_build("model_mod")
        if rc != 0 or "LNK1104" in out or "LNK1104" in err:
            print(f"[FAIL] model_mod iteration {i} failed (rc={rc}, dt={dt:.2f}s):\nSTDOUT:\n{out}\nSTDERR:\n{err}", flush=True)
            sys.exit(1)
        print(f"[PASS] model_mod iteration {i} compiled in {dt:.2f}s with 0 LNK1104 collisions.", flush=True)

        print(f">>> Iteration {i}: Building lab...", flush=True)
        rc, out, err, dt = run_build("lab")
        if rc != 0 or "LNK1104" in out or "LNK1104" in err:
            print(f"[FAIL] lab iteration {i} failed (rc={rc}, dt={dt:.2f}s):\nSTDOUT:\n{out}\nSTDERR:\n{err}", flush=True)
            sys.exit(1)
        print(f"[PASS] lab iteration {i} compiled in {dt:.2f}s with 0 LNK1104 collisions.", flush=True)

    print("\n[SUCCESS] All rapid repeated builds of model_mod and lab passed with 0 LNK1104 collisions!", flush=True)

if __name__ == "__main__":
    main()
