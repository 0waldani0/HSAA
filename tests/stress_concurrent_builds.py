import subprocess
import concurrent.futures
import sys
import os

def run_build(target):
    cmd = ["cmd.exe", "/c", f"build.bat {target}"]
    res = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    return target, res.returncode, res.stdout, res.stderr

def main():
    print("[*] Launching concurrent builds: model_mod vs lab...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_build, "model_mod")
        f2 = executor.submit(run_build, "lab")
        res1 = f1.result()
        res2 = f2.result()

    print(f"[RESULT] target={res1[0]} code={res1[1]}")
    print(f"[RESULT] target={res2[0]} code={res2[1]}")

    has_error = False
    for res in [res1, res2]:
        out = res[2] + res[3]
        if "LNK1104" in out:
            print(f"[FAIL] Found LNK1104 in {res[0]} output!")
            has_error = True
        if res[1] != 0:
            print(f"[FAIL] {res[0]} failed with exit code {res[1]}")
            print(out)
            has_error = True

    if not has_error and res1[1] == 0 and res2[1] == 0:
        print("[SUCCESS] Both concurrent builds completed with exit code 0 and ZERO LNK1104 lock collisions!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
