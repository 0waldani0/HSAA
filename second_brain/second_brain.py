# [Gemini 3.8 Flash] Start: Python ctypes wrapper for PopCount Second Brain (AVX-512 VPOPCNTDQ)
import ctypes
import os
import time

DLL_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(DLL_DIR, "popcnt_brain.dll"),
    os.path.join(DLL_DIR, "build", "popcnt_brain.dll"),
    r"C:\Users\Администратор\Documents\project upgrade\second_brain\build\popcnt_brain.dll"
]

DLL_PATH = None
for p in candidate_paths:
    if os.path.exists(p):
        DLL_PATH = p
        break

class PopCountBrainStruct(ctypes.Structure):
    pass

class SecondBrain:
    def __init__(self, max_pages=4096, state_file="C:/models/second_brain_state.bin"):
        if not DLL_PATH or not os.path.exists(DLL_PATH):
            raise FileNotFoundError(f"popcnt_brain.dll not found in {candidate_paths}")
        
        self.lib = ctypes.CDLL(DLL_PATH)
        
        # Function signatures
        self.lib.brain_create.argtypes = [ctypes.c_size_t, ctypes.c_char_p]
        self.lib.brain_create.restype = ctypes.POINTER(PopCountBrainStruct)
        
        self.lib.brain_destroy.argtypes = [ctypes.POINTER(PopCountBrainStruct)]
        self.lib.brain_destroy.restype = None
        
        self.lib.brain_store_fact.argtypes = [ctypes.POINTER(PopCountBrainStruct), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.brain_store_fact.restype = ctypes.c_int
        
        self.lib.brain_query_relation.argtypes = [ctypes.POINTER(PopCountBrainStruct), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.brain_query_relation.restype = ctypes.c_int
        
        self.lib.brain_check_conflict.argtypes = [ctypes.POINTER(PopCountBrainStruct), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.brain_check_conflict.restype = ctypes.c_int
        
        self.lib.brain_get_stats.argtypes = [ctypes.POINTER(PopCountBrainStruct), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.brain_get_stats.restype = None
        
        # Ensure directory exists for state_file if provided
        if state_file:
            os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
            path_bytes = state_file.encode("ascii", errors="ignore")
        else:
            path_bytes = b"C:/models/second_brain_state.bin"
            
        self.handle = self.lib.brain_create(max_pages, path_bytes)
        if not self.handle:
            raise RuntimeError(f"Failed to create PopCount Second Brain instance on path: {state_file}")

    def store_fact(self, subject: str, relation: str, obj: str) -> bool:
        return bool(self.lib.brain_store_fact(self.handle, subject.encode("utf-8"), relation.encode("utf-8"), obj.encode("utf-8")))

    def query(self, subject: str, relation: str) -> str:
        buf = ctypes.create_string_buffer(512)
        res = self.lib.brain_query_relation(self.handle, subject.encode("utf-8"), relation.encode("utf-8"), buf, 512)
        if res:
            return buf.value.decode("utf-8")
        return None

    def has_conflict(self, subject: str, relation: str, proposed_obj: str) -> bool:
        return bool(self.lib.brain_check_conflict(self.handle, subject.encode("utf-8"), relation.encode("utf-8"), proposed_obj.encode("utf-8")))

    def get_stats(self):
        pages = ctypes.c_size_t(0)
        rules = ctypes.c_size_t(0)
        self.lib.brain_get_stats(self.handle, ctypes.byref(pages), ctypes.byref(rules))
        return {"pages": pages.value, "rules": rules.value}

    def close(self):
        if hasattr(self, "handle") and self.handle:
            self.lib.brain_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    print("[*] Initializing SecondBrain in Python (ASCII path C:/models/second_brain_state.bin)...")
    with SecondBrain(max_pages=1024) as brain:
        t0 = time.perf_counter()
        brain.store_fact("Qwen27B", "runs_on", "RTX3050_Laptop")
        brain.store_fact("RTX3050_Laptop", "vram_limit_gb", "4")
        brain.store_fact("Qwen27B", "gpu_layers", "14")
        t_store = (time.perf_counter() - t0) * 1e6
        print(f"[+] 3 facts stored in {t_store:.2f} us ({t_store/3:.2f} us/fact)")
        
        t0 = time.perf_counter()
        obj = brain.query("Qwen27B", "gpu_layers")
        t_query = (time.perf_counter() - t0) * 1e6
        print(f"[+] Query result: Qwen27B gpu_layers -> {obj} (in {t_query:.2f} us)")
        
        conflict = brain.has_conflict("Qwen27B", "gpu_layers", "65")
        print(f"[+] Conflict check (proposing '65' layers when '14' stored): Contradiction = {conflict}")
        
        stats = brain.get_stats()
        print(f"[+] Brain stats: {stats}")
    print("[SUCCESS] SecondBrain Python test completed flawlessly!")
# [Gemini 3.8 Flash] End
