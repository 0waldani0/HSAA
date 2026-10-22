#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>

#pragma comment(lib, "psapi.lib")

// =============================================================================
// Constants & Hardware Invariants
// =============================================================================
#define VSA_DIM 512
#define VSA_WORDS (VSA_DIM / 64)       // 8 x uint64_t = 64 bytes = 1 cache line / __m512i
#define HIDDEN_DIM 1024
#define TENSOR_ALIGNMENT 64
#define MAX_WORKSTATION_RAM_MB  32768
#define MAX_WORKSTATION_VRAM_MB 4096

// =============================================================================
// Test Harness Assertion & Failure Exceptions
// =============================================================================
class TestFailureException : public std::runtime_error {
public:
    std::string file;
    int line;
    std::string expression;
    TestFailureException(const std::string& msg, const std::string& f, int l, const std::string& expr)
        : std::runtime_error(msg), file(f), line(l), expression(expr) {}
};

#define TEST_FAIL(msg) \
    throw TestFailureException(msg, __FILE__, __LINE__, #msg)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw TestFailureException("Assertion failed: " #cond " is FALSE", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) \
    do { \
        if ((cond)) { \
            throw TestFailureException("Assertion failed: " #cond " is TRUE", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (!(_a == _b)) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " == " #b " (" << _a << " != " << _b << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " == " #b); \
        } \
    } while (0)

#define ASSERT_NE(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (_a == _b) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " != " #b " (" << _a << " == " << _b << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " != " #b); \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        double _diff = std::fabs(static_cast<double>(a) - static_cast<double>(b)); \
        if (_diff > (eps)) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: |" #a " - " #b "| <= " #eps " (diff=" << _diff << " > " << (eps) << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " near " #b); \
        } \
    } while (0)

#define ASSERT_GT(a, b) \
    do { \
        if (!((a) > (b))) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " > " #b " (" << (a) << " <= " << (b) << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " > " #b); \
        } \
    } while (0)

#define ASSERT_GE(a, b) \
    do { \
        if (!((a) >= (b))) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " >= " #b " (" << (a) << " < " << (b) << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " >= " #b); \
        } \
    } while (0)

#define ASSERT_LT(a, b) \
    do { \
        if (!((a) < (b))) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " < " #b " (" << (a) << " >= " << (b) << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " < " #b); \
        } \
    } while (0)

#define ASSERT_LE(a, b) \
    do { \
        if (!((a) <= (b))) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " <= " #b " (" << (a) << " > " << (b) << ")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " <= " #b); \
        } \
    } while (0)

#define ASSERT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == nullptr) { \
            throw TestFailureException("Assertion failed: " #ptr " is NULL", __FILE__, __LINE__, #ptr " != nullptr"); \
        } \
    } while (0)

#define ASSERT_NULL(ptr) \
    do { \
        if ((ptr) != nullptr) { \
            throw TestFailureException("Assertion failed: " #ptr " is NOT NULL", __FILE__, __LINE__, #ptr " == nullptr"); \
        } \
    } while (0)

#define ASSERT_STR_EQ(a, b) \
    do { \
        const char* _sa = (a); \
        const char* _sb = (b); \
        if (_sa == nullptr || _sb == nullptr || std::strcmp(_sa, _sb) != 0) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #a " == " #b " (\"" << (_sa ? _sa : "NULL") << "\" != \"" << (_sb ? _sb : "NULL") << "\")"; \
            throw TestFailureException(_oss.str(), __FILE__, __LINE__, #a " == " #b); \
        } \
    } while (0)

// =============================================================================
// Test Metadata & Registry
// =============================================================================
struct TestCase {
    std::string name;
    int tier;                  // 1, 2, 3, 4
    std::string feature;       // e.g. "F1.1", "F1.2"
    std::string description;
    std::function<void()> func;
    bool passed = false;
    double elapsed_us = 0.0;
    std::string failure_message;
    std::string failure_file;
    int failure_line = 0;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void register_test(const std::string& name, int tier, const std::string& feature,
                       const std::string& desc, std::function<void()> func) {
        TestCase tc;
        tc.name = name;
        tc.tier = tier;
        tc.feature = feature;
        tc.description = desc;
        tc.func = func;
        tests_.push_back(tc);
    }

    std::vector<TestCase>& get_tests() { return tests_; }

private:
    std::vector<TestCase> tests_;
};

struct TestRegistrar {
    TestRegistrar(const std::string& name, int tier, const std::string& feature,
                  const std::string& desc, std::function<void()> func) {
        TestRegistry::instance().register_test(name, tier, feature, desc, func);
    }
};

#define TEST_CASE(name, tier, feature, desc) \
    static void test_func_##name(); \
    static TestRegistrar test_registrar_##name(#name, tier, feature, desc, test_func_##name); \
    static void test_func_##name()

// =============================================================================
// Hardware Metrics & Profiling Utilities
// =============================================================================
inline double qpc_time_us() {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (static_cast<double>(now.QuadPart) * 1000000.0) / static_cast<double>(freq.QuadPart);
}

inline size_t get_process_ram_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
}

inline size_t get_cuda_vram_mb() {
    // Dynamic query via nvcuda.dll Driver API (cuInit + cuDevicePrimaryCtxRetain + cuMemGetInfo_v2)
    typedef int CUresult;
    typedef int CUdevice;
    typedef void* CUcontext;

    typedef CUresult (__stdcall *PFN_cuInit)(unsigned int);
    typedef CUresult (__stdcall *PFN_cuDeviceGet)(CUdevice*, int);
    typedef CUresult (__stdcall *PFN_cuCtxGetCurrent)(CUcontext*);
    typedef CUresult (__stdcall *PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    typedef CUresult (__stdcall *PFN_cuCtxSetCurrent)(CUcontext);
    typedef CUresult (__stdcall *PFN_cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice);
    typedef CUresult (__stdcall *PFN_cuMemGetInfo_v2)(size_t*, size_t*);

    static HMODULE hCuda = NULL;
    static PFN_cuMemGetInfo_v2 pfnMemGetInfo = NULL;
    static bool attempted = false;
    static bool supported = false;

    if (!attempted) {
        attempted = true;
        hCuda = LoadLibraryA("nvcuda.dll");
        if (hCuda) {
            PFN_cuInit pfnInit = (PFN_cuInit)GetProcAddress(hCuda, "cuInit");
            PFN_cuDeviceGet pfnDeviceGet = (PFN_cuDeviceGet)GetProcAddress(hCuda, "cuDeviceGet");
            PFN_cuCtxGetCurrent pfnCtxGetCurrent = (PFN_cuCtxGetCurrent)GetProcAddress(hCuda, "cuCtxGetCurrent");
            PFN_cuDevicePrimaryCtxRetain pfnPrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(hCuda, "cuDevicePrimaryCtxRetain");
            PFN_cuCtxSetCurrent pfnCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(hCuda, "cuCtxSetCurrent");
            PFN_cuCtxCreate_v2 pfnCtxCreate = (PFN_cuCtxCreate_v2)GetProcAddress(hCuda, "cuCtxCreate_v2");
            pfnMemGetInfo = (PFN_cuMemGetInfo_v2)GetProcAddress(hCuda, "cuMemGetInfo_v2");

            if (pfnInit && pfnDeviceGet && pfnMemGetInfo && pfnInit(0) == 0) {
                // Tier 1: Check if a CUDA context is already active on this thread
                CUcontext cur = NULL;
                if (pfnCtxGetCurrent && pfnCtxGetCurrent(&cur) == 0 && cur != NULL) {
                    supported = true;
                } else {
                    // Tier 2: Acquire device 0 and retain primary context or create driver context
                    CUdevice dev = 0;
                    if (pfnDeviceGet(&dev, 0) == 0) {
                        CUcontext ctx = NULL;
                        if (pfnPrimaryCtxRetain && pfnCtxSetCurrent && pfnPrimaryCtxRetain(&ctx, dev) == 0 && ctx != NULL) {
                            pfnCtxSetCurrent(ctx);
                            supported = true;
                        } else if (pfnCtxCreate && pfnCtxCreate(&ctx, 0, dev) == 0 && ctx != NULL) {
                            supported = true;
                        }
                    }
                }
            }
        }
    }

    if (supported && pfnMemGetInfo) {
        size_t free_bytes = 0, total_bytes = 0;
        if (pfnMemGetInfo(&free_bytes, &total_bytes) == 0 && total_bytes > 0) {
            size_t used_bytes = total_bytes - free_bytes;
            return used_bytes / (1024 * 1024);
        }
    }
    return 0; // Explicit 0 indicates no CUDA driver / GPU present or query failure
}

// =============================================================================
// Mathematical & Algorithmic Reference Utilities
// =============================================================================
inline float compute_cosine_similarity(const float* a, const float* b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-300) return 0.0f;
    double cos_val = dot / denom;
    if (cos_val > 1.0) cos_val = 1.0;
    if (cos_val < -1.0) cos_val = -1.0;
    return static_cast<float>(cos_val);
}

inline float compute_sqnr_db(const float* signal, const float* noise, size_t n) {
    if (!signal || !noise || n == 0) return 0.0f;
    double p_signal = 0.0, p_noise = 0.0;
    for (size_t i = 0; i < n; ++i) {
        p_signal += static_cast<double>(signal[i]) * static_cast<double>(signal[i]);
        double err = static_cast<double>(signal[i]) - static_cast<double>(noise[i]);
        p_noise += err * err;
    }
    if (p_noise < 1e-15) return 120.0f; // Perfect reconstruction ceiling
    if (p_signal < 1e-15) return 0.0f;
    return static_cast<float>(10.0 * std::log10(p_signal / p_noise));
}

inline void vsa_bind_xnor(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    for (int w = 0; w < VSA_WORDS; ++w) {
        out[w] = ~(a[w] ^ b[w]);
    }
}

inline void vsa_permute_rotl(const uint64_t* in, int shift, uint64_t* out) {
    shift = (shift % VSA_DIM + VSA_DIM) % VSA_DIM;
    int word_shift = shift / 64;
    int bit_shift = shift % 64;

    uint64_t temp[VSA_WORDS];
    for (int i = 0; i < VSA_WORDS; ++i) {
        temp[(i + word_shift) % VSA_WORDS] = in[i];
    }
    if (bit_shift == 0) {
        for (int i = 0; i < VSA_WORDS; ++i) out[i] = temp[i];
        return;
    }
    for (int i = 0; i < VSA_WORDS; ++i) {
        int next = (i + 1) % VSA_WORDS;
        out[next] = (temp[next] << bit_shift) | (temp[i] >> (64 - bit_shift));
    }
}

inline int vsa_hamming_similarity(const uint64_t* a, const uint64_t* b) {
    int matches = 0;
    for (int w = 0; w < VSA_WORDS; ++w) {
        uint64_t xnor = ~(a[w] ^ b[w]);
        matches += static_cast<int>(__popcnt64(xnor));
    }
    return matches;
}

inline uint64_t fnv1a_hash(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    if (!str) return hash;
    while (*str) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(*str++));
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline void hash_to_vsa_vector(const char* str, uint64_t* out_vec) {
    uint64_t h = fnv1a_hash(str);
    for (int w = 0; w < VSA_WORDS; ++w) {
        // Linear congruential pseudo-random expansion
        h = h * 6364136223846793005ULL + 1442695040888963407ULL;
        out_vec[w] = h;
    }
}

// =============================================================================
// Interface Contracts & Reference Implementations
// =============================================================================

// --- 1. Bridge ---
struct HybridBuffer {
    alignas(64) uint64_t vsa_vec[VSA_WORDS];
    alignas(64) float    hidden_cpu[HIDDEN_DIM];
    void*                d_hidden_gpu;
    size_t               bytes_allocated;
    bool                 is_pinned;
};

inline HybridBuffer* test_bridge_alloc_buffer(size_t hidden_dim) {
    if (hidden_dim == 0) return nullptr;
    HybridBuffer* buf = (HybridBuffer*)_aligned_malloc(sizeof(HybridBuffer), TENSOR_ALIGNMENT);
    if (!buf) return nullptr;
    std::memset(buf->vsa_vec, 0, sizeof(buf->vsa_vec));
    std::memset(buf->hidden_cpu, 0, sizeof(buf->hidden_cpu));
    buf->d_hidden_gpu = (void*)(uintptr_t)0xDEADBEEF; // Simulated device pointer
    buf->bytes_allocated = sizeof(HybridBuffer);
    buf->is_pinned = true;
    return buf;
}

inline void test_bridge_free_buffer(HybridBuffer* buf) {
    if (buf) {
        _aligned_free(buf);
    }
}

inline int test_bridge_sync_vsa_to_gpu(HybridBuffer* buf, const uint64_t* vsa_vec, float scale) {
    if (!buf || !vsa_vec) return -1;
    std::memcpy(buf->vsa_vec, vsa_vec, sizeof(buf->vsa_vec));
    // Project 512 bits to 1024 floats (bipolar {-1, +1} * scale)
    for (int i = 0; i < 512; ++i) {
        int word = i / 64;
        int bit = i % 64;
        float val = ((vsa_vec[word] >> bit) & 1ULL) ? 1.0f : -1.0f;
        buf->hidden_cpu[i] = val * scale;
        buf->hidden_cpu[i + 512] = val * scale * 0.5f;
    }
    return 0;
}

inline int test_bridge_sync_gpu_to_vsa(HybridBuffer* buf, uint64_t* out_vsa_vec) {
    if (!buf || !out_vsa_vec) return -1;
    std::memset(out_vsa_vec, 0, VSA_WORDS * sizeof(uint64_t));
    for (int i = 0; i < 512; ++i) {
        if (buf->hidden_cpu[i] > 0.0f) {
            out_vsa_vec[i / 64] |= (1ULL << (i % 64));
        }
    }
    return 0;
}

// --- 2. VsaMemoryDatabase ---
enum VsaDbCapacity {
    VSA_CAP_MICRO  = 1024,
    VSA_CAP_MEDIUM = 16384,
    VSA_CAP_LARGE  = 262144
};

struct VsaFactTriple {
    std::string subject;
    std::string relation;
    std::string object;
    uint64_t vsa_vec[VSA_WORDS];
};

struct VsaDbHandle {
    VsaDbCapacity capacity;
    std::string persist_path;
    std::unordered_map<std::string, VsaFactTriple> facts;
    std::unordered_map<uint64_t, std::vector<uint64_t>> raw_vectors;
    size_t active_pages = 0;
};

inline VsaDbHandle* test_vsa_db_create(VsaDbCapacity cap, const char* persist_path) {
    if (cap != VSA_CAP_MICRO && cap != VSA_CAP_MEDIUM && cap != VSA_CAP_LARGE) {
        return nullptr;
    }
    VsaDbHandle* db = new VsaDbHandle();
    db->capacity = cap;
    if (persist_path) db->persist_path = persist_path;
    db->active_pages = 1;
    return db;
}

inline void test_vsa_db_destroy(VsaDbHandle* db) {
    if (db) {
        delete db;
    }
}

inline int test_vsa_db_flush(VsaDbHandle* db) {
    if (!db) return -1;
    return 0;
}

inline int test_vsa_db_clone(const VsaDbHandle* src, VsaDbHandle** dst, const char* new_persist_path) {
    if (!src || !dst) return -1;
    VsaDbHandle* clone = new VsaDbHandle();
    clone->capacity = src->capacity;
    clone->facts = src->facts;
    clone->raw_vectors = src->raw_vectors;
    clone->active_pages = src->active_pages;
    if (new_persist_path) clone->persist_path = new_persist_path;
    *dst = clone;
    return 0;
}

inline int test_vsa_db_insert_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* obj) {
    if (!db || !subj || !rel || !obj) return -1;
    std::string key = std::string(subj) + "::" + std::string(rel);
    VsaFactTriple trip;
    trip.subject = subj;
    trip.relation = rel;
    trip.object = obj;
    hash_to_vsa_vector((std::string(subj) + " " + rel + " " + obj).c_str(), trip.vsa_vec);
    db->facts[key] = trip;
    db->active_pages = 1 + (db->facts.size() / 16);
    return 0;
}

inline int test_vsa_db_delete_fact(VsaDbHandle* db, const char* subj, const char* rel) {
    if (!db || !subj || !rel) return -1;
    std::string key = std::string(subj) + "::" + std::string(rel);
    auto it = db->facts.find(key);
    if (it == db->facts.end()) return 0;
    db->facts.erase(it);
    db->active_pages = 1 + (db->facts.size() / 16);
    return 1;
}

inline int test_vsa_db_mutate_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* new_obj) {
    if (!db || !subj || !rel || !new_obj) return -1;
    return test_vsa_db_insert_fact(db, subj, rel, new_obj);
}

inline int test_vsa_db_query_relation(VsaDbHandle* db, const char* subj, const char* rel, char* out_obj, size_t max_len) {
    if (!db || !subj || !rel || !out_obj || max_len == 0) return 0;
    std::string key = std::string(subj) + "::" + std::string(rel);
    auto it = db->facts.find(key);
    if (it == db->facts.end()) return 0;
    strncpy_s(out_obj, max_len, it->second.object.c_str(), _TRUNCATE);
    return 1;
}

inline int test_vsa_db_check_conflict(VsaDbHandle* db, const char* subj, const char* rel, const char* obj) {
    if (!db || !subj || !rel || !obj) return -1;
    std::string key = std::string(subj) + "::" + std::string(rel);
    auto it = db->facts.find(key);
    if (it == db->facts.end()) return 0; // No existing fact, no conflict
    if (it->second.object == obj) return 0; // Same fact, consistent
    return 1; // Contradiction!
}

inline int test_vsa_db_insert_vector(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512) {
    if (!db || !vec_512) return -1;
    std::vector<uint64_t> v(vec_512, vec_512 + VSA_WORDS);
    db->raw_vectors[key_hash] = v;
    return 0;
}

inline int test_vsa_db_query_vector(VsaDbHandle* db, const uint64_t* query_512, uint64_t* out_best_vec, float* out_sim) {
    if (!db || !query_512 || !out_best_vec || !out_sim) return -1;
    if (db->raw_vectors.empty()) return 0;
    int best_matches = -1;
    const uint64_t* best_ptr = nullptr;
    for (const auto& kv : db->raw_vectors) {
        int m = vsa_hamming_similarity(query_512, kv.second.data());
        if (m > best_matches) {
            best_matches = m;
            best_ptr = kv.second.data();
        }
    }
    if (best_ptr) {
        std::memcpy(out_best_vec, best_ptr, VSA_WORDS * sizeof(uint64_t));
        *out_sim = static_cast<float>(2 * best_matches - VSA_DIM) / static_cast<float>(VSA_DIM);
        return 1;
    }
    return 0;
}

// --- 3. Model Modification ---
enum QuantPrecision {
    QUANT_2BIT = 2,
    QUANT_3BIT = 3,
    QUANT_4BIT = 4,
    QUANT_DELTA_24 = 24,
    QUANT_DELTA_34 = 34
};

enum LayerType {
    LAYER_GQA_ATTENTION = 0,
    LAYER_GATED_DELTANET = 1,
    LAYER_AVX512_VSA_ASSOCIATIVE = 2
};

struct ModelLayerState {
    uint32_t layer_idx;
    QuantPrecision precision = QUANT_4BIT;
    LayerType layer_type = LAYER_GQA_ATTENTION;
    std::vector<float> tt_u;
    std::vector<float> tt_v;
    uint32_t M = 1024, N = 1024, R = 16;
};

struct ModelContext {
    uint32_t num_layers = 28;
    std::vector<ModelLayerState> layers;
    ModelContext() {
        layers.resize(num_layers);
        for (uint32_t l = 0; l < num_layers; ++l) {
            layers[l].layer_idx = l;
            layers[l].tt_u.resize(16 * 1024, 0.0f);
            layers[l].tt_v.resize(16 * 1024, 0.0f);
        }
    }
};

inline int test_model_set_layer_precision(void* ctx, uint32_t layer_idx, QuantPrecision prec) {
    if (!ctx) return -1;
    ModelContext* m = static_cast<ModelContext*>(ctx);
    if (layer_idx >= m->num_layers) return -2;
    if (prec != QUANT_2BIT && prec != QUANT_3BIT && prec != QUANT_4BIT &&
        prec != QUANT_DELTA_24 && prec != QUANT_DELTA_34) {
        return -3;
    }
    m->layers[layer_idx].precision = prec;
    return 0;
}

inline int test_model_update_tt16_residual(void* ctx, uint32_t layer_idx, const float* u_delta, const float* v_delta) {
    if (!ctx || !u_delta || !v_delta) return -1;
    ModelContext* m = static_cast<ModelContext*>(ctx);
    if (layer_idx >= m->num_layers) return -2;
    size_t sz_u = m->layers[layer_idx].R * m->layers[layer_idx].N;
    size_t sz_v = m->layers[layer_idx].R * m->layers[layer_idx].M;
    float* dst_u = m->layers[layer_idx].tt_u.data();
    float* dst_v = m->layers[layer_idx].tt_v.data();
    size_t i = 0;
    for (; i + 16 <= sz_u; i += 16) {
        __m512 d = _mm512_loadu_ps(u_delta + i);
        __m512 u = _mm512_loadu_ps(dst_u + i);
        _mm512_storeu_ps(dst_u + i, _mm512_add_ps(u, d));
    }
    for (; i < sz_u; ++i) dst_u[i] += u_delta[i];

    size_t j = 0;
    for (; j + 16 <= sz_v; j += 16) {
        __m512 d = _mm512_loadu_ps(v_delta + j);
        __m512 v = _mm512_loadu_ps(dst_v + j);
        _mm512_storeu_ps(dst_v + j, _mm512_add_ps(v, d));
    }
    for (; j < sz_v; ++j) dst_v[j] += v_delta[j];
    return 0;
}

inline int test_model_swap_layer_block(void* ctx, uint32_t layer_idx, LayerType new_type) {
    if (!ctx) return -1;
    ModelContext* m = static_cast<ModelContext*>(ctx);
    if (layer_idx >= m->num_layers) return -2;
    if (new_type != LAYER_GQA_ATTENTION && new_type != LAYER_GATED_DELTANET &&
        new_type != LAYER_AVX512_VSA_ASSOCIATIVE) {
        return -3;
    }
    m->layers[layer_idx].layer_type = new_type;
    return 0;
}

// --- 4. Benchmark Runner ---
struct BenchmarkMetrics {
    double latency_us;
    double tokens_per_sec;
    size_t host_ram_working_set_mb;
    size_t gpu_vram_used_mb;
    float  cosine_similarity;
    float  sqnr_db;
};

inline int test_bench_run_step_profile(void* ctx, BenchmarkMetrics* out_metrics) {
    if (!out_metrics) return -1;
    double t0 = qpc_time_us();
    // Simulate step math:
    volatile float sum = 0.0f;
    for (int i = 0; i < 10000; ++i) sum += std::sin(static_cast<float>(i));
    double t1 = qpc_time_us();

    out_metrics->latency_us = t1 - t0;
    out_metrics->tokens_per_sec = out_metrics->latency_us > 0.0 ? (1000000.0 / out_metrics->latency_us) : 0.0;
    out_metrics->host_ram_working_set_mb = get_process_ram_mb();
    out_metrics->gpu_vram_used_mb = get_cuda_vram_mb();
    out_metrics->cosine_similarity = 0.9999f;
    out_metrics->sqnr_db = 42.5f;
    return 0;
}

inline int test_bench_assert_hardware_bounds(size_t max_ram_mb, size_t max_vram_mb) {
    if (max_ram_mb == 0 || max_vram_mb == 0) return -1;
    size_t cur_ram = get_process_ram_mb();
    size_t cur_vram = get_cuda_vram_mb();
    if (cur_ram > max_ram_mb) return -2;
    if (cur_vram > max_vram_mb) return -3;
    return 0;
}

inline int test_bench_run_adversarial_matrix() {
    // 0-1-N-NULL-BOUND test
    int errs = 0;
    if (test_bridge_alloc_buffer(0) != nullptr) errs++;
    if (test_bridge_sync_vsa_to_gpu(nullptr, nullptr, 1.0f) != -1) errs++;
    if (test_vsa_db_create((VsaDbCapacity)999, nullptr) != nullptr) errs++;
    if (test_vsa_db_insert_fact(nullptr, nullptr, nullptr, nullptr) != -1) errs++;
    if (test_model_set_layer_precision(nullptr, 0, QUANT_4BIT) != -1) errs++;
    if (test_bench_assert_hardware_bounds(0, 0) != -1) errs++;
    return errs == 0 ? 0 : -1;
}
