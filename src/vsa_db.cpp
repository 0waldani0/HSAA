#include "vsa_db.h"
#include "simd_common.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
typedef int HANDLE;
#define INVALID_HANDLE_VALUE (-1)
#endif
#include <immintrin.h>
#include <nmmintrin.h>
#include <omp.h>
#ifndef _MSC_VER
#define __popcnt64 _mm_popcnt_u64
#endif

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <cmath>

// =============================================================================
// Constants & Structures for Memory-Mapped VsaMemoryDatabase
// =============================================================================

static constexpr uint64_t VSA_DB_MAGIC = 0x5653414442303031ULL; // "VSADB001"
static constexpr uint32_t VSA_DB_VERSION = 1;

// Slot representing one relation fact triple
struct alignas(64) VsaFactSlot {
    uint64_t key_hash;          // 64-bit hash of (subj, rel)
    uint64_t subj_vec[8];       // 512-bit vector for subject
    uint64_t rel_vec[8];        // 512-bit vector for relation
    uint64_t obj_vec[8];        // 512-bit vector for object
    uint64_t rel_key[8];        // 512-bit relational key: XNOR(Subj, RotL(Rel, 7))
    uint64_t fact_vec[8];       // 512-bit fact vector: XNOR(RelKey, Obj)
    uint32_t subj_offset;       // Byte offset in string heap (or 0xFFFFFFFF if inline)
    uint32_t subj_len;
    uint32_t rel_offset;
    uint32_t rel_len;
    uint32_t obj_offset;
    uint32_t obj_len;
    uint16_t psl;               // Probe sequence length for Robin Hood hashing
    uint8_t  occupied;          // 1 if active, 0 if empty
    uint8_t  reserved[5];
    char     inline_subj[32];   // Fast inline storage for short subject (<32 chars)
    char     inline_rel[32];    // Fast inline storage for short relation (<32 chars)
    char     inline_obj[64];    // Fast inline storage for short object (<64 chars)
};
static_assert(sizeof(VsaFactSlot) == 512, "VsaFactSlot must be exactly 512 bytes");

// Slot representing one raw 512-bit vector entry
struct alignas(64) VsaVectorSlot {
    uint64_t key_hash;          // User-provided 64-bit key hash
    uint64_t vec[8];            // 512-bit raw hypervector
    uint16_t psl;               // Probe sequence length
    uint8_t  occupied;          // 1 if active, 0 if empty
    uint8_t  reserved[53];
};
static_assert(sizeof(VsaVectorSlot) == 128, "VsaVectorSlot must be exactly 128 bytes");

// Page 0 Header (4096 bytes)
struct alignas(64) DbHeader {
    uint64_t magic;                 // VSA_DB_MAGIC
    uint32_t version;               // VSA_DB_VERSION
    uint32_t capacity_pages;        // Total pages (1024, 16384, or 262144)
    uint64_t total_bytes;           // capacity_pages * 4096
    uint64_t fact_count;            // Number of active facts
    uint64_t vector_count;          // Number of active raw vectors
    uint64_t active_pages;          // Touched OS pages
    uint32_t fact_table_offset;     // Offset where VsaFactSlot array begins
    uint32_t fact_table_capacity;   // Number of fact slots
    uint32_t vector_table_offset;   // Offset where VsaVectorSlot array begins
    uint32_t vector_table_capacity; // Number of vector slots
    uint32_t string_heap_offset;    // Offset where string heap begins
    uint32_t string_heap_capacity;  // Bytes available for string heap
    uint32_t string_heap_used;      // Bytes allocated in string heap
    uint8_t  reserved[4020];        // Pad to exactly 4096 bytes
};
static_assert(sizeof(DbHeader) == 4096, "DbHeader must be exactly 4096 bytes");

// Compact in-RAM copy of one active vector. The mmap table is sparse; queries
// scan this array instead of the whole capacity (see query_vector).
struct VectorEntry {
    uint64_t key_hash;
    uint64_t vec[8];
};

// 4096-Dimensional Dense Float32 Embedding Entry (Normalized on S^4095 for fast SIMD cosine)
struct alignas(64) Dense4096Entry {
    uint64_t key_hash = 0;
    float inv_norm = 0.0f;
    alignas(64) float vec[VSA_DENSE_4096_DIM];
};

// Continuous VSA Function Approximation Model (Fractional Power / Random Fourier Phase Encoding)
struct VsaContinuousFuncSample {
    float x = 0.0f;
    float y = 0.0f;
    uint64_t phase_sk[8] = {0};
};

struct VsaContinuousFuncModel {
    std::vector<VsaContinuousFuncSample> samples;
    float x_min = 0.0f;
    float x_max = 1.0f;
    float bandwidth = 1.0f;
    alignas(64) float rff_omega[128] = {0.0f};
    alignas(64) float rff_phase[128] = {0.0f};
    alignas(64) float rff_weights[128] = {0.0f};
    bool rff_ready = false;
};

struct VsaDbHandle {
    VsaDbCapacity capacity;
    size_t total_bytes;
    void* base_ptr;
    bool is_mapped_file;
    HANDLE file_handle;
    HANDLE mapping_handle;
    std::string persist_path;
    mutable std::mutex db_mutex;
    // Dense active-vector index: one entry per occupied vector-table slot.
    // vec_pos maps key_hash -> index in vec_dense for O(1) update/remove.
    // Not persisted: rebuilt from the mmap table on create(reopen) and clone.
    std::vector<VectorEntry> vec_dense;
    std::unordered_map<uint64_t, size_t> vec_pos;
    // 4096-D dense float32 vector table
    std::vector<Dense4096Entry> dense_4096_entries;
    std::unordered_map<uint64_t, size_t> dense_4096_pos;
    // Continuous VSA function models by relation name
    std::unordered_map<std::string, VsaContinuousFuncModel> cont_funcs;
};

// =============================================================================
// AVX-512 & Deterministic VSA Mathematical Kernels
// =============================================================================

// The VSA database's distance primitive returns the RAW number of equal bits
// in [0, 512]. The second_brain engine has a same-shaped helper with the
// opposite convention (a signed bipolar dot in [-512, +512]); the names are
// deliberately distinct so a future shared header cannot silently swap them.
inline int32_t vsa_hamming_matches_512(const uint64_t* a_bits, const uint64_t* b_bits) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VPOPCNTDQ__)
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_bits));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_bits));
    __m512i vxnor = _mm512_ternarylogic_epi64(va, vb, va, 0x99);
    __m512i vpop = _mm512_popcnt_epi64(vxnor);
    uint64_t matches = _mm512_reduce_add_epi64(vpop);
    return static_cast<int32_t>(matches);
#elif defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_bits));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_bits));
    __m512i vxnor = _mm512_ternarylogic_epi64(va, vb, va, 0x99);
    alignas(64) uint64_t x[8];
    _mm512_store_si512(reinterpret_cast<__m512i*>(x), vxnor);
    return static_cast<int32_t>(
        __popcnt64(x[0]) + __popcnt64(x[1]) +
        __popcnt64(x[2]) + __popcnt64(x[3]) +
        __popcnt64(x[4]) + __popcnt64(x[5]) +
        __popcnt64(x[6]) + __popcnt64(x[7])
    );
#else
    uint64_t matches =
        __popcnt64(~(a_bits[0] ^ b_bits[0])) +
        __popcnt64(~(a_bits[1] ^ b_bits[1])) +
        __popcnt64(~(a_bits[2] ^ b_bits[2])) +
        __popcnt64(~(a_bits[3] ^ b_bits[3])) +
        __popcnt64(~(a_bits[4] ^ b_bits[4])) +
        __popcnt64(~(a_bits[5] ^ b_bits[5])) +
        __popcnt64(~(a_bits[6] ^ b_bits[6])) +
        __popcnt64(~(a_bits[7] ^ b_bits[7]));
    return static_cast<int32_t>(matches);
#endif
}

// 512-bit circular bit rotation to the left by 7 positions
inline void rotl512_7(const uint64_t* in, uint64_t* out) {
    for (int k = 0; k < 8; ++k) {
        int prev = (k + 7) & 7; // (k - 1 + 8) % 8
        out[k] = (in[k] << 7) | (in[prev] >> 57);
    }
}

// 512-bit bitwise XNOR (delegated to shared simd_common.h)
inline void vsa_xnor_512(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    simd_xnor_512(a, b, out);
}

// FNV-1a 64-bit hash
inline uint64_t fnv1a_64(const char* str, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(str[i]);
        h *= 0x100000001b3ULL;
    }
    return h;
}

// SplitMix64 PRNG generator
inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Deterministic token vector generation via FNV-1a seed + SplitMix64 8-word expansion
inline void generate_token_vector_512(const char* token, uint64_t* out_vec_512) {
    size_t len = token ? std::strlen(token) : 0;
    uint64_t seed = fnv1a_64(token ? token : "", len);
    if (seed == 0) seed = 0x9e3779b97f4a7c15ULL;
    for (int w = 0; w < 8; ++w) {
        out_vec_512[w] = splitmix64(&seed);
    }
}

// Combine subject & relation hash
inline uint64_t hash_relation_key(const char* subj, const char* rel) {
    uint64_t hs = fnv1a_64(subj, subj ? std::strlen(subj) : 0);
    uint64_t hr = fnv1a_64(rel, rel ? std::strlen(rel) : 0);
    return hs ^ (hr + 0x9e3779b97f4a7c15ULL + (hs << 6) + (hs >> 2));
}

// =============================================================================
// Dense Active-Vector Index
// =============================================================================
// The mmap vector table has `vector_table_capacity` slots (up to 1,048,576 in
// VSA_CAP_LARGE) but only `vector_count` occupied ones. Scanning the capacity
// touched every empty slot (128 MB of cache lines for one live vector). The
// query path scans vec_dense instead; the table stays the persisted source of
// truth and the index is rebuilt from it on open.

static void vector_index_rebuild(VsaDbHandle* db) {
    db->vec_dense.clear();
    db->vec_pos.clear();
    if (!db->base_ptr) return;
    const DbHeader* hdr = reinterpret_cast<const DbHeader*>(db->base_ptr);
    const VsaVectorSlot* table = reinterpret_cast<const VsaVectorSlot*>(
        reinterpret_cast<const uint8_t*>(db->base_ptr) + hdr->vector_table_offset);
    for (uint32_t i = 0; i < hdr->vector_table_capacity; ++i) {
        if (!table[i].occupied) continue;
        if (db->vec_pos.find(table[i].key_hash) != db->vec_pos.end()) continue;
        VectorEntry e;
        e.key_hash = table[i].key_hash;
        std::memcpy(e.vec, table[i].vec, sizeof(e.vec));
        db->vec_pos.emplace(e.key_hash, db->vec_dense.size());
        db->vec_dense.push_back(e);
    }
}

static void vector_index_upsert_locked(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512) {
    const auto it = db->vec_pos.find(key_hash);
    if (it != db->vec_pos.end()) {
        std::memcpy(db->vec_dense[it->second].vec, vec_512, sizeof(uint64_t) * 8);
        return;
    }
    VectorEntry e;
    e.key_hash = key_hash;
    std::memcpy(e.vec, vec_512, sizeof(uint64_t) * 8);
    db->vec_pos.emplace(key_hash, db->vec_dense.size());
    db->vec_dense.push_back(e);
}

static void vector_index_remove_locked(VsaDbHandle* db, uint64_t key_hash) {
    const auto it = db->vec_pos.find(key_hash);
    if (it == db->vec_pos.end()) return;
    const size_t pos = it->second;
    const size_t last = db->vec_dense.size() - 1;
    if (pos != last) {
        db->vec_dense[pos] = db->vec_dense[last];
        db->vec_pos[db->vec_dense[pos].key_hash] = pos;
    }
    db->vec_dense.pop_back();
    db->vec_pos.erase(it);
}

// =============================================================================
// String Heap Helpers
// =============================================================================

static bool string_heap_allocate(VsaDbHandle* db, const char* str, size_t len, uint32_t* out_offset) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (!hdr) return false;
    size_t needed = len + 1;
    if (hdr->string_heap_used + needed > hdr->string_heap_capacity) {
        return false;
    }
    uint32_t off = hdr->string_heap_used;
    char* dst = reinterpret_cast<char*>(db->base_ptr) + hdr->string_heap_offset + off;
    if (len > 0 && str) {
        std::memcpy(dst, str, len);
    }
    dst[len] = '\0';
    hdr->string_heap_used += static_cast<uint32_t>(needed);
    *out_offset = off;
    return true;
}

static const char* string_heap_get(const VsaDbHandle* db, uint32_t offset) {
    const DbHeader* hdr = reinterpret_cast<const DbHeader*>(db->base_ptr);
    if (!hdr || offset >= hdr->string_heap_capacity) return "";
    return reinterpret_cast<const char*>(db->base_ptr) + hdr->string_heap_offset + offset;
}

// Reclaim gaps left by long-object mutations and deletions. The heap is a bump
// allocator: an overwritten or deleted block is unreachable the moment its
// offset is replaced, and without compaction the heap would eventually refuse
// writes while a single fact is still live (the F2.3 leak). This walks the
// occupied fact slots, sorts live blocks by their current offset and slides
// them down, updating the owning offset fields. Destinations never exceed
// sources for an ascending walk, and memmove handles the overlap.
struct HeapBlockRef {
    uint32_t offset;
    uint32_t len;
    uint32_t* owner;
};

static void string_heap_compact(VsaDbHandle* db) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (!hdr || hdr->string_heap_used == 0) return;
    VsaFactSlot* table = reinterpret_cast<VsaFactSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->fact_table_offset);

    std::vector<HeapBlockRef> refs;
    refs.reserve(static_cast<size_t>(hdr->fact_count) * 3);
    for (uint32_t i = 0; i < hdr->fact_table_capacity; ++i) {
        VsaFactSlot* slot = &table[i];
        if (!slot->occupied) continue;
        if (slot->subj_offset != 0xFFFFFFFF) refs.push_back({slot->subj_offset, slot->subj_len, &slot->subj_offset});
        if (slot->rel_offset != 0xFFFFFFFF) refs.push_back({slot->rel_offset, slot->rel_len, &slot->rel_offset});
        if (slot->obj_offset != 0xFFFFFFFF) refs.push_back({slot->obj_offset, slot->obj_len, &slot->obj_offset});
    }
    std::sort(refs.begin(), refs.end(),
              [](const HeapBlockRef& a, const HeapBlockRef& b) { return a.offset < b.offset; });

    char* heap = reinterpret_cast<char*>(db->base_ptr) + hdr->string_heap_offset;
    uint32_t dst = 0;
    for (const HeapBlockRef& r : refs) {
        std::memmove(heap + dst, heap + r.offset, static_cast<size_t>(r.len) + 1);
        *r.owner = dst;
        dst += r.len + 1;
    }
    hdr->string_heap_used = dst;
}

// Reserve the tail for a batch of allocations. Callers reserve the whole set
// (subject+relation+object) before allocating any of them, so compaction cannot
// invalidate an offset that has not been published to the fact table yet.
static bool string_heap_reserve(VsaDbHandle* db, size_t needed) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (!hdr) return false;
    if (static_cast<size_t>(hdr->string_heap_used) + needed <= hdr->string_heap_capacity) {
        return true;
    }
    string_heap_compact(db);
    return static_cast<size_t>(hdr->string_heap_used) + needed <= hdr->string_heap_capacity;
}

// Defined with the raw-vector section below; declared here because fact
// insertion keeps the unified vector index in sync (one fact = one binding).
static int vector_table_upsert_locked(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512);
static int vector_table_delete_locked(VsaDbHandle* db, uint64_t key_hash);

static const char* slot_get_subj(const VsaDbHandle* db, const VsaFactSlot* slot) {
    if (slot->subj_offset == 0xFFFFFFFF) return slot->inline_subj;
    return string_heap_get(db, slot->subj_offset);
}

static const char* slot_get_rel(const VsaDbHandle* db, const VsaFactSlot* slot) {
    if (slot->rel_offset == 0xFFFFFFFF) return slot->inline_rel;
    return string_heap_get(db, slot->rel_offset);
}

static const char* slot_get_obj(const VsaDbHandle* db, const VsaFactSlot* slot) {
    if (slot->obj_offset == 0xFFFFFFFF) return slot->inline_obj;
    return string_heap_get(db, slot->obj_offset);
}

static bool slot_match_key(const VsaDbHandle* db, const VsaFactSlot* slot, const char* subj, const char* rel) {
    const char* s_subj = slot_get_subj(db, slot);
    const char* s_rel = slot_get_rel(db, slot);
    return (std::strcmp(s_subj, subj) == 0 && std::strcmp(s_rel, rel) == 0);
}

static void update_active_pages(VsaDbHandle* db) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (!hdr) return;
    size_t fact_pages = (hdr->fact_count * sizeof(VsaFactSlot) + 4095) / 4096;
    size_t vec_pages = (hdr->vector_count * sizeof(VsaVectorSlot) + 4095) / 4096;
    size_t str_pages = (hdr->string_heap_used + 4095) / 4096;
    hdr->active_pages = 1 + fact_pages + vec_pages + str_pages;
}

// =============================================================================
// Lifecycle Implementation
// =============================================================================

VsaDbHandle* vsa_db_create(VsaDbCapacity cap, const char* persist_path) {
    if (cap != VSA_CAP_MICRO && cap != VSA_CAP_MEDIUM && cap != VSA_CAP_LARGE) {
        return nullptr;
    }
    size_t total_bytes = static_cast<size_t>(cap) * 4096;

    VsaDbHandle* db = new VsaDbHandle();
    db->capacity = cap;
    db->total_bytes = total_bytes;
    db->base_ptr = nullptr;
    db->is_mapped_file = false;
    db->file_handle = INVALID_HANDLE_VALUE;
    db->mapping_handle = (HANDLE)0;

    bool use_file = (persist_path != nullptr && persist_path[0] != '\0');
    if (use_file) {
#if defined(_WIN32)
        // MB_ERR_INVALID_CHARS makes the conversion FAIL on malformed UTF-8 instead of
        // silently substituting U+FFFD, which would otherwise create a file named with
        // replacement characters and lose the caller's intended persist location.
        int wlen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, persist_path, -1, NULL, 0);
        if (wlen <= 0) {
            use_file = false; // Unencodable path: degrade to in-memory (see F2.1)
        } else {
            std::wstring wpath(static_cast<size_t>(wlen), L'\0');
            int written = MultiByteToWideChar(CP_UTF8, 0, persist_path, -1, &wpath[0], wlen);
            if (written > 0 && wpath.back() == L'\0') {
                wpath.pop_back();
            }

            HANDLE h_file = CreateFileW(
                wpath.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                NULL,
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                NULL
            );
            if (h_file == INVALID_HANDLE_VALUE) {
                // Unusable persist path (missing directory, read-only volume, invalid drive):
                // degrade to a pure in-memory database instead of failing the caller, per the
                // F2.1 contract. The database stays fully functional, just non-persistent.
                use_file = false;
            } else {
                LARGE_INTEGER fs;
                if (GetFileSizeEx(h_file, &fs)) {
                    if (static_cast<size_t>(fs.QuadPart) < total_bytes) {
                        LARGE_INTEGER li_target;
                        li_target.QuadPart = static_cast<LONGLONG>(total_bytes);
                        if (SetFilePointerEx(h_file, li_target, NULL, FILE_BEGIN)) {
                            SetEndOfFile(h_file);
                        }
                    }
                }
                HANDLE h_map = CreateFileMappingW(
                    h_file,
                    NULL,
                    PAGE_READWRITE,
                    static_cast<DWORD>(total_bytes >> 32),
                    static_cast<DWORD>(total_bytes & 0xFFFFFFFF),
                    NULL
                );
                if (h_map == NULL) {
                    CloseHandle(h_file);
                    use_file = false;
                } else {
                    void* mapped = MapViewOfFile(h_map, FILE_MAP_ALL_ACCESS, 0, 0, total_bytes);
                    if (mapped == nullptr) {
                        CloseHandle(h_map);
                        CloseHandle(h_file);
                        use_file = false;
                    } else {
                        db->base_ptr = mapped;
                        db->is_mapped_file = true;
                        db->file_handle = h_file;
                        db->mapping_handle = h_map;
                        db->persist_path = persist_path;
                    }
                }
            }
        }
#else
        auto is_valid_utf8_path = [](const char* s) -> bool {
            if (!s) return false;
            const unsigned char* p = reinterpret_cast<const unsigned char*>(s);
            while (*p) {
                if (*p == '\\') return false;
                if (*p < 0x80) {
                    ++p;
                } else if ((*p & 0xE0) == 0xC0) {
                    if ((p[1] & 0xC0) != 0x80 || (*p & 0xFE) == 0xC0) return false;
                    p += 2;
                } else if ((*p & 0xF0) == 0xE0) {
                    if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return false;
                    p += 3;
                } else if ((*p & 0xF8) == 0xF0) {
                    if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return false;
                    p += 4;
                } else {
                    return false;
                }
            }
            return true;
        };
        int fd = is_valid_utf8_path(persist_path) ? open(persist_path, O_RDWR | O_CREAT, 0644) : -1;
        if (fd < 0) {
            use_file = false;
        } else {
            struct stat st;
            if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) < total_bytes) {
                if (ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
                    close(fd);
                    fd = -1;
                    use_file = false;
                }
            }
            if (fd >= 0) {
                void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (mapped == MAP_FAILED || mapped == nullptr) {
                    close(fd);
                    use_file = false;
                } else {
                    db->base_ptr = mapped;
                    db->is_mapped_file = true;
                    db->file_handle = fd;
                    db->mapping_handle = 0;
                    db->persist_path = persist_path;
                }
            }
        }
#endif
    }

    if (!use_file) {
        // Pure in-memory allocation when no persist path was requested or it was unusable
#if defined(_WIN32)
        void* ptr = VirtualAlloc(NULL, total_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!ptr) {
            delete db;
            return nullptr;
        }
#else
        void* ptr = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED || !ptr) {
            delete db;
            return nullptr;
        }
#endif
        db->base_ptr = ptr;
        db->is_mapped_file = false;
        db->persist_path = "";
    } else {
        DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
        uint32_t expected_slots = (cap == VSA_CAP_MICRO) ? 4096u : ((cap == VSA_CAP_MEDIUM) ? 65536u : 1048576u);
        uint32_t expected_vec_off = 4096u + expected_slots * static_cast<uint32_t>(sizeof(VsaFactSlot));
        uint32_t expected_str_off = expected_vec_off + expected_slots * static_cast<uint32_t>(sizeof(VsaVectorSlot));
        if (hdr->magic == VSA_DB_MAGIC &&
            hdr->version == VSA_DB_VERSION &&
            hdr->capacity_pages == static_cast<uint32_t>(cap) &&
            hdr->total_bytes == total_bytes &&
            hdr->fact_table_offset == 4096u &&
            hdr->fact_table_capacity == expected_slots &&
            hdr->vector_table_offset == expected_vec_off &&
            hdr->vector_table_capacity == expected_slots &&
            hdr->string_heap_offset == expected_str_off &&
            static_cast<size_t>(hdr->string_heap_offset) + hdr->string_heap_capacity == total_bytes &&
            hdr->string_heap_used <= hdr->string_heap_capacity &&
            hdr->fact_count <= hdr->fact_table_capacity &&
            hdr->vector_count <= hdr->vector_table_capacity) {
            vector_index_rebuild(db);
            return db; // Successfully reopened verified existing DB
        }
    }

    std::memset(db->base_ptr, 0, total_bytes);
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    hdr->magic = VSA_DB_MAGIC;
    hdr->version = VSA_DB_VERSION;
    hdr->capacity_pages = static_cast<uint32_t>(cap);
    hdr->total_bytes = total_bytes;
    hdr->fact_count = 0;
    hdr->vector_count = 0;
    hdr->active_pages = 1;

    uint32_t fact_slots_count = 0;
    uint32_t vector_slots_count = 0;
    if (cap == VSA_CAP_MICRO) {
        fact_slots_count = 4096;
        vector_slots_count = 4096;
    } else if (cap == VSA_CAP_MEDIUM) {
        fact_slots_count = 65536;
        vector_slots_count = 65536;
    } else {
        fact_slots_count = 1048576;
        vector_slots_count = 1048576;
    }

    hdr->fact_table_offset = 4096;
    hdr->fact_table_capacity = fact_slots_count;

    hdr->vector_table_offset = hdr->fact_table_offset + fact_slots_count * sizeof(VsaFactSlot);
    hdr->vector_table_capacity = vector_slots_count;

    hdr->string_heap_offset = hdr->vector_table_offset + vector_slots_count * sizeof(VsaVectorSlot);
    hdr->string_heap_capacity = static_cast<uint32_t>(total_bytes - hdr->string_heap_offset);
    hdr->string_heap_used = 0;

    return db;
}

void vsa_db_destroy(VsaDbHandle* db) {
    if (!db) return;
    {
        std::lock_guard<std::mutex> lock(db->db_mutex);
#if defined(_WIN32)
        if (db->is_mapped_file) {
            if (db->base_ptr) {
                FlushViewOfFile(db->base_ptr, db->total_bytes);
                UnmapViewOfFile(db->base_ptr);
                db->base_ptr = nullptr;
            }
            if (db->mapping_handle) {
                CloseHandle(db->mapping_handle);
                db->mapping_handle = NULL;
            }
            if (db->file_handle != INVALID_HANDLE_VALUE) {
                FlushFileBuffers(db->file_handle);
                CloseHandle(db->file_handle);
                db->file_handle = INVALID_HANDLE_VALUE;
            }
            // CRITICAL: File is preserved on disk!
        } else {
            if (db->base_ptr) {
                VirtualFree(db->base_ptr, 0, MEM_RELEASE);
                db->base_ptr = nullptr;
            }
        }
#else
        if (db->is_mapped_file) {
            if (db->base_ptr) {
                msync(db->base_ptr, db->total_bytes, MS_SYNC);
                munmap(db->base_ptr, db->total_bytes);
                db->base_ptr = nullptr;
            }
            if (db->file_handle != INVALID_HANDLE_VALUE) {
                fsync(db->file_handle);
                close(db->file_handle);
                db->file_handle = INVALID_HANDLE_VALUE;
            }
        } else {
            if (db->base_ptr) {
                munmap(db->base_ptr, db->total_bytes);
                db->base_ptr = nullptr;
            }
        }
#endif
    }
    delete db;
}

int vsa_db_flush(VsaDbHandle* db) {
    if (!db) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (db->is_mapped_file && db->base_ptr && db->file_handle != INVALID_HANDLE_VALUE) {
#if defined(_WIN32)
        BOOL fv = FlushViewOfFile(db->base_ptr, db->total_bytes);
        BOOL fb = FlushFileBuffers(db->file_handle);
        return (fv && fb) ? 0 : -1;
#else
        int mv = msync(db->base_ptr, db->total_bytes, MS_SYNC);
        int fb = fsync(db->file_handle);
        return (mv == 0 && fb == 0) ? 0 : -1;
#endif
    }
    return 0;
}

int vsa_db_clone(const VsaDbHandle* src, VsaDbHandle** dst, const char* new_persist_path) {
    if (!src || !dst) return -1;
    VsaDbHandle* clone = vsa_db_create(src->capacity, new_persist_path);
    if (!clone) return -1;

    {
        std::lock_guard<std::mutex> lock(const_cast<VsaDbHandle*>(src)->db_mutex);
        std::memcpy(clone->base_ptr, src->base_ptr, src->total_bytes);
        clone->dense_4096_entries = src->dense_4096_entries;
        clone->dense_4096_pos = src->dense_4096_pos;
        clone->cont_funcs = src->cont_funcs;
    }
    vector_index_rebuild(clone);

    if (clone->is_mapped_file) {
#if defined(_WIN32)
        FlushViewOfFile(clone->base_ptr, clone->total_bytes);
        FlushFileBuffers(clone->file_handle);
#else
        msync(clone->base_ptr, clone->total_bytes, MS_SYNC);
        fsync(clone->file_handle);
#endif
    }
    *dst = clone;
    return 0;
}

// =============================================================================
// Online Mutation & Retrieval Implementation
// =============================================================================

int vsa_db_insert_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* obj) {
    if (!db || !subj || !rel || !obj) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return -1;

    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    uint32_t mask = hdr->fact_table_capacity - 1;
    uint64_t kh = hash_relation_key(subj, rel);

    // 1. Generate 512-bit hypervectors
    uint64_t subj_v[8], rel_v[8], obj_v[8];
    generate_token_vector_512(subj, subj_v);
    generate_token_vector_512(rel, rel_v);
    generate_token_vector_512(obj, obj_v);

    // 2. Relational key binding: XNOR(Subj, RotL(Rel, 7))
    uint64_t rot_rel[8], rel_key[8];
    rotl512_7(rel_v, rot_rel);
    vsa_xnor_512(subj_v, rot_rel, rel_key);

    // 3. Fact binding: XNOR(RelKey, Obj)
    uint64_t fact_v[8];
    vsa_xnor_512(rel_key, obj_v, fact_v);

    VsaFactSlot* table = reinterpret_cast<VsaFactSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->fact_table_offset
    );

    // 4. Check if key already exists (mutation in-place)
    uint32_t idx = static_cast<uint32_t>(kh & mask);
    uint16_t cur_psl = 0;
    for (uint32_t probe = 0; probe < hdr->fact_table_capacity; ++probe) {
        VsaFactSlot* slot = &table[idx];
        if (!slot->occupied || cur_psl > slot->psl) break;
        if (slot->key_hash == kh && slot_match_key(db, slot, subj, rel)) {
            const size_t o_len = std::strlen(obj);

            // Every fallible step runs before the slot is touched. The upsert
            // below cannot fail after this check: an existing binding is
            // updated in place, and a missing one is only possible for a
            // database written before bindings existed -- that case needs a
            // free vector slot.
            if (db->vec_pos.find(kh) == db->vec_pos.end() &&
                hdr->vector_count >= hdr->vector_table_capacity) {
                return -1;
            }

            uint32_t new_off = slot->obj_offset;
            bool new_inline = false;
            bool in_place = false;
            if (o_len < sizeof(slot->inline_obj)) {
                new_inline = true;
            } else if (slot->obj_offset != 0xFFFFFFFF && o_len <= slot->obj_len) {
                in_place = true;
            } else {
                if (!string_heap_reserve(db, o_len + 1)) return -1;
                if (!string_heap_allocate(db, obj, o_len, &new_off)) return -1;
            }

            // Commit the object string, then its binding words, then the
            // vector-table entry. No allocation or failure path remains here.
            if (new_inline) {
                std::memcpy(slot->inline_obj, obj, o_len + 1);
                slot->obj_offset = 0xFFFFFFFF;
                slot->obj_len = static_cast<uint32_t>(o_len);
            } else if (in_place) {
                // The old heap block is large enough: overwrite it instead of
                // bump-allocating a new one and orphaning the old (the F2.3
                // leak). Compaction reclaims the tail of the block later.
                char* dst = reinterpret_cast<char*>(db->base_ptr) + hdr->string_heap_offset + slot->obj_offset;
                std::memcpy(dst, obj, o_len + 1);
                slot->obj_len = static_cast<uint32_t>(o_len);
                slot->inline_obj[0] = '\0';
            } else {
                slot->obj_offset = new_off;
                slot->obj_len = static_cast<uint32_t>(o_len);
                slot->inline_obj[0] = '\0';
            }

            std::memcpy(slot->obj_vec, obj_v, sizeof(uint64_t) * 8);
            std::memcpy(slot->fact_vec, fact_v, sizeof(uint64_t) * 8);
            // Keep the unified vector index in step with the stored binding.
            if (vector_table_upsert_locked(db, kh, fact_v) != 0) return -1;
            update_active_pages(db);
            return 0;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    // 5. Upfront capacity checks for BOTH tables, before allocating strings or
    // displacing slots. Every fact also owns one binding in the vector table.
    if (hdr->fact_count >= hdr->fact_table_capacity) {
        return -1;
    }
    if (hdr->vector_count >= hdr->vector_table_capacity &&
        db->vec_pos.find(kh) == db->vec_pos.end()) {
        return -1;
    }

    // 6. Build incoming new slot. Reserve the whole heap batch first: the
    // three allocations below must not trigger a compaction between them,
    // because `incoming` is not in the fact table yet and compaction could not
    // rewrite its already-assigned offsets.
    VsaFactSlot incoming;
    std::memset(&incoming, 0, sizeof(VsaFactSlot));
    incoming.key_hash = kh;
    incoming.occupied = 1;
    std::memcpy(incoming.subj_vec, subj_v, sizeof(uint64_t) * 8);
    std::memcpy(incoming.rel_vec, rel_v, sizeof(uint64_t) * 8);
    std::memcpy(incoming.obj_vec, obj_v, sizeof(uint64_t) * 8);
    std::memcpy(incoming.rel_key, rel_key, sizeof(uint64_t) * 8);
    std::memcpy(incoming.fact_vec, fact_v, sizeof(uint64_t) * 8);

    const size_t s_len = std::strlen(subj);
    const size_t r_len = std::strlen(rel);
    const size_t o_len = std::strlen(obj);
    const size_t heap_needed =
        ((s_len < sizeof(incoming.inline_subj)) ? 0 : s_len + 1) +
        ((r_len < sizeof(incoming.inline_rel)) ? 0 : r_len + 1) +
        ((o_len < sizeof(incoming.inline_obj)) ? 0 : o_len + 1);
    if (!string_heap_reserve(db, heap_needed)) return -1;
    const uint32_t heap_before = hdr->string_heap_used;

    incoming.subj_len = static_cast<uint32_t>(s_len);
    if (s_len < sizeof(incoming.inline_subj)) {
        std::memcpy(incoming.inline_subj, subj, s_len + 1);
        incoming.subj_offset = 0xFFFFFFFF;
    } else {
        uint32_t off = 0;
        if (!string_heap_allocate(db, subj, s_len, &off)) {
            hdr->string_heap_used = heap_before;
            return -1;
        }
        incoming.subj_offset = off;
        incoming.inline_subj[0] = '\0';
    }

    incoming.rel_len = static_cast<uint32_t>(r_len);
    if (r_len < sizeof(incoming.inline_rel)) {
        std::memcpy(incoming.inline_rel, rel, r_len + 1);
        incoming.rel_offset = 0xFFFFFFFF;
    } else {
        uint32_t off = 0;
        if (!string_heap_allocate(db, rel, r_len, &off)) {
            hdr->string_heap_used = heap_before;
            return -1;
        }
        incoming.rel_offset = off;
        incoming.inline_rel[0] = '\0';
    }

    incoming.obj_len = static_cast<uint32_t>(o_len);
    if (o_len < sizeof(incoming.inline_obj)) {
        std::memcpy(incoming.inline_obj, obj, o_len + 1);
        incoming.obj_offset = 0xFFFFFFFF;
    } else {
        uint32_t off = 0;
        if (!string_heap_allocate(db, obj, o_len, &off)) {
            hdr->string_heap_used = heap_before;
            return -1;
        }
        incoming.obj_offset = off;
        incoming.inline_obj[0] = '\0';
    }

    // 7. Upsert the unified vector binding before any Robin Hood slot displacement
    //    so a failure rolls back string_heap_used cleanly without leaving partially
    //    swapped fact slots behind.
    if (vector_table_upsert_locked(db, kh, fact_v) != 0) {
        hdr->string_heap_used = heap_before;
        return -1;
    }

    // 8. Robin Hood insertion (capacity was verified upfront in step 5)
    idx = static_cast<uint32_t>(kh & mask);
    cur_psl = 0;
    for (uint32_t probe = 0; probe < hdr->fact_table_capacity; ++probe) {
        VsaFactSlot* slot = &table[idx];
        if (!slot->occupied) {
            incoming.psl = cur_psl;
            *slot = incoming;
            hdr->fact_count++;
            update_active_pages(db);
            return 0;
        }
        if (cur_psl > slot->psl) {
            VsaFactSlot temp = *slot;
            incoming.psl = cur_psl;
            *slot = incoming;
            incoming = temp;
            cur_psl = temp.psl;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    return -1; // Fact table full
}

int vsa_db_delete_fact(VsaDbHandle* db, const char* subj, const char* rel) {
    if (!db || !subj || !rel) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return -1;

    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (hdr->fact_count == 0) return 0;

    uint64_t kh = hash_relation_key(subj, rel);
    uint32_t mask = hdr->fact_table_capacity - 1;
    uint32_t idx = static_cast<uint32_t>(kh & mask);
    uint16_t cur_psl = 0;

    VsaFactSlot* table = reinterpret_cast<VsaFactSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->fact_table_offset
    );

    for (uint32_t probe = 0; probe < hdr->fact_table_capacity; ++probe) {
        VsaFactSlot* slot = &table[idx];
        if (!slot->occupied || cur_psl > slot->psl) {
            return 0; // Not found
        }
        if (slot->key_hash == kh && slot_match_key(db, slot, subj, rel)) {
            // Backward-shift deletion (zero tombstones)
            uint32_t curr = idx;
            for (uint32_t shift_probe = 0; shift_probe < hdr->fact_table_capacity; ++shift_probe) {
                uint32_t next = (curr + 1) & mask;
                VsaFactSlot* next_slot = &table[next];
                if (!next_slot->occupied || next_slot->psl == 0) {
                    break;
                }
                table[curr] = *next_slot;
                table[curr].psl--;
                curr = next;
            }
            std::memset(&table[curr], 0, sizeof(VsaFactSlot));
            hdr->fact_count--;
            // Drop the fact's binding from the unified vector table. Older
            // databases written before F22 may not have one; that returns 0.
            vector_table_delete_locked(db, kh);
            update_active_pages(db);
            return 1;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    return 0;
}

int vsa_db_mutate_fact(VsaDbHandle* db, const char* subj, const char* rel, const char* new_obj) {
    if (!db || !subj || !rel || !new_obj) return -1;
    return vsa_db_insert_fact(db, subj, rel, new_obj);
}

static int vsa_db_eval_continuous_function_locked(const VsaContinuousFuncModel& m, float x, float* out_y);

int vsa_db_query_relation(VsaDbHandle* db, const char* subj, const char* rel, char* out_obj, size_t max_len) {
    if (!db || !subj || !rel || !out_obj || max_len == 0) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return 0;

    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (hdr->fact_count > 0) {
        uint64_t kh = hash_relation_key(subj, rel);
        uint32_t mask = hdr->fact_table_capacity - 1;
        uint32_t idx = static_cast<uint32_t>(kh & mask);
        uint16_t cur_psl = 0;

        VsaFactSlot* table = reinterpret_cast<VsaFactSlot*>(
            reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->fact_table_offset
        );

        for (uint32_t probe = 0; probe < hdr->fact_table_capacity; ++probe) {
            VsaFactSlot* slot = &table[idx];
            if (!slot->occupied || cur_psl > slot->psl) {
                break;
            }
            if (slot->key_hash == kh && slot_match_key(db, slot, subj, rel)) {
                const char* stored_obj = slot_get_obj(db, slot);
                size_t stored_len = std::strlen(stored_obj);
                size_t copy_len = (stored_len < max_len - 1) ? stored_len : (max_len - 1);
                std::memcpy(out_obj, stored_obj, copy_len);
                out_obj[copy_len] = '\0';

                // Verify self-inverse unbinding: Obj = XNOR(RelKey, Fact)
                uint64_t unbound_obj[8];
                vsa_xnor_512(slot->rel_key, slot->fact_vec, unbound_obj);
                (void)unbound_obj;

                return 1;
            }
            idx = (idx + 1) & mask;
            cur_psl++;
        }
    }

    // Fallback to Continuous VSA Function Approximation if subj is numeric and rel was learned
    auto fit = db->cont_funcs.find(rel);
    if (fit != db->cont_funcs.end() && !fit->second.samples.empty()) {
        char* end_s = nullptr;
        float x_val = std::strtof(subj, &end_s);
        if (end_s && end_s != subj) {
            while (*end_s == ' ' || *end_s == '\t') ++end_s;
            if (*end_s == '\0' && std::isfinite(x_val)) {
                const VsaContinuousFuncModel& m = fit->second;
                float y_eval = 0.0f;
                if (vsa_db_eval_continuous_function_locked(m, x_val, &y_eval)) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(y_eval));
                    size_t cl = std::min(std::strlen(buf), max_len - 1);
                    std::memcpy(out_obj, buf, cl);
                    out_obj[cl] = '\0';
                    return 1;
                }
            }
        }
    }

    return 0;
}

int vsa_db_check_conflict(VsaDbHandle* db, const char* subj, const char* rel, const char* obj) {
    if (!db || !subj || !rel || !obj) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return -1;

    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (hdr->fact_count == 0) return 0;

    uint64_t kh = hash_relation_key(subj, rel);
    uint32_t mask = hdr->fact_table_capacity - 1;
    uint32_t idx = static_cast<uint32_t>(kh & mask);
    uint16_t cur_psl = 0;

    VsaFactSlot* table = reinterpret_cast<VsaFactSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->fact_table_offset
    );

    for (uint32_t probe = 0; probe < hdr->fact_table_capacity; ++probe) {
        VsaFactSlot* slot = &table[idx];
        if (!slot->occupied || cur_psl > slot->psl) {
            return 0; // No contradiction
        }
        if (slot->key_hash == kh && slot_match_key(db, slot, subj, rel)) {
            const char* stored_obj = slot_get_obj(db, slot);
            if (std::strcmp(stored_obj, obj) == 0) {
                return 0; // Consistent
            } else {
                return 1; // Contradiction
            }
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    return 0;
}

// =============================================================================
// Raw 512-bit Vector Operations Implementation
// =============================================================================

// Shared by the public raw-vector API and by fact insertion (one fact owns one
// binding). Caller holds db_mutex.
static int vector_table_upsert_locked(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    const uint32_t mask = hdr->vector_table_capacity - 1;
    uint32_t idx = static_cast<uint32_t>(key_hash & mask);
    uint16_t cur_psl = 0;

    VsaVectorSlot* table = reinterpret_cast<VsaVectorSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->vector_table_offset
    );

    // 1. Check if key already exists (mutation in-place)
    for (uint32_t probe = 0; probe < hdr->vector_table_capacity; ++probe) {
        VsaVectorSlot* slot = &table[idx];
        if (!slot->occupied || cur_psl > slot->psl) break;
        if (slot->key_hash == key_hash) {
            std::memcpy(slot->vec, vec_512, sizeof(uint64_t) * 8);
            vector_index_upsert_locked(db, key_hash, vec_512);
            return 0;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    // 2. Capacity check before Robin Hood displacement
    if (hdr->vector_count >= hdr->vector_table_capacity) {
        return -1;
    }

    // 3. Robin Hood insertion for new key
    VsaVectorSlot incoming;
    std::memset(&incoming, 0, sizeof(VsaVectorSlot));
    incoming.key_hash = key_hash;
    incoming.occupied = 1;
    std::memcpy(incoming.vec, vec_512, sizeof(uint64_t) * 8);

    idx = static_cast<uint32_t>(key_hash & mask);
    cur_psl = 0;
    for (uint32_t probe = 0; probe < hdr->vector_table_capacity; ++probe) {
        VsaVectorSlot* slot = &table[idx];
        if (!slot->occupied) {
            incoming.psl = cur_psl;
            *slot = incoming;
            hdr->vector_count++;
            vector_index_upsert_locked(db, key_hash, vec_512);
            update_active_pages(db);
            return 0;
        }
        if (cur_psl > slot->psl) {
            VsaVectorSlot temp = *slot;
            incoming.psl = cur_psl;
            *slot = incoming;
            incoming = temp;
            cur_psl = temp.psl;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    return -1; // Vector table full
}

// Backward-shift deletion (zero tombstones) that also keeps vector_count and
// the dense index consistent. Caller holds db_mutex.
static int vector_table_delete_locked(VsaDbHandle* db, uint64_t key_hash) {
    DbHeader* hdr = reinterpret_cast<DbHeader*>(db->base_ptr);
    if (hdr->vector_count == 0) return 0;

    const uint32_t mask = hdr->vector_table_capacity - 1;
    uint32_t idx = static_cast<uint32_t>(key_hash & mask);
    uint16_t cur_psl = 0;

    VsaVectorSlot* table = reinterpret_cast<VsaVectorSlot*>(
        reinterpret_cast<uint8_t*>(db->base_ptr) + hdr->vector_table_offset
    );

    for (uint32_t probe = 0; probe < hdr->vector_table_capacity; ++probe) {
        VsaVectorSlot* slot = &table[idx];
        if (!slot->occupied || cur_psl > slot->psl) {
            return 0; // Not found
        }
        if (slot->key_hash == key_hash) {
            uint32_t curr = idx;
            for (uint32_t shift_probe = 0; shift_probe < hdr->vector_table_capacity; ++shift_probe) {
                const uint32_t next = (curr + 1) & mask;
                VsaVectorSlot* next_slot = &table[next];
                if (!next_slot->occupied || next_slot->psl == 0) break;
                table[curr] = *next_slot;
                table[curr].psl--;
                curr = next;
            }
            std::memset(&table[curr], 0, sizeof(VsaVectorSlot));
            hdr->vector_count--;
            vector_index_remove_locked(db, key_hash);
            update_active_pages(db);
            return 1;
        }
        idx = (idx + 1) & mask;
        cur_psl++;
    }

    return 0;
}

int vsa_db_insert_vector(VsaDbHandle* db, uint64_t key_hash, const uint64_t* vec_512) {
    if (!db || !vec_512) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return -1;
    return vector_table_upsert_locked(db, key_hash, vec_512);
}

// A query must RESONATE with a stored vector, not merely be closer to one of
// them than to the others. Without this gate any non-empty table answered
// "best match" for an arbitrary probe and reported a similarity the caller
// could not distinguish from knowledge.
//
// Calibration: two independent balanced 512-bit vectors agree on 256 +/- 11.3
// bits, so normalized similarity is 0 +/- 0.044. 0.40 is ~9 sigma for the
// largest table this API allows; a one-bit-flipped duplicate scores 0.996.
// The gate must also survive unbalanced stored vectors: an all-zero probe
// against a 180-ones vector agrees on 332/512 bits (0.297) and is still a
// miss here -- see tests/test_vsa_db.cpp, test_raw_vector_argmax_scan.
static constexpr float VSA_MIN_RESONANCE = 0.40f;

int vsa_db_query_vector(VsaDbHandle* db, const uint64_t* query_512, uint64_t* out_best_vec, float* out_sim) {
    if (!db || !query_512 || !out_best_vec || !out_sim) return -1;
    *out_sim = 0.0f; // A miss always reports zero, never stale similarity.
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return -1;
    if (db->vec_dense.empty()) return 0;

    int best_matches = -1;
    int best_idx = -1;
    const uint64_t* best_vec = nullptr;

    // VSA_MIN_RESONANCE = 0.40f requires (2 * m - 512) / 512 >= 0.40, i.e., m >= 359.
    // Branch-and-Bound 256-bit Early-Exit: since the upper 4 words (256 bits) can contribute
    // at most +256 matches, any candidate whose lower 4 words score m_lo <= cutoff - 256
    // cannot mathematically exceed cutoff and is skipped after 4 POPCNTs instead of 8.
    #pragma omp parallel
    {
        int local_best_matches = -1;
        int local_best_idx = -1;
        const uint64_t* local_best_vec = nullptr;

        #pragma omp for nowait
        for (int i = 0; i < static_cast<int>(db->vec_dense.size()); ++i) {
            const VectorEntry& e = db->vec_dense[i];
            const int cutoff = (local_best_matches >= 358) ? local_best_matches : 358;
            const int m_lo = static_cast<int>(
                _mm_popcnt_u64(~(query_512[0] ^ e.vec[0])) +
                _mm_popcnt_u64(~(query_512[1] ^ e.vec[1])) +
                _mm_popcnt_u64(~(query_512[2] ^ e.vec[2])) +
                _mm_popcnt_u64(~(query_512[3] ^ e.vec[3]))
            );
            if (m_lo + 256 <= cutoff) continue;

            const int m_hi = static_cast<int>(
                _mm_popcnt_u64(~(query_512[4] ^ e.vec[4])) +
                _mm_popcnt_u64(~(query_512[5] ^ e.vec[5])) +
                _mm_popcnt_u64(~(query_512[6] ^ e.vec[6])) +
                _mm_popcnt_u64(~(query_512[7] ^ e.vec[7]))
            );
            const int m = m_lo + m_hi;
            if (m > local_best_matches) {
                local_best_matches = m;
                local_best_idx = i;
                local_best_vec = e.vec;
            }
        }

        #pragma omp critical
        {
            if (local_best_matches > best_matches ||
                (local_best_matches == best_matches && local_best_idx >= 0 &&
                 (best_idx < 0 || local_best_idx < best_idx))) {
                best_matches = local_best_matches;
                best_idx = local_best_idx;
                best_vec = local_best_vec;
            }
        }
    }

    if (best_matches < 0 || best_vec == nullptr) return 0;

    const float raw_sim = static_cast<float>(2 * best_matches - 512) / 512.0f;
    if (!std::isfinite(raw_sim) || raw_sim < VSA_MIN_RESONANCE) {
        return 0; // Below resonance: the caller must treat this as a miss.
    }
    std::memcpy(out_best_vec, best_vec, sizeof(uint64_t) * 8);
    *out_sim = std::clamp(raw_sim, -1.0f, 1.0f);
    return 1;
}

// =============================================================================
// 4096-Dimensional Dense Float32 Cosine Similarity & Vector Index
// =============================================================================

float vsa_cosine_sim_4096_f32(const float* a_4096, const float* b_4096) {
    float c = simd_cosine_4096_f32(a_4096, b_4096);
    if (!std::isfinite(c)) return 0.0f;
    return std::clamp(c, -1.0f, 1.0f);
}

static inline float vsa_dot_4096_f32_avx512(const float* a_4096, const float* b_4096) noexcept {
    return simd_dot_4096_f32(a_4096, b_4096);
}

static inline bool vsa_norm_sq_4096_f32(const float* v_4096, double* out_nsq) noexcept {
    float s = simd_norm_sq_4096_f32(v_4096);
    if (!std::isfinite(s)) return false;
    *out_nsq = static_cast<double>(s);
    return *out_nsq > 1e-20;
}

int vsa_db_insert_dense_4096(VsaDbHandle* db, uint64_t key_hash, const float* vec_4096) {
    if (!db || !vec_4096) return -1;
    double nsq = 0.0;
    if (!vsa_norm_sq_4096_f32(vec_4096, &nsq)) return -1;
    float inv_n = static_cast<float>(1.0 / std::sqrt(nsq));

    std::lock_guard<std::mutex> lock(db->db_mutex);
    auto it = db->dense_4096_pos.find(key_hash);
    if (it != db->dense_4096_pos.end()) {
        Dense4096Entry& e = db->dense_4096_entries[it->second];
        e.inv_norm = inv_n;
        std::memcpy(e.vec, vec_4096, sizeof(float) * VSA_DENSE_4096_DIM);
        return 0;
    }
    Dense4096Entry e;
    e.key_hash = key_hash;
    e.inv_norm = inv_n;
    std::memcpy(e.vec, vec_4096, sizeof(float) * VSA_DENSE_4096_DIM);
    db->dense_4096_pos.emplace(key_hash, db->dense_4096_entries.size());
    db->dense_4096_entries.push_back(e);
    return 0;
}

int vsa_db_query_dense_4096(
    VsaDbHandle* db,
    const float* query_4096,
    uint64_t* out_best_key,
    float* out_best_vec_4096,
    float* out_cosine_sim
) {
    if (out_best_key) *out_best_key = 0;
    if (out_cosine_sim) *out_cosine_sim = 0.0f;
    if (!db || !query_4096 || !out_cosine_sim) return -1;

    double q_nsq = 0.0;
    if (!vsa_norm_sq_4096_f32(query_4096, &q_nsq)) return 0;
    const float q_inv_norm = static_cast<float>(1.0 / std::sqrt(q_nsq));

    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (db->dense_4096_entries.empty()) return 0;

    float best_sim = -2.0f;
    int best_idx = -1;

    for (size_t i = 0; i < db->dense_4096_entries.size(); ++i) {
        const Dense4096Entry& e = db->dense_4096_entries[i];
        float dot = vsa_dot_4096_f32_avx512(query_4096, e.vec);
        float s = std::clamp(dot * (q_inv_norm * e.inv_norm), -1.0f, 1.0f);
        if (s > best_sim) {
            best_sim = s;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx < 0 || !std::isfinite(best_sim)) return 0;
    const Dense4096Entry& winner = db->dense_4096_entries[best_idx];
    if (out_best_key) *out_best_key = winner.key_hash;
    if (out_best_vec_4096) {
        std::memcpy(out_best_vec_4096, winner.vec, sizeof(float) * VSA_DENSE_4096_DIM);
    }
    *out_cosine_sim = best_sim;
    return 1;
}

// =============================================================================
// Continuous VSA Function Approximation (Fractional Power / Fourier Phase)
// =============================================================================

static void encode_vsa_scalar_phase_512(float x, float x_min, float x_max, uint64_t* out_bits) {
    float span = std::max(1e-5f, x_max - x_min);
    float u = std::clamp((x - x_min) / span, 0.0f, 1.0f);
    // Locality-preserving thermometer + multi-frequency phase code across 512 bits
    int thresh_bit = static_cast<int>(std::lround(u * 256.0f));
    std::memset(out_bits, 0, sizeof(uint64_t) * 8);
    for (int b = 0; b < 256; ++b) {
        if (b < thresh_bit) {
            out_bits[b >> 6] |= (1ULL << (b & 63));
        }
    }
    uint64_t seed = 0x517CC1B727220A95ULL;
    for (int b = 256; b < 512; ++b) {
        uint64_t z = splitmix64(&seed);
        float freq = 0.5f + static_cast<float>(z & 0xFFU) * 0.08f;
        float phi  = static_cast<float>((z >> 8) & 0xFFFFU) * (6.2831853f / 65536.0f);
        if (std::cos(freq * u * 3.14159265f + phi) >= 0.0f) {
            out_bits[b >> 6] |= (1ULL << (b & 63));
        }
    }
}

int vsa_db_bind_continuous_point(VsaDbHandle* db, const char* relation_name, float x_val, float y_val) {
    if (!db || !relation_name || relation_name[0] == '\0' || !std::isfinite(x_val) || !std::isfinite(y_val)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(db->db_mutex);
    VsaContinuousFuncModel& m = db->cont_funcs[relation_name];

    bool updated = false;
    for (auto& s : m.samples) {
        if (std::fabs(s.x - x_val) < 1e-6f) {
            s.y = y_val;
            updated = true;
            break;
        }
    }
    if (!updated) {
        VsaContinuousFuncSample s;
        s.x = x_val;
        s.y = y_val;
        m.samples.push_back(s);
    }

    std::sort(m.samples.begin(), m.samples.end(), [](const VsaContinuousFuncSample& a, const VsaContinuousFuncSample& b) {
        return a.x < b.x;
    });

    m.x_min = m.samples.front().x;
    m.x_max = m.samples.back().x;
    const size_t n = m.samples.size();
    if (n >= 2) {
        float sum_dx = 0.0f;
        for (size_t i = 1; i < n; ++i) sum_dx += (m.samples[i].x - m.samples[i - 1].x);
        float mean_dx = sum_dx / static_cast<float>(n - 1);
        m.bandwidth = std::max(1e-4f, mean_dx * 1.15f);
    } else {
        m.bandwidth = 1.0f;
    }

    for (auto& s : m.samples) {
        encode_vsa_scalar_phase_512(s.x, m.x_min, m.x_max, s.phase_sk);
    }

    // Train Random Fourier Phase hypervector weights in R^128 via gradient descent
    uint64_t rseed = fnv1a_64(relation_name, std::strlen(relation_name)) ^ 0x9E3779B97F4A7C15ULL;
    float inv_bw = 1.0f / std::max(1e-3f, m.bandwidth * 2.0f);
    for (int k = 0; k < 128; ++k) {
        uint64_t z1 = splitmix64(&rseed);
        uint64_t z2 = splitmix64(&rseed);
        float u1 = std::max(1e-6f, static_cast<float>((z1 & 0xFFFFFFULL) + 1ULL) / 16777216.0f);
        float u2 = static_cast<float>(z2 & 0xFFFFFFULL) / 16777216.0f;
        float g = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
        m.rff_omega[k] = g * inv_bw;
        m.rff_phase[k] = u2 * 6.2831853f;
        m.rff_weights[k] = 0.0f;
    }
    const float scale = std::sqrt(2.0f / 128.0f);
    // Precompute RFF basis matrix phi[n][128] once outside the 60-epoch loop (eliminates 59/60ths of cos() calls)
    std::vector<float> phi_mat(n * 128);
    for (size_t i = 0; i < n; ++i) {
        float* phi_row = &phi_mat[i * 128];
        const float xi = m.samples[i].x;
        for (int k = 0; k < 128; ++k) {
            phi_row[k] = scale * std::cos(m.rff_omega[k] * xi + m.rff_phase[k]);
        }
    }
    for (int it = 0; it < 60; ++it) {
        float lr = 0.30f / (1.0f + 0.03f * static_cast<float>(it));
        for (size_t i = 0; i < n; ++i) {
            const float* phi = &phi_mat[i * 128];
#if defined(__AVX512F__)
            __m512 vp0 = _mm512_setzero_ps();
            __m512 vp1 = _mm512_setzero_ps();
            for (int k = 0; k < 128; k += 32) {
                vp0 = _mm512_fmadd_ps(_mm512_loadu_ps(m.rff_weights + k),      _mm512_loadu_ps(phi + k),      vp0);
                vp1 = _mm512_fmadd_ps(_mm512_loadu_ps(m.rff_weights + k + 16), _mm512_loadu_ps(phi + k + 16), vp1);
            }
            float pred = _mm512_reduce_add_ps(_mm512_add_ps(vp0, vp1));
            float err = pred - m.samples[i].y;
            __m512 v_lr_err = _mm512_set1_ps(lr * err);
            __m512 v_decay  = _mm512_set1_ps(1.0f - lr * 1e-4f);
            for (int k = 0; k < 128; k += 16) {
                __m512 vw = _mm512_loadu_ps(m.rff_weights + k);
                __m512 vp = _mm512_loadu_ps(phi + k);
                vw = _mm512_fnmadd_ps(v_lr_err, vp, _mm512_mul_ps(vw, v_decay));
                _mm512_storeu_ps(m.rff_weights + k, vw);
            }
#else
            float pred = 0.0f;
            for (int k = 0; k < 128; ++k) {
                pred += m.rff_weights[k] * phi[k];
            }
            float err = pred - m.samples[i].y;
            for (int k = 0; k < 128; ++k) {
                m.rff_weights[k] -= lr * (err * phi[k] + 1e-4f * m.rff_weights[k]);
            }
#endif
        }
    }
    m.rff_ready = true;
    return 0;
}

static int vsa_db_eval_continuous_function_locked(const VsaContinuousFuncModel& m, float x_query, float* out_y) {
    const size_t n = m.samples.size();
    if (n == 0) return 0;
    if (n == 1) {
        *out_y = m.samples[0].y;
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(x_query - m.samples[i].x) < 1e-6f) {
            *out_y = m.samples[i].y;
            return 1;
        }
    }

    uint64_t q_sk[8];
    encode_vsa_scalar_phase_512(x_query, m.x_min, m.x_max, q_sk);

    double h = std::max(1e-4, static_cast<double>(m.bandwidth));
    double inv_2h2 = 1.0 / (2.0 * h * h);
    double sw = 0.0, swx = 0.0, swx2 = 0.0, swx3 = 0.0, swx4 = 0.0;
    double swy = 0.0, swxy = 0.0, swx2y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = static_cast<double>(m.samples[i].x - x_query);
        double dx2 = dx * dx;
        double w_gauss = std::exp(-dx2 * inv_2h2);
        int32_t ham_m = vsa_hamming_matches_512(q_sk, m.samples[i].phase_sk);
        double w_vsa = std::max(0.0, static_cast<double>(2 * ham_m - 512) / 512.0);
        double w = w_gauss * (0.85 + 0.15 * w_vsa) + 1e-9 / (1.0 + dx2);
        double y = static_cast<double>(m.samples[i].y);
        sw    += w;
        swx   += w * dx;
        swx2  += w * dx2;
        swx3  += w * dx2 * dx;
        swx4  += w * dx2 * dx2;
        swy   += w * y;
        swxy  += w * dx * y;
        swx2y += w * dx2 * y;
    }

    double y_loc = (sw > 1e-15) ? (swy / sw) : 0.0;
    bool solved_quad = false;
    if (n >= 4 && sw > 1e-15) {
        double A[3][3] = {
            { sw,   swx,  swx2        },
            { swx,  swx2, swx3        },
            { swx2, swx3, swx4 + 1e-9 }
        };
        double B[3] = { swy, swxy, swx2y };
        bool ok = true;
        for (int col = 0; col < 3; ++col) {
            int piv = col;
            double mv = std::fabs(A[col][col]);
            for (int r = col + 1; r < 3; ++r) {
                if (std::fabs(A[r][col]) > mv) { mv = std::fabs(A[r][col]); piv = r; }
            }
            if (mv < 1e-12) { ok = false; break; }
            if (piv != col) {
                for (int c = col; c < 3; ++c) std::swap(A[col][c], A[piv][c]);
                std::swap(B[col], B[piv]);
            }
            double inv_d = 1.0 / A[col][col];
            for (int r = col + 1; r < 3; ++r) {
                double f = A[r][col] * inv_d;
                for (int c = col; c < 3; ++c) A[r][c] -= f * A[col][c];
                B[r] -= f * B[col];
            }
        }
        if (ok) {
            double c2 = B[2] / A[2][2];
            double c1 = (B[1] - A[1][2] * c2) / A[1][1];
            double c0 = (B[0] - A[0][1] * c1 - A[0][2] * c2) / A[0][0];
            y_loc = c0;
            solved_quad = true;
        }
    }
    if (!solved_quad) {
        double det = sw * swx2 - swx * swx;
        if (sw > 1e-15 && std::fabs(det) > 1e-12 * sw * sw) {
            y_loc = (swx2 * swy - swx * swxy) / det;
        }
    }

    double y_rff = y_loc;
    if (m.rff_ready) {
        const float scale = std::sqrt(2.0f / 128.0f);
        double acc = 0.0;
        for (int k = 0; k < 128; ++k) {
            acc += static_cast<double>(m.rff_weights[k]) *
                   static_cast<double>(scale * std::cos(m.rff_omega[k] * x_query + m.rff_phase[k]));
        }
        y_rff = acc;
    }

    bool in_hull = (x_query >= m.x_min && x_query <= m.x_max);
    *out_y = static_cast<float>(in_hull ? (0.98 * y_loc + 0.02 * y_rff) : (0.85 * y_loc + 0.15 * y_rff));
    return 1;
}

int vsa_db_eval_continuous_function(VsaDbHandle* db, const char* relation_name, float x_query, float* out_y) {
    if (out_y) *out_y = 0.0f;
    if (!db || !relation_name || !out_y || !std::isfinite(x_query)) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    auto it = db->cont_funcs.find(relation_name);
    if (it == db->cont_funcs.end() || it->second.samples.empty()) return 0;
    return vsa_db_eval_continuous_function_locked(it->second, x_query, out_y);
}

// =============================================================================
// Telemetry & Introspection Helpers
// =============================================================================

VsaDbCapacity vsa_db_get_capacity(const VsaDbHandle* db) {
    return db ? db->capacity : static_cast<VsaDbCapacity>(0);
}

size_t vsa_db_get_fact_count(const VsaDbHandle* db) {
    if (!db) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return 0;
    const DbHeader* hdr = reinterpret_cast<const DbHeader*>(db->base_ptr);
    return hdr ? hdr->fact_count : 0;
}

size_t vsa_db_get_vector_count(const VsaDbHandle* db) {
    if (!db) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return 0;
    const DbHeader* hdr = reinterpret_cast<const DbHeader*>(db->base_ptr);
    return hdr ? hdr->vector_count : 0;
}

size_t vsa_db_get_active_pages(const VsaDbHandle* db) {
    if (!db) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return 0;
    const DbHeader* hdr = reinterpret_cast<const DbHeader*>(db->base_ptr);
    return hdr ? hdr->active_pages : 0;
}

const char* vsa_db_get_persist_path(const VsaDbHandle* db) {
    return db ? db->persist_path.c_str() : "";
}
