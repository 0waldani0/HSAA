#include "vsa_db.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <omp.h>

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

// 512-bit bitwise XNOR
inline void vsa_xnor_512(const uint64_t* a, const uint64_t* b, uint64_t* out) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
    __m512i vxnor = _mm512_ternarylogic_epi64(va, vb, va, 0x99);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out), vxnor);
#else
    for (int w = 0; w < 8; ++w) {
        out[w] = ~(a[w] ^ b[w]);
    }
#endif
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
    db->mapping_handle = NULL;

    bool use_file = (persist_path != nullptr && persist_path[0] != '\0');
    if (use_file) {
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
    }

    if (!use_file) {
        // Pure in-memory allocation when no persist path was requested or it was unusable
        void* ptr = VirtualAlloc(NULL, total_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!ptr) {
            delete db;
            return nullptr;
        }
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
    }
    delete db;
}

int vsa_db_flush(VsaDbHandle* db) {
    if (!db) return -1;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (db->is_mapped_file && db->base_ptr && db->file_handle != INVALID_HANDLE_VALUE) {
        BOOL fv = FlushViewOfFile(db->base_ptr, db->total_bytes);
        BOOL fb = FlushFileBuffers(db->file_handle);
        return (fv && fb) ? 0 : -1;
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
    }
    vector_index_rebuild(clone);

    if (clone->is_mapped_file) {
        FlushViewOfFile(clone->base_ptr, clone->total_bytes);
        FlushFileBuffers(clone->file_handle);
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

int vsa_db_query_relation(VsaDbHandle* db, const char* subj, const char* rel, char* out_obj, size_t max_len) {
    if (!db || !subj || !rel || !out_obj || max_len == 0) return 0;
    std::lock_guard<std::mutex> lock(db->db_mutex);
    if (!db->base_ptr) return 0;

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
            return 0;
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
