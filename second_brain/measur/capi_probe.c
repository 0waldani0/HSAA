// The C-ABI integration probe (a tracked regression step, run by
// measur/regress_par.py): exercises the shipped C interface exactly the way
// an external integrator would and pins the contract documented in
// second_brain/popcnt_brain.h after F22:
//
//   1. exact raw-string facts always resolve;
//   2. case/whitespace variants resolve through the lexical fallback;
//   3. a one-character typo in the subject resolves at >=99% (the calibrated
//      200/512 dot gate);
//   4. a one-character typo in a short predicate resolves at the measured
//      (lower) relation rate and never yields a wrong relation;
//   5. unrelated subjects and exact-subject+bogus-relation queries return 0;
//   6. brain_check_conflict follows the same exact-then-associative path.
//
// Any violation exits non-zero. The threshold itself is calibrated by
// measur/assoc_calib.cpp, never guessed here.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "popcnt_brain.h"

#define N_FACTS 300
#define TYPO_RECALL_MIN 297        // clean measurement is 300/300 (template population)
#define REL_TYPO_RECALL_MIN 240    // calibration: 240/280 = 85.7% at dot >= 200

static uint64_t g_state = 0x123456789ABCDEFULL;

static uint64_t next_rand(void) {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

static void pseudo_word(char* out, int len) {
    for (int i = 0; i < len; ++i) out[i] = (char)('a' + (int)(next_rand() % 26));
    out[len] = '\0';
}

static void make_subject(char* out, size_t cap) {
    char w[16];
    pseudo_word(w, 8);
    snprintf(out, cap, "entity_%s", w);
}

static void make_object(char* out, size_t cap) {
    char w[16];
    pseudo_word(w, 8);
    snprintf(out, cap, "object_%s", w);
}

// One-character typo at a fixed position inside the distinguishing suffix.
static void typo_subject(const char* src, char* out, size_t cap) {
    snprintf(out, cap, "%s", src);
    const size_t i = 10;  // "entity_" is 7 bytes; 10 is inside the 8-char word
    if (i < strlen(out)) out[i] = (out[i] == 'z') ? 'a' : (char)(out[i] + 1);
}

static void typo_relation(const char* src, char* out, size_t cap) {
    snprintf(out, cap, "%s", src);
    const size_t i = 2;
    if (i < strlen(out)) out[i] = (out[i] == 'z') ? 'a' : (char)(out[i] + 1);
}

static void upper_ascii(const char* src, char* out, size_t cap) {
    snprintf(out, cap, "%s", src);
    for (size_t i = 0; out[i]; ++i) {
        if (out[i] >= 'a' && out[i] <= 'z') out[i] = (char)(out[i] - 'a' + 'A');
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   // redirect-safe: a crash must not eat the trace

    static const char* kRelations[7] = {
        "born_in", "lives_in", "works_at", "capital_of", "member_of", "located_in", "part_of"
    };
    static char subjects[N_FACTS][32];
    static char relations[N_FACTS][16];
    static char objects[N_FACTS][32];

    printf("[trace] before brain_create\n");
    PopCountBrain* b = brain_create(4096, "build/api_probe.bin");
    if (!b) { printf("create FAILED\n"); return 1; }
    printf("[trace] brain created\n");

    size_t pages0 = 0, rules0 = 0;
    brain_get_stats(b, &pages0, &rules0);
    printf("after create        : active_pages=%zu active_rules=%zu\n", pages0, rules0);

    for (int i = 0; i < N_FACTS; ++i) {
        make_subject(subjects[i], sizeof(subjects[i]));
        snprintf(relations[i], sizeof(relations[i]), "%s", kRelations[i % 7]);
        make_object(objects[i], sizeof(objects[i]));
        if (brain_store_fact(b, subjects[i], relations[i], objects[i]) != 1) {
            printf("store %d FAILED\n", i);
            return 2;
        }
    }
    size_t pages1 = 0, rules1 = 0;
    brain_get_stats(b, &pages1, &rules1);
    printf("after %d stores     : active_pages=%zu active_rules=%zu\n", N_FACTS, pages1, rules1);
    if (rules1 != (size_t)N_FACTS) { printf("FAIL: active_rules=%zu expected %d\n", rules1, N_FACTS); return 3; }

    for (int i = 0; i < 50; ++i) {
        brain_store_fact(b, subjects[1], relations[1], objects[1]);
    }
    size_t pages2 = 0, rules2 = 0;
    brain_get_stats(b, &pages2, &rules2);
    printf("after 50 dup stores : active_pages=%zu active_rules=%zu (calls, not distinct)\n", pages2, rules2);
    if (rules2 != (size_t)N_FACTS + 50) { printf("FAIL: duplicate accounting changed\n"); return 3; }

    int failures = 0;
    char ans[256];

    // 1. exact hits -- every stored fact must come back verbatim.
    int exact_ok = 0;
    for (int i = 0; i < N_FACTS; ++i) {
        ans[0] = '\0';
        const int rc = brain_query_relation(b, subjects[i], relations[i], ans, sizeof(ans));
        if (rc == 1 && strcmp(ans, objects[i]) == 0) ++exact_ok;
    }
    printf("exact        : %d/%d correct\n", exact_ok, N_FACTS);
    if (exact_ok != N_FACTS) ++failures;

    // 2. case variants -- normalization makes these exact in sketch space.
    int case_ok = 0;
    for (int i = 0; i < N_FACTS; ++i) {
        char upper[32];
        upper_ascii(subjects[i], upper, sizeof(upper));
        ans[0] = '\0';
        const int rc = brain_query_relation(b, upper, relations[i], ans, sizeof(ans));
        if (rc == 1 && strcmp(ans, objects[i]) == 0) ++case_ok;
    }
    printf("case         : %d/%d correct\n", case_ok, N_FACTS);
    if (case_ok != N_FACTS) ++failures;

    // 3. one-character subject typo, exact relation.
    int typo_ok = 0;
    for (int i = 0; i < N_FACTS; ++i) {
        char typo[32];
        typo_subject(subjects[i], typo, sizeof(typo));
        ans[0] = '\0';
        const int rc = brain_query_relation(b, typo, relations[i], ans, sizeof(ans));
        if (rc == 1 && strcmp(ans, objects[i]) == 0) ++typo_ok;
    }
    printf("typo_subject : %d/%d correct (>= %d required)\n", typo_ok, N_FACTS, TYPO_RECALL_MIN);
    if (typo_ok < TYPO_RECALL_MIN) ++failures;

    // 4. one-character predicate typo, exact subject. Each subject owns one
    //    fact here, so any hit is necessarily the right relation.
    int rel_ok = 0, rel_hits = 0;
    for (int i = 0; i < N_FACTS; ++i) {
        char typo[16];
        typo_relation(relations[i], typo, sizeof(typo));
        ans[0] = '\0';
        const int rc = brain_query_relation(b, subjects[i], typo, ans, sizeof(ans));
        if (rc == 1) {
            ++rel_hits;
            if (strcmp(ans, objects[i]) == 0) ++rel_ok;
        }
    }
    printf("typo_relation: %d/%d hits, %d correct (>= %d required, 0 wrong)\n",
           rel_hits, N_FACTS, rel_ok, REL_TYPO_RECALL_MIN);
    if (rel_hits < REL_TYPO_RECALL_MIN || rel_ok != rel_hits) ++failures;

    // 5. false-positive guards: unrelated subject, and exact subject with a
    //    predicate the store has never seen.
    int unrelated_hits = 0, bogus_rel_hits = 0;
    for (int i = 0; i < N_FACTS; ++i) {
        char word[16], foreign[32];
        pseudo_word(word, 10);
        snprintf(foreign, sizeof(foreign), "unrelated_%s", word);
        if (brain_query_relation(b, foreign, "no_such_relation", ans, sizeof(ans)) == 1) ++unrelated_hits;
        if (brain_query_relation(b, subjects[i], "no_such_relation", ans, sizeof(ans)) == 1) ++bogus_rel_hits;
    }
    printf("unrelated    : %d/%d false hits\n", unrelated_hits, N_FACTS);
    printf("bogus_rel    : %d/%d false hits\n", bogus_rel_hits, N_FACTS);
    if (unrelated_hits != 0 || bogus_rel_hits != 0) ++failures;

    // 6. conflict semantics on the same exact-then-associative path.
    int conflict_exact_ok = 0, conflict_fuzzy_resolved = 0, conflict_fuzzy_ok = 0;
    for (int i = 0; i < 10; ++i) {
        if (brain_check_conflict(b, subjects[i], relations[i], "different_object") == 1 &&
            brain_check_conflict(b, subjects[i], relations[i], objects[i]) == 0) {
            ++conflict_exact_ok;
        }
        char typo[32];
        typo_subject(subjects[i], typo, sizeof(typo));
        if (brain_query_relation(b, typo, relations[i], ans, sizeof(ans)) == 1 &&
            strcmp(ans, objects[i]) == 0) {
            ++conflict_fuzzy_resolved;
            if (brain_check_conflict(b, typo, relations[i], "different_object") == 1 &&
                brain_check_conflict(b, typo, relations[i], objects[i]) == 0) {
                ++conflict_fuzzy_ok;
            }
        }
    }
    printf("conflict     : exact %d/10, fuzzy resolved %d/10, fuzzy semantics %d/%d\n",
           conflict_exact_ok, conflict_fuzzy_resolved, conflict_fuzzy_ok, conflict_fuzzy_resolved);
    if (conflict_exact_ok != 10 || conflict_fuzzy_resolved < 9 ||
        conflict_fuzzy_ok != conflict_fuzzy_resolved) {
        ++failures;
    }
    if (brain_check_conflict(b, "unrelated_zzzzzzzzzz", "no_such_relation", "x") != 0) ++failures;

    brain_destroy(b);
    printf("RESULT: %s (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
