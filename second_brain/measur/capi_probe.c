// The C-ABI integration probe (a tracked regression step, run by
// measur/regress_par.py): exercises the shipped C interface exactly the way
// an external integrator would, and pins the query path's DOCUMENTED
// semantics: brain_query_relation answers exact strings only -- it reads the
// direct_facts dictionary and never the VSA bank (see the header before
// integrating). The case-differentiated miss below is the regression proof
// of that boundary.
#include <stdio.h>
#include <string.h>
#include "popcnt_brain.h"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   // redirect-safe: a crash must not eat the trace
    printf("[trace] before brain_create\n");
    PopCountBrain* b = brain_create(4096, "build/api_probe.bin");
    if (!b) { printf("create FAILED\n"); return 1; }
    printf("[trace] brain created\n");

    size_t pages = 0, rules = 0;
    brain_get_stats(b, &pages, &rules);
    printf("after create        : active_pages=%zu active_rules=%zu\n", pages, rules);

    const int N = 200;
    for (int i = 0; i < N; ++i) {
        printf("[trace] store #%d\n", i);
        char s[64], r[64], o[64];
        snprintf(s, sizeof(s), "entity_subject_%d", i);
        snprintf(r, sizeof(r), "relation_kind_%d", i % 7);
        snprintf(o, sizeof(o), "object_value_%d", i);
        if (brain_store_fact(b, s, r, o) != 1) { printf("store %d FAILED\n", i); return 2; }
    }
    brain_get_stats(b, &pages, &rules);
    printf("after %d stores     : active_pages=%zu active_rules=%zu\n", N, pages, rules);

    // repeated stores of the SAME fact: does active_rules double-count?
    for (int i = 0; i < 50; ++i) {
        brain_store_fact(b, "entity_subject_1", "relation_kind_1", "object_value_1");
    }
    brain_get_stats(b, &pages, &rules);
    printf("after 50 dup stores : active_pages=%zu active_rules=%zu\n", pages, rules);

    // the query path: exact strings only, documented in popcnt_brain.h
    char ans[256] = {0};
    int q = brain_query_relation(b, "entity_subject_1", "relation_kind_1", ans, sizeof(ans));
    printf("query exact hit=%d ans=%s\n", q, ans);
    int q2 = brain_query_relation(b, "entity_subject_1", "relation_kind_2", ans, sizeof(ans));
    printf("query miss        =%d ans=[%s]\n", q2, ans);
    int q3 = brain_query_relation(b, "ENTITY_SUBJECT_1", "relation_kind_1", ans, sizeof(ans));
    printf("query case-diff   =%d ans=[%s]  (0 => pure string, no fuzzy/HDC)\n", q3, ans);

    brain_destroy(b);
    printf("done\n");
    return 0;
}
