#include "vsa_db.h"
#include <iostream>
#include <cassert>
#include <windows.h>

int main() {
    const char* cyr_path = "C:\\Users\\Администратор\\Documents\\project upgrade\\build\\test_cyr_persist.bin";
    
    HANDLE h = CreateFileA(cyr_path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD err = GetLastError();
    std::cout << "[TEST] Direct CreateFileA on cyr_path: Handle=" << h << ", GetLastError=" << err << std::endl;
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);

    const wchar_t* w_cyr_path = L"C:\\Users\\Администратор\\Documents\\project upgrade\\build\\test_cyr_persist.bin";
    HANDLE hw = CreateFileW(w_cyr_path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD err_w = GetLastError();
    std::cout << "[TEST] Direct CreateFileW on w_cyr_path: Handle=" << hw << ", GetLastError=" << err_w << std::endl;
    if (hw != INVALID_HANDLE_VALUE) CloseHandle(hw);

    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, cyr_path);
    if (!db) {
        std::cerr << "[FAIL] db is null" << std::endl;
        return 1;
    }
    vsa_db_insert_fact(db, "CyrSubject", "CyrRelation", "CyrObject");
    vsa_db_flush(db);
    vsa_db_destroy(db);

    // Check if the file was created on disk
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(cyr_path, &fd);
    std::cout << "[TEST] FindFirstFileA on cyr_path: Found=" << (hFind != INVALID_HANDLE_VALUE) << std::endl;
    if (hFind != INVALID_HANDLE_VALUE) FindClose(hFind);

    WIN32_FIND_DATAW fdw;
    HANDLE hFindW = FindFirstFileW(w_cyr_path, &fdw);
    std::cout << "[TEST] FindFirstFileW on w_cyr_path: Found=" << (hFindW != INVALID_HANDLE_VALUE) << std::endl;
    if (hFindW != INVALID_HANDLE_VALUE) FindClose(hFindW);

    // Check reopening and retrieving fact from disk
    VsaDbHandle* db2 = vsa_db_create(VSA_CAP_MICRO, cyr_path);
    if (!db2) {
        std::cerr << "[FAIL] Failed to reopen DB on Cyrillic path" << std::endl;
        return 1;
    }
    char out_obj[64] = {0};
    int qrc = vsa_db_query_relation(db2, "CyrSubject", "CyrRelation", out_obj, sizeof(out_obj));
    vsa_db_destroy(db2);
    if (qrc != 1 || std::string(out_obj) != "CyrObject") {
        std::cerr << "[FAIL] Failed to retrieve fact from Cyrillic path DB" << std::endl;
        return 1;
    }
    std::cout << "[TEST] Successfully retrieved fact from Cyrillic path DB: " << out_obj << std::endl;

    return 0;
}
