#include "vsa_db.h"
#include <filesystem>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    std::error_code ec;
    std::filesystem::create_directories("build", ec);

    const char* cyr_path = "build/тест_кириллицы_persist.bin";
    std::filesystem::remove(std::filesystem::u8path(cyr_path), ec);

#ifdef _WIN32
    const wchar_t* w_cyr_path = L"build\\тест_кириллицы_persist.bin";
    HANDLE hw = CreateFileW(w_cyr_path, GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD err_w = GetLastError();
    std::cout << "[TEST] Direct CreateFileW on w_cyr_path: Handle=" << hw << ", GetLastError=" << err_w << std::endl;
    if (hw != INVALID_HANDLE_VALUE) CloseHandle(hw);
    DeleteFileW(w_cyr_path);
#endif

    VsaDbHandle* db = vsa_db_create(VSA_CAP_MICRO, cyr_path);
    if (!db) {
        std::cerr << "[FAIL] db is null" << std::endl;
        return 1;
    }
    vsa_db_insert_fact(db, "CyrSubject", "CyrRelation", "CyrObject");
    vsa_db_flush(db);
    vsa_db_destroy(db);

    bool exists = std::filesystem::exists(std::filesystem::u8path(cyr_path), ec);
    std::cout << "[TEST] Cyrillic DB file exists on disk: " << (exists ? 1 : 0) << std::endl;
    if (!exists) {
        std::cerr << "[FAIL] Cyrillic DB file was not created on disk" << std::endl;
        return 1;
    }

    VsaDbHandle* db2 = vsa_db_create(VSA_CAP_MICRO, cyr_path);
    if (!db2) {
        std::cerr << "[FAIL] Failed to reopen DB on Cyrillic path" << std::endl;
        std::filesystem::remove(std::filesystem::u8path(cyr_path), ec);
        return 1;
    }
    char out_obj[64] = {0};
    int qrc = vsa_db_query_relation(db2, "CyrSubject", "CyrRelation", out_obj, sizeof(out_obj));
    vsa_db_destroy(db2);
    std::filesystem::remove(std::filesystem::u8path(cyr_path), ec);

    if (qrc != 1 || std::string(out_obj) != "CyrObject") {
        std::cerr << "[FAIL] Failed to retrieve fact from Cyrillic path DB" << std::endl;
        return 1;
    }
    std::cout << "[TEST] Successfully retrieved fact from Cyrillic path DB: " << out_obj << std::endl;

    return 0;
}
