# Project Upgrade: Hybrid Neural & PopCount Architecture

Проект объединяет два специализированных вычислительных контура, оптимизированных под аппаратные возможности этой рабочей станции (Intel Core i7-11800H + NVIDIA RTX 3050 Laptop):

```
project upgrade/
├── include/            # Заголовки HSSA, VSA DB, ModelMod, Bridge и BenchRunner (C11/C++20 ABI)
├── src/                # Реализация HSSA (Gated DeltaNet + Hopfield), VSA DB, Bridge, CLI Lab
├── second_brain/       # PopCount VSA Ассоциативная Память и Генеративный Контур
│   ├── popcnt_brain.h  # Публичный чистый C-ABI интерфейс
│   ├── second_brain_cpp.hpp # C++20 RAII-обёртка (hsaa::SecondBrain / hssa::SecondBrain)
│   ├── popcnt_brain.cpp# Ядро движка (AVX-512 VPOPCNTDQ, 4KB Demand-Paged VSA, Causal OOV Priors)
│   ├── test_c_api.cpp  # Интеграционный C-ABI тест
│   └── measur/         # Нативные бенчмарки и калибровочные стенды
├── legacy_ptq/         # PTQ CUDA-движок 2-битного квантованного инференса (bsl_lut_gemv_sm86)
├── tests/              # Модульные, стресс- и E2E-тесты (M1–M3, LLM/Code, Tier-5 Adversarial)
├── docs/               # Архитектурная спецификация HSSA
├── build.bat           # Главный оркестратор сборки проекта (MSVC + NVCC)
├── CMakeLists.txt      # Корневой CMake-конфиг (Windows / Linux)
├── PROJECT.md          # Полная техническая карта модулей и инвариантов
└── BRIDGE.md           # Архитектурный мост и протокол верификации
```

---

## Быстрый Старт

### Сборка и запуск модуля Second Brain:
```cmd
build.bat run
```
Запускает полный верификационный комплекс (9 тестов, AVX-512 VPOPCNTDQ, 510K шагов/с).

### Запуск C-ABI интеграционного теста:
```cmd
build.bat test
```
Собирает библиотеку `second_brain/build/popcnt_brain.lib`, компилирует `test_c_api.cpp` и проверяет запись фактов, разрешение связей и детекцию логических противоречий за микросекунды.

### Очистка:
```cmd
build.bat clean
```
Удаляет все временные артефакты и директории сборки.

---

## License

MIT -- see [LICENSE](LICENSE). The repository history keeps the
review trail (the `.agents/` audit artifacts); the published tree
does not.
