# Project Upgrade: Hybrid Neural & PopCount Architecture

Проект объединяет два специализированных вычислительных контура, оптимизированных под аппаратные возможности этой рабочей станции (Intel Core i7-11800H + NVIDIA RTX 3050 Laptop):

```
project upgrade/
├── second_brain/       # [НОВОЕ] PopCount VSA Ассоциативная Память («Второй Мозг» для LLM)
│   ├── popcnt_brain.h  # Публичный чистый C-ABI интерфейс
│   ├── popcnt_brain.cpp# Ядро движка (AVX-512 VPOPCNTDQ, SimHash, FFF, 4KB Demand-Paged VSA)
│   ├── test_c_api.cpp  # Интеграционный C-тест
│   ├── build.bat       # Сборка движка, тестов и статической библиотеки
│   ├── CMakeLists.txt  # Сборка через CMake (exe, static lib, shared dll)
│   └── README.md       # Подробная документация и примеры связки с LLM / Python
│
├── legacy_ptq/         # Исходный PTQ CUDA-движок квантованного инференса
│   ├── engine.c        # Хостовый пайплайн
│   ├── kernel.cu       # CUDA-ядра инференса
│   └── core.h          # Заголовочный файл
│
├── build.bat           # Главный оркестратор сборки проекта
├── CMakeLists.txt      # Корневой CMake-конфиг
├── BRIDGE.md           # Архитектурный мост и протокол взаимодействия
└── .gitignore          # Исключения бинарных артефактов и временных файлов
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
