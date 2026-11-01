# HSAA v3.0 — Unified Neuro-Symbolic Architecture & Generalization Lab

Интерактивный исследовательский стенд, эмпирический верификатор и пропатченные исходники C++20 для гибридной нейро-символьной архитектуры **[0waldani0/HSAA](https://github.com/0waldani0/HSAA)** (Hyperdimensional Spline Associative Architecture).

---

## Структура репозитория

```text
├── hsaa_cpp/                        # Пропатченные C++20 исходники из 0waldani0/HSAA (v3.0 Unified)
│   ├── include/
│   │   ├── hssa.h                   # Заголовок 28-слойного гиперсферического ядра
│   │   ├── hsaa_unified_v3.hpp      # Единый класс hsaa_v3::HsaaUnifiedCoreV3 (Shared 1024-D Bus)
│   │   └── unified_engine.h         # Заголовок Unified Engine
│   ├── second_brain/
│   │   └── popcnt_brain.h           # Popcnt-ассоциативная память + динамический граф триплетов
│   └── src/
│       ├── hssa.cpp                 # Мульти-масштабный распад (Patch #3) across 28 layers
│       ├── unified_engine.cpp       # 7-базисный спектральный солвер (Patch #1) + 4096-D декодер (Patch #2)
│       └── vsa_db.cpp               # 512-битная VSA база + Simplex Gate проверка конфликтов
├── public/
│   └── hsaa_v3_unified.patch        # Готовый git-патч для применения поверх 0waldani0/HSAA
└── src/                             # Интерактивный стенд Angular 21 + математическое ядро TypeScript
    └── app/
        ├── engine/
        │   ├── hsaa-engine.ts       # Полная реализация 512-bit VSA, RFF/KAN, 7-basis solver, v3.0 Bus
        │   └── hsaa-engine.test.ts  # 8 модульных тестов (node:test / tsx)
        ├── app.ts                   # Логика стенда, реактивные формы и генератор C++20
        └── app.html                 # Интерфейс лаборатории + Инженерный самоаудит (Radical Honesty)
```

---

## Что исправлено в HSAA v2.1 / v3.0 (`hsaa_cpp/` и `hsaa_v3_unified.patch`)

1. **Patch #1 — 7-Basis Adaptive Spectral Solver (`hsaa_cpp/src/unified_engine.cpp`)**
   - Расширяет 5-точечный конечный разностный стенсил до 7-функционного спектрально-дифференциального базиса `{1, t, t², sin(ωt), cos(ωt), e^(-γt)cos(ωt), e^(-γt)sin(ωt)}` с двухэтапным уточнением координат $(\gamma, \omega)$ и QR/Cholesky МНК.
2. **Patch #2 — Dynamic VSA Multi-Hop Graph Traversal (`hsaa_cpp/second_brain/popcnt_brain.h`)**
   - Заменяет жесткие `strstr`-шаблоны в `popcnt_brain_query` на итеративный многошаговый обход `brain->assoc_pages` по косинусному/Хэмминговому резонансу `Unbind(M, Role_Subj ⊕ Concept)`.
3. **Patch #3 — Multi-Scale Decay Schedule (`hsaa_cpp/src/hssa.cpp`)**
   - Вводит логарифмическое распределение временных констант $\gamma_l \in [0.85, 0.998]$ и ортогональное фазовое вращение по 28 слоям вместо вырожденного $\gamma = 0.99$.
4. **Patch #4 — Manifold-to-VSA Resonator Decoder (`hsaa_cpp/src/unified_engine.cpp`)**
   - Декодирует 4096-D непрерывный вектор `thought_state` обратно в символьные концепты из `eng->stored_facts` с проверкой непротиворечивости через `vsa_db_check_conflict`.
5. **Patch #5 — Convex Hull Extrapolation Gate (`hsaa_cpp/include/hsaa_unified_v3.hpp`)**
   - Вычисляет расстояние Махаланобиса $D_M(x)$ до обучающего распределения и при выходе за пределы выпуклой оболочки ($D_M > 1.0$) переключает предсказание с локальных RFF-сплайнов на глобальный закон сохранения символьного инварианта.

---

## Быстрый старт

### 1. Запуск интерактивного стенда
```bash
npm install
npm start
```
Приложение откроется на `http://localhost:3000`.

### 2. Запуск модульных тестов математического ядра
```bash
npx tsx --test src/app/engine/hsaa-engine.test.ts
```

### 3. Применение C++20 патча к оригинальному репозиторию `0waldani0/HSAA`
```bash
git clone https://github.com/0waldani0/HSAA.git
cd HSAA
git apply /path/to/public/hsaa_v3_unified.patch
```
