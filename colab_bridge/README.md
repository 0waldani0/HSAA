# Интеграция Google Colab с локальным ПК

Настроена полная двухсторонняя интеграция:
1. **Local Runtime (Локальная среда)**: веб-интерфейс Google Colab выполняет код на ресурсах вашего локального ПК (Intel Core i7-11800H + NVIDIA RTX 3050 Laptop 4GB) с прямым доступом к файлам проекта.
2. **Remote GPU SSH (Удаленный доступ)**: локальный терминал и IDE подключаются к мощным облачным GPU Google Colab (T4 / V100 / A100 / L4) через безопасный туннель Cloudflare.

---

## Режим 1. Локальный рантайм (Local Runtime)

Google Colab исполняет код на вашем компьютере.

### Быстрый запуск
1. Запустите файл `colab_bridge\start_local_runtime.bat` (или выполните `.\start_local_runtime.ps1` в PowerShell).
2. В консоли появится URL с токеном доступа, например:
   ```text
   http://localhost:8888/tree?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Откройте любой блокнот в [Google Colab](https://colab.research.google.com).
4. В правом верхнем углу нажмите стрелку рядом с кнопкой **Подключиться (Connect)** $\rightarrow$ **Подключиться к локальной среде выполнения (Connect to local runtime)**.
5. Вставьте скопированный URL в поле адреса и нажмите **Подключиться**.

> Вся директория проекта `project upgrade` монтируется как корневая папка сервера, поэтому все локальные файлы и C++/CUDA сборки доступны напрямую из блокнота.

---

## Режим 2. Удаленный доступ к GPU Colab (Remote SSH via Cloudflare)

Вы пишете код на локальном ПК, а вычисления производятся на облачных GPU Google Colab.

### Быстрый запуск
1. Откройте [Google Colab](https://colab.research.google.com) в браузере.
2. Загрузите файл блокнота `colab_bridge\colab_remote_gpu.ipynb` (или скопируйте код из `colab_bridge\colab_remote_gpu_cell.py` в пустую ячейку).
3. Переключите среду на GPU:
   - **Среда выполнения (Runtime)** $\rightarrow$ **Сменить среду (Change runtime type)** $\rightarrow$ выберите **T4 GPU** или **A100 GPU**.
4. Запустите ячейку.
5. Скрипт настроит SSH-сервер, авторизует ваш локальный открытый ключ и выведет команду:
   ```text
   ssh root@xxxx-xxxx.trycloudflare.com
   ```
6. Выполните эту команду в терминале Windows (PowerShell или cmd).
   - Подключение происходит автоматически по ключу `id_ed25519` **без ввода пароля**.
7. В **Antigravity IDE / VS Code**:
   - Нажмите `F1` $\rightarrow$ `Remote-SSH: Connect to Host...` $\rightarrow$ укажите `root@xxxx-xxxx.trycloudflare.com`.

---

## Режим 3. Готовый блокнот для запуска ИИ-моделей (GPU / LLM / Gradio)

Файл блокнота: `colab_bridge\colab_ai_models_runner.ipynb`.
Оптимизирован для GPU NVIDIA (T4, L4, A100) с 4-битным квантованием BitsAndBytes (NF4) и Gradio.

---

## Режим 4. Максимальная производительность на Google Cloud TPU v5e (JAX / bfloat16)

Файл блокнота: `colab_bridge\colab_tpu_v5e_runner.ipynb`.

Специализированный блокнот под ускорители **TPU v5e** (197 TFLOPS, 16 ГБ высокоскоростной памяти HBM со скоростью 810 ГБ/с):
1. **Диагностика TPU**: проверка `jax.devices()` и `torch.accelerator`.
2. **PyTorch TPU (`torch-tpu`)**: запуск моделей Hugging Face (Qwen 2.5, Llama 3) в нативном формате `torch.bfloat16` через `device_map='tpu'`.
3. **Google KerasHub (JAX XLA)**: максимальный throughput благодаря аппаратной компиляции графов XLA под матричные ядра TPU (модели уровня Gemma 2).
4. **Gradio Chat**: интерактивный чат с моделью через браузер.
5. **SSH к ноде TPU**: подключение с локального ПК прямо в терминал ноды TPU.

Содержит настроенный пайплайн для работы с открытыми моделями (Qwen 2.5, Llama 3, Mistral) на GPU Google Colab:
1. **Проверка GPU**: проверка `nvidia-smi` и объема доступной видеопамяти.
2. **Стек библиотек**: `transformers`, `accelerate`, `bitsandbytes`, `gradio`.
3. **4-битное квантование (NF4)**: позволяет запускать модели до 7B/8B параметров в пределах бесплатного 15GB VRAM T4 GPU.
4. **Gradio Web UI**: готовый веб-интерфейс чата с моделью с публичной ссылкой (`share=True`).
5. **Встроенный SSH-туннель**: возможность при необходимости переключиться на управление из локальной IDE.

---

## Системные компоненты и конфигурация

- **Cloudflared**: установлен в `C:\Program Files (x86)\cloudflared\cloudflared.exe` и добавлен в системный `PATH`.
- **SSH конфигурация**: в `~/.ssh/config` настроен `ProxyCommand` для домена `*.trycloudflare.com`.
- **Ключи SSH**:
  - Приватный ключ: `C:\Users\Администратор\.ssh\id_ed25519`
  - Публичный ключ: `C:\Users\Администратор\.ssh\id_ed25519.pub`
- **Jupyter расширение**: `jupyter_http_over_ws` (версия 0.0.7) адаптирована под Python 3.12 и зарегистрирована в `jupyter_server_config.d`.
