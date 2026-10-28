import json
import os

channel_id = "waldani_qwen38_27b_session"

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cell_header",
   "metadata": {},
   "source": [
    "# 📦 One-Time Kaggle Dataset Creator for Qwen3.8-27B Heretic GGUF\n",
    "- **Source**: `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`\n",
    "- **File**: `RVN-Q5_K_S-multilingual-mtp.gguf` (~17.8 GB)\n",
    "- **Target Dataset**: `waldani/qwen38-27b-heretic`\n",
    "- **Hardware**: CPU Only (0 GPU Quota consumed!)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cell_create_dataset",
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess, sys, os, time, urllib.request, json\n",
    "\n",
    "def notify(msg):\n",
    "    print(f'[NOTIFY] {msg}', flush=True)\n",
    f"    channel = '{channel_id}'\n",
    "    try:\n",
    "        req = urllib.request.Request(f'https://ntfy.sh/{channel}', data=msg.encode('utf-8'))\n",
    "        urllib.request.urlopen(req, timeout=5)\n",
    "    except Exception as e:\n",
    "        print(f'Notify err: {e}')\n",
    "\n",
    "notify('Dataset creator started on Kaggle CPU...')\n",
    "\n",
    "# 1. Настройка окружения и утилит\n",
    "os.system('apt-get update -qq && apt-get install -y -qq aria2')\n",
    "os.system('pip install -q kaggle')\n",
    "\n",
    "# 2. Настройка Kaggle API ключа внутри контейнера\n",
    "kaggle_dir = os.path.expanduser('~/.kaggle')\n",
    "os.makedirs(kaggle_dir, exist_ok=True)\n",
    "token_file = os.path.join(kaggle_dir, 'access_token')\n",
    "with open(token_file, 'w') as f:\n",
    "    f.write('KGAT_bbed244431692ad717c105d2133d292f')\n",
    "os.chmod(token_file, 0o600)\n",
    "os.environ['KAGGLE_API_TOKEN'] = 'KGAT_bbed244431692ad717c105d2133d292f'\n",
    "\n",
    "# 3. Загрузка файла модели в /tmp/qwen_dataset\n",
    "ds_dir = '/tmp/qwen_dataset'\n",
    "os.makedirs(ds_dir, exist_ok=True)\n",
    "\n",
    "MODEL_FILENAME = 'RVN-Q5_K_S-multilingual-mtp.gguf'\n",
    "MODEL_PATH = os.path.join(ds_dir, MODEL_FILENAME)\n",
    "HF_URL = f'https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/{MODEL_FILENAME}'\n",
    "\n",
    "notify(f'Downloading {MODEL_FILENAME} (~17.8 GB) via aria2c (16 threads)...')\n",
    "t0 = time.time()\n",
    "cmd = [\n",
    "    'aria2c',\n",
    "    '-x', '16',\n",
    "    '-s', '16',\n",
    "    '-j', '16',\n",
    "    '-k', '2M',\n",
    "    '--file-allocation=none',\n",
    "    '--summary-interval=10',\n",
    "    '-d', ds_dir,\n",
    "    '-o', MODEL_FILENAME,\n",
    "    HF_URL\n",
    "]\n",
    "subprocess.run(cmd, check=True)\n",
    "dur = time.time() - t0\n",
    "size_gb = os.path.getsize(MODEL_PATH) / (1024**3)\n",
    "notify(f'Downloaded {size_gb:.2f} GB in {dur:.1f}s! Creating Kaggle Dataset...')\n",
    "\n",
    "# 4. Создание dataset-metadata.json\n",
    "meta = {\n",
    "    'title': 'qwen38-27b-heretic',\n",
    "    'id': 'waldani/qwen38-27b-heretic',\n",
    "    'licenses': [{'name': 'apache-2.0'}]\n",
    "}\n",
    "with open(os.path.join(ds_dir, 'dataset-metadata.json'), 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "# 5. Вызов Kaggle API для создания датасета\n",
    "print('Creating Kaggle dataset waldani/qwen38-27b-heretic...')\n",
    "from kaggle.api.kaggle_api_extended import KaggleApi\n",
    "api = KaggleApi()\n",
    "api.authenticate()\n",
    "res = api.dataset_create_new(folder=ds_dir, public=False, convert_to_csv=False, dir_mode='skip')\n",
    "print('Dataset create result:', res)\n",
    "notify('DATASET_CREATED: waldani/qwen38-27b-heretic is READY for instant mounting!')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(r'c:\Users\Администратор\Documents\project upgrade\kaggle_session\nb2\notebook5f9c599132.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('Generated nb2 successfully!')
