# ==============================================================================
# Google Colab SSH Bridge (colab-ssh + Cloudflare Tunnel)
# ==============================================================================
!pip install -q colab_ssh --upgrade

from colab_ssh import launch_ssh_cloudflared

# Пароль для входа (на случай ручного ввода)
PASSWORD = "colab"

# Запуск туннеля через официальную библиотеку colab-ssh
launch_ssh_cloudflared(password=PASSWORD)

# Авторизация открытого ключа вашего ПК для мгновенного входа без пароля
LOCAL_PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAlcVVcs5DNr2kNfwLJtBjYQocNsdLCDcUKRIgquwK9J colab-bridge@local"
!mkdir -p /root/.ssh && echo "{LOCAL_PUBKEY}" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
