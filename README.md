# Серверы и порты

- **Свелти фронт** `front` — порт `7778`
- **Бэк для фронта** `server_main` — порт `7779`
- **Сервер для видео** `server_video_local.py` — порт `7780`
- **Иммерс** `server_storage.py` — порт `7781`

## Видео обработка

1. `check` — проверка видео
2. `convert` — конвертация
3. `preprocessing` — предварительная обработка

## Запуск сервера

Чтобы запустить сервер, выполните команду:

```bash
source .venv/bin/activate && nohup python server_storage/server_storage.py > server.log 2>&1 &