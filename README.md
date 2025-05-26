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
```

## Обучение классификатора лиц

### Датасеты

- **Женские лица** — 6318 изображений
  - Источник: [FaceShape Dataset](https://universe.roboflow.com/faceshape/faceshape-dfw6w)
  - Классы: Heart, Oblong, Oval, Round, Square

- **Мужские лица** — 1816 изображений
  - Источник: [Male Face Dataset](https://universe.roboflow.com/university-ufvlh/male-face)
  - Классы: Square, Oval, Round
  - Дополнительные метки: Curly Hair, Straight Hair

### Аугментация данных

- Горизонтальное отражение (flip)

### Результаты

- Достигнута точность классификации: 95%

