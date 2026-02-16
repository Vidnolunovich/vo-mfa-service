# vo-mfa-service

**Montreal Forced Aligner microservice** для получения точных word-level timestamps в TTS аудио.

## Зачем это нужно

Google Cloud STT даёт timestamps с точностью ±300-700ms. Для точной нарезки Single-Shot TTS аудио по сценам этого недостаточно — слова обрезаются на границах.

MFA (Montreal Forced Aligner) — специализированный инструмент для forced alignment:
- Точность: **±10-20ms** на границы слов
- С RMS refinement: **±5-10ms** на endTime
- Стоимость: **~80x дешевле** Google STT

## Архитектура

```
vo-workflow-code (Node.js)
        │
        ▼ POST /align
vo-mfa-service (Python + MFA)
        │
        ▼
{words: [{word, start, end}, ...]}
```

## API

### POST /align

**Request:**
```json
{
  "audio_base64": "...",
  "transcript": "Hello world today we will make...",
  "language": "en",
  "refine_endpoints": true,
  "sample_rate": 24000
}
```

**Response:**
```json
{
  "words": [
    {"word": "Hello", "start": 0.100, "end": 0.455},
    {"word": "world", "start": 0.455, "end": 0.823}
  ],
  "total_duration": 45.230,
  "processing_time_ms": 2340,
  "model_used": "english_us_arpa",
  "refined": true
}
```

### GET /health

```json
{
  "status": "healthy",
  "models_loaded": ["en", "ru", "es", "de", "pt"],
  "version": "1.0.0"
}
```

## Поддерживаемые языки

| Язык | Код | MFA модель |
|------|-----|------------|
| English | en | english_us_arpa |
| Russian | ru | russian_mfa |
| Spanish | es | spanish_mfa |
| German | de | german_mfa |
| Portuguese | pt | portuguese_mfa |

## Локальный запуск

### С Docker (рекомендуется)

```bash
# Сборка (долго, ~15-20 мин — качаются модели)
docker build -t vo-mfa-service .

# Запуск
docker run -p 8080:8080 vo-mfa-service

# Тест
curl http://localhost:8080/health
```

### Без Docker (для разработки)

```bash
# Установить MFA через conda
conda install -c conda-forge montreal-forced-aligner

# Скачать модели
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Установить Python зависимости
pip install -r requirements.txt

# Запустить
python main.py
```

## Деплой в Cloud Run

```bash
# Сборка и пуш в Container Registry
gcloud builds submit --tag gcr.io/gen-lang-client-0960712901/vo-mfa-service

# Деплой
gcloud run deploy vo-mfa-service \
  --image gcr.io/gen-lang-client-0960712901/vo-mfa-service \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 10
```

## RMS Refinement

MFA может давать endTime чуть раньше реального окончания звука (особенно на шипящих: s, sh, f).

RMS refinement:
1. Берёт окно от endTime - 20ms до endTime + 80ms
2. Вычисляет RMS энергию в 5ms кадрах
3. Находит первый кадр с RMS < -40dB
4. Сдвигает endTime туда + 5ms padding

Результат: **endTime гарантированно НЕ раньше реального окончания слова**.

## Интеграция с vo-workflow-code

См. `services/mfa.js` в основном репозитории.

```javascript
const response = await fetch('https://vo-mfa-service-xxx.run.app/align', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    audio_base64: audioBuffer.toString('base64'),
    transcript: fullScript,
    language: 'en',
    refine_endpoints: true,
    sample_rate: 24000
  })
});

const { words } = await response.json();
// words = [{word: "Hello", start: 0.100, end: 0.455}, ...]
```

## Лицензия

MIT
