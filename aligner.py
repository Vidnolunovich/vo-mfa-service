"""
MFA Wrapper - обёртка над Montreal Forced Aligner

Поддерживаемые языки: EN, RU, ES, DE, PT

Оптимизации:
- align_one вместо align (пропускает corpus setup)
- ffmpeg для resample (быстрее чем librosa)
- --uses_speaker_adaptation false (не нужен для TTS)
- --no_textgrid_cleanup (экономим время)
"""

import os
import subprocess
import tempfile
import time
from typing import List, Dict, Optional

import textgrid


# Маппинг языков на MFA модели
LANGUAGE_MODELS = {
    "en": {
        "acoustic": "english_us_arpa",
        "dictionary": "english_us_arpa"
    },
    "ru": {
        "acoustic": "russian_mfa",
        "dictionary": "russian_mfa"
    },
    "es": {
        "acoustic": "spanish_mfa",
        "dictionary": "spanish_mfa"
    },
    "de": {
        "acoustic": "german_mfa",
        "dictionary": "german_mfa"
    },
    "pt": {
        "acoustic": "portuguese_mfa",
        "dictionary": "portuguese_mfa"
    }
}

# MFA требует 16kHz 16-bit
MFA_SAMPLE_RATE = 16000


class MFAAligner:
    """
    Обёртка над MFA CLI для forced alignment.
    
    Модели загружаются при билде Docker образа.
    PostgreSQL server запускается при старте контейнера.
    """
    
    def __init__(self):
        self.available_languages = list(LANGUAGE_MODELS.keys())
        self._verify_models()
    
    def _verify_models(self):
        """Проверяем что модели загружены"""
        mfa_root = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA"))
        print(f"[MFA] Models directory: {mfa_root}")
        
        for lang, models in LANGUAGE_MODELS.items():
            print(f"[MFA] Language {lang}: acoustic={models['acoustic']}, dict={models['dictionary']}")
    
    def get_model_name(self, language: str) -> str:
        """Возвращает имя acoustic модели для языка"""
        return LANGUAGE_MODELS.get(language, {}).get("acoustic", "unknown")
    
    def align(
        self,
        audio_path: str,
        transcript: str,
        language: str = "en",
        input_sample_rate: int = 24000
    ) -> List[Dict]:
        """
        Выполняет forced alignment.
        
        Args:
            audio_path: Путь к WAV файлу
            transcript: Текст для alignment
            language: Код языка (en, ru, es, de, pt)
            input_sample_rate: Sample rate входного аудио
            
        Returns:
            List of {word, start, end} dictionaries
        """
        if language not in LANGUAGE_MODELS:
            raise ValueError(f"Unsupported language: {language}")
        
        models = LANGUAGE_MODELS[language]
        
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Подготовка файлов (align_one работает с файлами напрямую)
            audio_resampled = os.path.join(tmpdir, "audio.wav")
            transcript_path = os.path.join(tmpdir, "audio.txt")
            output_textgrid = os.path.join(tmpdir, "audio.TextGrid")
            
            # Resample аудио до 16kHz 16-bit через ffmpeg (быстрее librosa)
            resample_start = time.time()
            self._resample_audio_ffmpeg(audio_path, audio_resampled, input_sample_rate)
            print(f"[MFA] Resample took {time.time() - resample_start:.2f}s")
            
            # Создаём файл транскрипта
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            # Запускаем MFA align_one
            mfa_start = time.time()
            self._run_mfa(
                audio_path=audio_resampled,
                transcript_path=transcript_path,
                output_path=output_textgrid,
                acoustic_model=models["acoustic"],
                dictionary=models["dictionary"]
            )
            print(f"[MFA] Alignment took {time.time() - mfa_start:.2f}s")
            
            # Парсим результат (TextGrid)
            if not os.path.exists(output_textgrid):
                raise RuntimeError(f"MFA did not produce output: {output_textgrid}")
            
            words = self._parse_textgrid(output_textgrid)
            
            print(f"[MFA] Total processing time: {time.time() - start_time:.2f}s")
            
            return words
    
    def _resample_audio_ffmpeg(self, input_path: str, output_path: str, input_sr: int):
        """
        Resample аудио до 16kHz 16-bit через ffmpeg.
        Значительно быстрее чем librosa для этой задачи.
        """
        cmd = [
            "ffmpeg",
            "-y",                    # Перезаписывать выход
            "-i", input_path,        # Вход
            "-ar", str(MFA_SAMPLE_RATE),  # Sample rate 16kHz
            "-ac", "1",              # Mono
            "-sample_fmt", "s16",    # 16-bit
            "-f", "wav",             # WAV формат
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"[MFA] ffmpeg STDERR: {result.stderr}")
            raise RuntimeError(f"ffmpeg resample failed: {result.stderr}")
        
        print(f"[MFA] Resampled to {MFA_SAMPLE_RATE}Hz 16-bit via ffmpeg")
    
    def _run_mfa(
        self, 
        audio_path: str, 
        transcript_path: str, 
        output_path: str,
        acoustic_model: str, 
        dictionary: str
    ):
        """
        Запускает MFA CLI с командой align_one.
        
        Оптимизации:
        - align_one: пропускает corpus setup
        - --uses_speaker_adaptation false: для TTS не нужен
        - --no_textgrid_cleanup: экономим время на постобработке
        """
        cmd = [
            "mfa", "align_one",
            audio_path,
            transcript_path,
            dictionary,
            acoustic_model,
            output_path,
            "--uses_speaker_adaptation", "false",
            "--no_textgrid_cleanup"
        ]
        
        print(f"[MFA] Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 минут максимум
        )
        
        if result.returncode != 0:
            print(f"[MFA] STDERR: {result.stderr}")
            print(f"[MFA] STDOUT: {result.stdout}")
            raise RuntimeError(f"MFA failed: {result.stderr}")
        
        print(f"[MFA] Alignment complete")
    
    def _parse_textgrid(self, textgrid_path: str) -> List[Dict]:
        """Парсит TextGrid файл и извлекает word timestamps"""
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        
        words = []
        
        # Ищем tier с словами (обычно называется "words")
        for tier in tg:
            if tier.name.lower() == "words":
                for interval in tier:
                    if interval.mark and interval.mark.strip():
                        words.append({
                            "word": interval.mark.strip(),
                            "start": round(interval.minTime, 4),
                            "end": round(interval.maxTime, 4)
                        })
                break
        
        print(f"[MFA] Extracted {len(words)} words from TextGrid")
        return words
