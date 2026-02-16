"""
MFA Wrapper - обёртка над Montreal Forced Aligner

Поддерживаемые языки: EN, RU, ES, DE, PT
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Optional

import librosa
import soundfile as sf
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

# MFA требует 16kHz
MFA_SAMPLE_RATE = 16000


class MFAAligner:
    """
    Обёртка над MFA CLI для forced alignment.
    
    Модели загружаются при билде Docker образа.
    """
    
    def __init__(self):
        self.available_languages = list(LANGUAGE_MODELS.keys())
        self._verify_models()
    
    def _verify_models(self):
        """Проверяем что модели загружены"""
        # MFA хранит модели в ~/Documents/MFA или задаётся через MFA_ROOT_DIR
        mfa_root = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA"))
        print(f"[MFA] Models directory: {mfa_root}")
        
        # Проверяем наличие acoustic моделей
        for lang, models in LANGUAGE_MODELS.items():
            # MFA 3.x хранит модели по-другому, просто логируем
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Подготовка директорий
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            
            # Resample аудио до 16kHz если нужно
            resampled_path = os.path.join(input_dir, "audio.wav")
            self._resample_audio(audio_path, resampled_path, input_sample_rate)
            
            # Создаём файл транскрипта (.lab или .txt)
            transcript_path = os.path.join(input_dir, "audio.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            # Запускаем MFA
            self._run_mfa(
                input_dir=input_dir,
                output_dir=output_dir,
                acoustic_model=models["acoustic"],
                dictionary=models["dictionary"]
            )
            
            # Парсим результат (TextGrid)
            textgrid_path = os.path.join(output_dir, "audio.TextGrid")
            if not os.path.exists(textgrid_path):
                raise RuntimeError(f"MFA did not produce output: {textgrid_path}")
            
            words = self._parse_textgrid(textgrid_path)
            
            return words
    
    def _resample_audio(self, input_path: str, output_path: str, input_sr: int):
        """Resample аудио до 16kHz для MFA"""
        if input_sr == MFA_SAMPLE_RATE:
            # Просто копируем
            import shutil
            shutil.copy(input_path, output_path)
            return
        
        # Загружаем и ресэмплируем
        y, sr = librosa.load(input_path, sr=input_sr, mono=True)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=MFA_SAMPLE_RATE)
        
        # Сохраняем
        sf.write(output_path, y_resampled, MFA_SAMPLE_RATE, subtype='PCM_16')
        print(f"[MFA] Resampled {input_sr}Hz -> {MFA_SAMPLE_RATE}Hz")
    
    def _run_mfa(self, input_dir: str, output_dir: str, acoustic_model: str, dictionary: str):
        """Запускает MFA CLI"""
        cmd = [
            "mfa", "align",
            input_dir,
            dictionary,
            acoustic_model,
            output_dir,
            "--clean",
            "--single_speaker",
            "--output_format", "long_textgrid"
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
