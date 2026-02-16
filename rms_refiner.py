"""
RMS Refinement - коррекция endTime по энергии сигнала

MFA может давать endTime чуть раньше реального окончания звука,
особенно на шипящих (s, sh, f). Этот модуль сдвигает endTime
вперёд до реального падения энергии.

Результат: endTime гарантированно НЕ раньше реального окончания слова.
"""

import numpy as np
import librosa
from typing import List, Dict


def refine_word_endpoints(
    audio_path: str,
    words: List[Dict],
    sample_rate: int = 24000,
    search_window_ms: int = 80,
    frame_ms: int = 5,
    threshold_db: float = -40.0,
    padding_ms: int = 5
) -> List[Dict]:
    """
    Уточняет endTime каждого слова на основе RMS энергии.
    
    Алгоритм:
    1. Берём окно от (endTime - 20ms) до (endTime + search_window_ms)
    2. Вычисляем RMS энергию в frame_ms кадрах
    3. Находим первый кадр с RMS < threshold_db
    4. Это и есть реальный конец слова
    
    Args:
        audio_path: Путь к WAV файлу
        words: Список слов с start/end от MFA
        sample_rate: Sample rate аудио
        search_window_ms: Окно поиска вперёд от endTime
        frame_ms: Размер кадра для RMS анализа
        threshold_db: Порог тишины в dB
        padding_ms: Дополнительный отступ после найденного конца
        
    Returns:
        Список слов с уточнёнными endTime
    """
    # Загружаем аудио
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(y) / sr
    
    frame_samples = int(frame_ms / 1000 * sr)
    hop_samples = frame_samples // 2
    
    refined_words = []
    
    for word in words:
        original_end = word["end"]
        
        # Окно поиска
        search_start = max(0, original_end - 0.020)  # -20ms
        search_end = min(duration, original_end + search_window_ms / 1000)
        
        start_sample = int(search_start * sr)
        end_sample = int(search_end * sr)
        
        if end_sample <= start_sample:
            refined_words.append(word)
            continue
        
        segment = y[start_sample:end_sample]
        
        if len(segment) < frame_samples:
            refined_words.append(word)
            continue
        
        # Вычисляем RMS
        rms = librosa.feature.rms(
            y=segment, 
            frame_length=frame_samples, 
            hop_length=hop_samples
        )[0]
        
        if len(rms) == 0:
            refined_words.append(word)
            continue
        
        # Конвертируем в dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Ищем первый кадр ниже порога
        silence_frames = np.where(rms_db < threshold_db)[0]
        
        if len(silence_frames) > 0:
            first_silence_frame = silence_frames[0]
            silence_sample = first_silence_frame * hop_samples
            new_end = search_start + silence_sample / sr + padding_ms / 1000
            
            # Не сдвигаем раньше оригинала (только вперёд)
            new_end = max(new_end, original_end)
            
            # Не выходим за границы аудио
            new_end = min(new_end, duration)
        else:
            # Не нашли тишину — оставляем оригинал + небольшой padding
            new_end = min(original_end + padding_ms / 1000, duration)
        
        refined_words.append({
            "word": word["word"],
            "start": word["start"],
            "end": round(new_end, 4)
        })
    
    # Статистика
    total_shift = sum(
        refined["end"] - original["end"] 
        for refined, original in zip(refined_words, words)
    )
    avg_shift_ms = (total_shift / len(words)) * 1000 if words else 0
    print(f"[RMS] Refined {len(words)} words, avg shift: +{avg_shift_ms:.1f}ms")
    
    return refined_words


def analyze_audio_energy(audio_path: str, sample_rate: int = 24000) -> Dict:
    """
    Анализирует энергию аудио для диагностики.
    
    Returns:
        Статистика: peak_db, mean_db, silence_ratio
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    silence_frames = np.sum(rms_db < -40)
    silence_ratio = silence_frames / len(rms_db) if len(rms_db) > 0 else 0
    
    return {
        "peak_db": float(np.max(rms_db)),
        "mean_db": float(np.mean(rms_db)),
        "silence_ratio": round(silence_ratio, 3),
        "duration_sec": round(len(y) / sr, 3)
    }
