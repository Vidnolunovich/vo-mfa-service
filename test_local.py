"""
Локальный тест MFA без Docker.

Использование:
1. Установить MFA: conda install -c conda-forge montreal-forced-aligner
2. Скачать модель: mfa model download acoustic english_us_arpa
3. Скачать словарь: mfa model download dictionary english_us_arpa
4. Положить test.wav в текущую директорию
5. Запустить: python test_local.py
"""

import os
import sys

# Добавляем текущую директорию в path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aligner import MFAAligner
from rms_refiner import refine_word_endpoints, analyze_audio_energy


def test_alignment():
    """Тестовый alignment"""
    
    # Тестовые данные
    audio_path = "test.wav"  # Положите ваш WAV файл сюда
    transcript = "Hello world today we will make something amazing"
    
    if not os.path.exists(audio_path):
        print(f"ERROR: {audio_path} not found")
        print("Please provide a test WAV file")
        return
    
    # Анализ аудио
    print("\n=== Audio Analysis ===")
    stats = analyze_audio_energy(audio_path)
    print(f"Duration: {stats['duration_sec']}s")
    print(f"Peak: {stats['peak_db']:.1f}dB")
    print(f"Mean: {stats['mean_db']:.1f}dB")
    print(f"Silence ratio: {stats['silence_ratio']:.1%}")
    
    # Alignment
    print("\n=== MFA Alignment ===")
    aligner = MFAAligner()
    
    words = aligner.align(
        audio_path=audio_path,
        transcript=transcript,
        language="en",
        input_sample_rate=24000
    )
    
    print(f"\nExtracted {len(words)} words:")
    for w in words:
        print(f"  {w['word']:15} {w['start']:.3f} - {w['end']:.3f}")
    
    # RMS Refinement
    print("\n=== RMS Refinement ===")
    refined = refine_word_endpoints(
        audio_path=audio_path,
        words=words,
        sample_rate=24000
    )
    
    print(f"\nRefined words:")
    for orig, ref in zip(words, refined):
        shift_ms = (ref['end'] - orig['end']) * 1000
        print(f"  {ref['word']:15} {ref['start']:.3f} - {ref['end']:.3f}  (+{shift_ms:.1f}ms)")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    test_alignment()
