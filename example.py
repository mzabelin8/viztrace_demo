from viztracer import VizTracer
import re
import time
import random
from collections import Counter

sample_text = """
Python is a versatile programming language that was created by Guido van Rossum
and first released in 1991. Python's design philosophy emphasizes code readability 
with its notable use of significant whitespace. Its language constructs as well as its
object-oriented approach aim to help programmers write clear, logical code for small and 
large-scale projects. Python is dynamically typed and garbage-collected. It supports 
multiple programming paradigms, including procedural, object-oriented, and functional 
programming. Python is often described as a "batteries included" language due to its 
comprehensive standard library.
"""

def get_timestamp():
    return time.strftime("%H:%M:%S.%f")[:-3]

def tokenize_text(text, min_length=3):
    tracer.log_instant(f"🚩 ПРОФИЛЬ: Начало токенизации [{get_timestamp()}]")
    
    tracer.log_instant("Начало токенизации")
    
    tracer.log_var("input_text_length", len(text))
    
    start_time = time.time()
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    preprocess_time = time.time() - start_time
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Предобработка текста заняла {preprocess_time:.4f}с")
    
    words = text.split()
    
    filtered_words = [word for word in words if len(word) >= min_length]
    
    tracer.log_var("words_before_filter", len(words))
    tracer.log_var("words_after_filter", len(filtered_words))
    
    tracer.log_instant("Конец токенизации")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"✅ ПРОФИЛЬ: Токенизация завершена [{get_timestamp()}] - {total_time:.4f}с")
    
    time.sleep(0.05)
    
    return filtered_words

def remove_stopwords(word_list, stopwords):
    tracer.log_instant(f"🚩 ПРОФИЛЬ: Начало удаления стоп-слов [{get_timestamp()}]")
    
    tracer.log_instant("Удаление стоп-слов[")
    
    tracer.log_var("word_list_size", len(word_list))
    tracer.log_var("stopwords_count", len(stopwords))
    
    start_time = time.time()
    
    filtered_words = []
    removed_count = 0
    
    for i, word in enumerate(word_list):
        if i % 10 == 0:
            tracer.log_var("progress", i)
        
        if word not in stopwords:
            filtered_words.append(word)
        else:
            removed_count += 1
            
        if i == len(word_list) // 2:
            mid_time = time.time() - start_time
            tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Половина стоп-слов обработана за {mid_time:.4f}с")
    
    tracer.log_var("removed_count", removed_count)
    tracer.log_var("remaining_count", len(filtered_words))
    
    tracer.log_instant("Удаление стоп-слов]")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"✅ ПРОФИЛЬ: Удаление стоп-слов завершено [{get_timestamp()}] - {total_time:.4f}с")
    
    time.sleep(0.05)
    
    return filtered_words

def calculate_word_frequency(word_list):
    tracer.log_instant(f"🚩 ПРОФИЛЬ: Начало расчета частоты слов [{get_timestamp()}]")
    
    tracer.log_instant("Расчет частоты слов[")
    
    tracer.log_var("unique_words", len(set(word_list)))
    tracer.log_var("total_words", len(word_list))
    
    start_time = time.time()
    
    if len(word_list) > 50:
        tracer.log_instant(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Большой набор слов ({len(word_list)}) может занять время")
    
    time.sleep(0.1)
    
    word_freq = Counter(word_list)
    
    count_time = time.time() - start_time
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Подсчет частоты слов занял {count_time:.4f}с")
    
    most_common = word_freq.most_common(5)
    
    most_common_str = ", ".join([f"{word}:{count}" for word, count in most_common[:3]])
    tracer.log_var("most_common_words", most_common_str)
    
    tracer.log_instant("Расчет частоты слов]")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"✅ ПРОФИЛЬ: Расчет частоты завершен [{get_timestamp()}] - {total_time:.4f}с")
    
    return word_freq

def process_text_segment(text_segment, segment_id):
    tracer.log_instant(f"🚩 ПРОФИЛЬ: Начало обработки сегмента {segment_id} [{get_timestamp()}]")
    
    tracer.log_instant(f"Обработка сегмента {segment_id}[")
    
    tracer.log_var(f"segment_{segment_id}_length", len(text_segment))
    
    start_time = time.time()
    
    tokens = tokenize_text(text_segment)
    
    tokenize_time = time.time() - start_time
    tracer.log_instant(f"Токенизация сегмента {segment_id} завершена")
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Токенизация сегмента {segment_id} - {tokenize_time:.4f}с")
    
    stopwords = ['the', 'and', 'is', 'to', 'in', 'that', 'its', 'with', 'as', 'for']
    
    stopwords_start = time.time()
    filtered_tokens = remove_stopwords(tokens, stopwords)
    
    stopwords_time = time.time() - stopwords_start
    tracer.log_instant(f"Удаление стоп-слов для сегмента {segment_id} завершено")
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Удаление стоп-слов сегмента {segment_id} - {stopwords_time:.4f}с")
    
    freq_start = time.time()
    word_freq = calculate_word_frequency(filtered_tokens)
    freq_time = time.time() - freq_start
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Расчет частоты сегмента {segment_id} - {freq_time:.4f}с")
    
    delay = random.uniform(0.05, 0.2)
    time.sleep(delay)
    
    result = {
        'segment_id': segment_id,
        'token_count': len(tokens),
        'filtered_count': len(filtered_tokens),
        'most_common': word_freq.most_common(3)
    }
    
    tracer.log_var(f"segment_{segment_id}_tokens", result['token_count'])
    tracer.log_var(f"segment_{segment_id}_filtered", result['filtered_count'])
    
    tracer.log_instant(f"Обработка сегмента {segment_id}]")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"✅ ПРОФИЛЬ: Сегмент {segment_id} обработан [{get_timestamp()}] - {total_time:.4f}с")
    
    return result

def analyze_text(text):
    tracer.log_instant(f"🚩 ПРОФИЛЬ: Начало полного анализа текста [{get_timestamp()}]")
    
    tracer.log_instant("Анализ текста[")
    
    tracer.log_var("text_size", len(text))
    
    start_time = time.time()
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    tracer.log_var("sentence_count", len(sentences))
    
    tracer.log_instant(f"ℹ️ ИНФОРМАЦИЯ: Обнаружено {len(sentences)} предложений для анализа")
    
    results = []
    
    for i, sentence in enumerate(sentences):
        tracer.log_instant(f"Начало обработки предложения {i}")
        
        result = process_text_segment(sentence, i)
        results.append(result)
        
        tracer.log_instant(f"Конец обработки предложения {i}")
        
        progress_pct = (i + 1) / len(sentences) * 100
        tracer.log_instant(f"📊 ПРОГРЕСС: Обработано {i+1}/{len(sentences)} предложений ({progress_pct:.1f}%)")
    
    total_tokens = sum(r['token_count'] for r in results)
    total_filtered = sum(r['filtered_count'] for r in results)
    
    process_time = time.time() - start_time
    tracer.log_instant(f"⏱️ ИЗМЕРЕНИЕ: Основная обработка завершена за {process_time:.4f}с")
    
    all_words = []
    for r in results:
        all_words.extend([word for word, count in r['most_common']])
    
    overall_freq = Counter(all_words)
    
    tracer.log_var("total_sentences", len(sentences))
    tracer.log_var("total_tokens", total_tokens)
    tracer.log_var("total_filtered", total_filtered)
    
    tracer.log_instant("Анализ текста]")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"✅ ПРОФИЛЬ: Анализ текста завершен [{get_timestamp()}] - {total_time:.4f}с")
    
    return {
        'sentence_count': len(sentences),
        'total_tokens': total_tokens,
        'total_filtered': total_filtered,
        'overall_most_common': overall_freq.most_common(5)
    }

def main():
    start_time = time.time()
    tracer.log_instant(f"🚀 ЗАПУСК ПРОГРАММЫ: [{get_timestamp()}]")
    
    tracer.log_instant("Программа[")
    
    tracer.log_var("start_time", time.strftime("%H:%M:%S"))
    
    result = analyze_text(sample_text)
    
    print("\nРезультаты анализа текста:")
    print(f"Количество предложений: {result['sentence_count']}")
    print(f"Общее количество токенов: {result['total_tokens']}")
    print(f"Количество токенов после фильтрации: {result['total_filtered']}")
    print("Наиболее частые слова:")
    for word, count in result['overall_most_common']:
        print(f"  - {word}: {count}")
    
    tracer.log_var("end_time", time.strftime("%H:%M:%S"))
    
    tracer.log_instant("Программа]")
    
    total_time = time.time() - start_time
    tracer.log_instant(f"🏁 ПРОГРАММА ЗАВЕРШЕНА: [{get_timestamp()}] - Общее время {total_time:.4f}с")

if __name__ == "__main__":
    tracer = VizTracer(
        log_gc=True,
        max_stack_depth=10,
        output_file="viztracer_text_analysis.json"
    )
    
    tracer.start()
    
    main()
    
    tracer.stop()
    tracer.save()
