#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <immintrin.h>
#include <cstring>

using namespace std;

// ============================================================================
// СКАЛЯРНАЯ ВЕРСИЯ (эталонная)
// ============================================================================
void prewitt_scalar(const unsigned char* src, unsigned char* dst, int w, int h) {
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int A = src[(y-1) * w + (x-1)];
            int B = src[(y-1) * w + x];
            int C = src[(y-1) * w + (x+1)];
            int D = src[y * w + (x-1)];
            int F = src[y * w + (x+1)];
            int G = src[(y+1) * w + (x-1)];
            int H = src[(y+1) * w + x];
            int I = src[(y+1) * w + (x+1)];
            
            // Горизонтальная свертка (вертикальные границы)
            int H_h = (A + B + C) - (G + H + I);
            // Вертикальная свертка (горизонтальные границы)
            int H_v = (A + D + G) - (C + F + I);
            
            float val = sqrtf(static_cast<float>(H_h * H_h + H_v * H_v));
            if (val > 255.0f) val = 255.0f;
            
            dst[y * w + x] = static_cast<unsigned char>(val);
        }
    }
}

// ============================================================================
// SIMD-ВЕРСИЯ (AVX2) - обработка по 8 пикселей
// ============================================================================
void prewitt_simd_avx2(const unsigned char* src, unsigned char* dst, int w, int h) {
    // Константа для ограничения (float)
    __m256 v_max_255 = _mm256_set1_ps(255.0f);
    
    for (int y = 1; y < h - 1; ++y) {
        int x = 1;
        
        // Основной цикл: обрабатываем по 8 пикселей
        for (; x <= w - 1 - 8; x += 8) {
            // -----------------------------------------------------------------
            // 1. ЗАГРУЗКА ДАННЫХ (8 пикселей в 256-битные регистры)
            // -----------------------------------------------------------------
            // Строка y-1 (верхняя)
            __m256i v_A = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y-1)*w + (x-1)]));
            __m256i v_B = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y-1)*w + x]));
            __m256i v_C = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y-1)*w + (x+1)]));
            
            // Строка y (средняя)
            __m256i v_D = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[y*w + (x-1)]));
            __m256i v_F = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[y*w + (x+1)]));
            
            // Строка y+1 (нижняя)
            __m256i v_G = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y+1)*w + (x-1)]));
            __m256i v_H = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y+1)*w + x]));
            __m256i v_I = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[(y+1)*w + (x+1)]));
            
            // -----------------------------------------------------------------
            // 2. РАСШИРЕНИЕ 8-bit → 16-bit → 32-bit (целые)
            // -----------------------------------------------------------------
            // Функция распаковки для каждого вектора:
            // - младшие 128 бит расширяем до 16-bit, затем до 32-bit
            // - старшие 128 бит расширяем до 16-bit, затем до 32-bit
            // - объединяем результаты
            
            // Вспомогательная лямбда для расширения 8-bit → 32-bit
            auto expand_u8_to_i32 = [](__m256i v) -> __m256i {
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                
                __m256i lo_16 = _mm256_cvtepu8_epi16(lo);
                __m256i hi_16 = _mm256_cvtepu8_epi16(hi);
                
                __m256i lo_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo_16));
                __m256i hi_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi_16));
                
                return _mm256_set_m128i(_mm256_castsi256_si128(hi_32), _mm256_castsi256_si128(lo_32));
            };
            
            __m256i v_A32 = expand_u8_to_i32(v_A);
            __m256i v_B32 = expand_u8_to_i32(v_B);
            __m256i v_C32 = expand_u8_to_i32(v_C);
            __m256i v_D32 = expand_u8_to_i32(v_D);
            __m256i v_F32 = expand_u8_to_i32(v_F);
            __m256i v_G32 = expand_u8_to_i32(v_G);
            __m256i v_H32 = expand_u8_to_i32(v_H);
            __m256i v_I32 = expand_u8_to_i32(v_I);
            
            // -----------------------------------------------------------------
            // 3. ВЫЧИСЛЕНИЕ СВЕРТОК (целочисленное)
            // -----------------------------------------------------------------
            // Горизонтальная свертка H_h = (A+B+C) - (G+H+I)
            __m256i sum_h_plus = _mm256_add_epi32(v_A32, _mm256_add_epi32(v_B32, v_C32));
            __m256i sum_h_minus = _mm256_add_epi32(v_G32, _mm256_add_epi32(v_H32, v_I32));
            __m256i H_h = _mm256_sub_epi32(sum_h_plus, sum_h_minus);
            
            // Вертикальная свертка H_v = (A+D+G) - (C+F+I)
            __m256i sum_v_plus = _mm256_add_epi32(v_A32, _mm256_add_epi32(v_D32, v_G32));
            __m256i sum_v_minus = _mm256_add_epi32(v_C32, _mm256_add_epi32(v_F32, v_I32));
            __m256i H_v = _mm256_sub_epi32(sum_v_plus, sum_v_minus);
            
            // -----------------------------------------------------------------
            // 4. ПРЕОБРАЗОВАНИЕ int32 → float
            // -----------------------------------------------------------------
            __m256 v_Hh_f = _mm256_cvtepi32_ps(H_h);
            __m256 v_Hv_f = _mm256_cvtepi32_ps(H_v);
            
            // -----------------------------------------------------------------
            // 5. ВЫЧИСЛЕНИЕ d = √(H_h² + H_v²)
            // -----------------------------------------------------------------
            __m256 v_h2 = _mm256_mul_ps(v_Hh_f, v_Hh_f);
            __m256 v_v2 = _mm256_mul_ps(v_Hv_f, v_Hv_f);
            __m256 v_sum = _mm256_add_ps(v_h2, v_v2);
            __m256 v_res = _mm256_sqrt_ps(v_sum);
            
            // -----------------------------------------------------------------
            // 6. ОГРАНИЧЕНИЕ ДО 255
            // -----------------------------------------------------------------
            v_res = _mm256_min_ps(v_res, v_max_255);
            
            // -----------------------------------------------------------------
            // 7. ПРЕОБРАЗОВАНИЕ float → 32-bit int → 16-bit → 8-bit
            // -----------------------------------------------------------------
            __m256i v_int = _mm256_cvttps_epi32(v_res);           // float → int32
            
            // Упаковка 32→16→8
            __m256i v_short = _mm256_packus_epi32(v_int, v_int);  // int32 → uint16
            __m256i v_byte = _mm256_packus_epi16(v_short, v_short); // uint16 → uint8
            
            // -----------------------------------------------------------------
            // 8. СОХРАНЕНИЕ РЕЗУЛЬТАТА
            // -----------------------------------------------------------------
            // Извлекаем младшие 128 бит (8 пикселей) и сохраняем
            __m128i res_lo = _mm256_castsi256_si128(v_byte);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[y * w + x]), res_lo);
        }
        
        // ---------------------------------------------------------------------
        // ОСТАТОЧНЫЕ ПИКСЕЛИ (обработка скалярно)
        // ---------------------------------------------------------------------
        for (; x < w - 1; ++x) {
            int A = src[(y-1) * w + (x-1)];
            int B = src[(y-1) * w + x];
            int C = src[(y-1) * w + (x+1)];
            int D = src[y * w + (x-1)];
            int F = src[y * w + (x+1)];
            int G = src[(y+1) * w + (x-1)];
            int H = src[(y+1) * w + x];
            int I = src[(y+1) * w + (x+1)];
            
            int H_h = (A + B + C) - (G + H + I);
            int H_v = (A + D + G) - (C + F + I);
            
            float val = sqrtf(static_cast<float>(H_h * H_h + H_v * H_v));
            if (val > 255.0f) val = 255.0f;
            dst[y * w + x] = static_cast<unsigned char>(val);
        }
    }
}

// ============================================================================
// ТЕСТИРОВАНИЕ
// ============================================================================
int main() {
    const int w = 2048;
    const int h = 2048;
    
    // Создание тестового изображения
    vector<unsigned char> src(w * h, 128);
    for (int i = 0; i < h; ++i) {
        src[i * w + w/2] = 255;     // вертикальная линия
        src[i * w + w/4] = 50;      // тёмная вертикальная линия
    }
    for (int i = 0; i < w; ++i) {
        src[h/2 * w + i] = 200;     // горизонтальная линия
    }
    
    vector<unsigned char> dst_scalar(w * h, 0);
    vector<unsigned char> dst_simd(w * h, 0);
    
    cout << "========================================" << endl;
    cout << "Оператор Превита (выделение границ)" << endl;
    cout << "Размер изображения: " << w << "x" << h << endl;
    cout << "========================================" << endl;
    
    // Замер времени (скалярная версия)
    cout << "Запуск скалярной версии..." << endl;
    clock_t start_scalar = clock();
    prewitt_scalar(src.data(), dst_scalar.data(), w, h);
    clock_t end_scalar = clock();
    double time_scalar = static_cast<double>(end_scalar - start_scalar) / CLOCKS_PER_SEC;
    
    // Замер времени (SIMD версия)
    cout << "Запуск SIMD (AVX2) версии..." << endl;
    clock_t start_simd = clock();
    prewitt_simd_avx2(src.data(), dst_simd.data(), w, h);
    clock_t end_simd = clock();
    double time_simd = static_cast<double>(end_simd - start_simd) / CLOCKS_PER_SEC;
    
    // Проверка корректности
    bool correct = true;
    for (int i = 0; i < w * h; ++i) {
        if (dst_scalar[i] != dst_simd[i]) {
            correct = false;
            break;
        }
    }
    
    // Вывод результатов
    cout << "----------------------------------------" << endl;
    cout << "Скалярная версия:    " << time_scalar << " сек" << endl;
    cout << "SIMD (AVX2) версия:  " << time_simd << " сек" << endl;
    cout << "Ускорение:           " << (time_scalar / time_simd) << "x" << endl;
    cout << "Проверка:            " << (correct ? "ПРОЙДЕНА (результаты совпадают)" : "ОШИБКА (несовпадение)") << endl;
    cout << "========================================" << endl;
    
    return 0;
}
