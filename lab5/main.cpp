#include <iostream>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <mmintrin.h>
#include <x86intrin.h>

using namespace std;
using namespace chrono;

const int N = 1024;
const float ALPHA = 0.3f;
const float BETA = 1.0f - ALPHA;  // 0.7
const int ITERATIONS = 1000000;   // для замера времени

// 1. Стандартный C++
void blend_cpp(const int8_t* A, const int8_t* B, int8_t* C, int n, float alpha) {
    float beta = 1.0f - alpha;
    for (int i = 0; i < n; ++i) {
        float val = A[i] * alpha + B[i] * beta;
        C[i] = static_cast<int8_t>(round(val));
    }
}

// 2. MMX (64-bit)
void blend_mmx(const int8_t* A, const int8_t* B, int8_t* C, int n, float alpha, float beta) {
    // MMX не имеет операций с плавающей точкой, поэтому используем фиксированную точку
    // alpha = 0.3 = 0.3 * 256 = 76.8 -> 77
    // beta = 0.7 = 0.7 * 256 = 179.2 -> 179
    int16_t alpha_fixed = static_cast<int16_t>(alpha * 256);
    int16_t beta_fixed = static_cast<int16_t>(beta * 256);
    
    __m64 v_alpha = _mm_set1_pi16(alpha_fixed);
    __m64 v_beta = _mm_set1_pi16(beta_fixed);
    __m64 v_zero = _mm_setzero_si64();
    
    for (int i = 0; i < n; i += 4) {  // 4 int16_t (8 int8_t -> распаковка)
        // загружаем 4 байта (половина от 8, так как MMX маленький)
        __m64 a = *reinterpret_cast<const __m64*>(&A[i]);
        __m64 b = *reinterpret_cast<const __m64*>(&B[i]);
        
        // знаковое расширение до 16-bit
        __m64 sign_a = _mm_cmpgt_pi8(v_zero, a);
        __m64 a_lo = _mm_unpacklo_pi8(a, sign_a);
        
        __m64 sign_b = _mm_cmpgt_pi8(v_zero, b);
        __m64 b_lo = _mm_unpacklo_pi8(b, sign_b);
        
        // умножение a * alpha_fixed и b * beta_fixed
        __m64 a_mul = _mm_mullo_pi16(a_lo, v_alpha);
        __m64 b_mul = _mm_mullo_pi16(b_lo, v_beta);
        
        // сложение (a*alpha + b*beta)
        __m64 sum = _mm_add_pi16(a_mul, b_mul);
        
        // деление на 256 (сдвиг вправо на 8) и упаковка обратно в 8-bit
        __m64 shifted = _mm_srli_pi16(sum, 8);
        __m64 packed = _mm_packs_pi16(shifted, shifted);
        
        // сохранение 4 байт
        *reinterpret_cast<int32_t*>(&C[i]) = _mm_cvtsi64_si32(packed);
    }
    _mm_empty();
}

// 3. SSE2 (128-bit) - полный векторный
void blend_sse2(const int8_t* A, const int8_t* B, int8_t* C, int n, float alpha, float beta) {
    // преобразуем alpha и beta в 16-bit фиксированную точку с масштабом 256
    __m128i v_alpha = _mm_set1_epi16(static_cast<int16_t>(alpha * 256));
    __m128i v_beta = _mm_set1_epi16(static_cast<int16_t>(beta * 256));
    __m128i zero = _mm_setzero_si128();
    
    for (int i = 0; i < n; i += 16) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&A[i]));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[i]));
        
        // знаковое расширение до 16-bit
        __m128i sign_a = _mm_cmpgt_epi8(zero, a);
        __m128i a_lo = _mm_unpacklo_epi8(a, sign_a);
        __m128i a_hi = _mm_unpackhi_epi8(a, sign_a);
        
        __m128i sign_b = _mm_cmpgt_epi8(zero, b);
        __m128i b_lo = _mm_unpacklo_epi8(b, sign_b);
        __m128i b_hi = _mm_unpackhi_epi8(b, sign_b);
        
        // умножение и сложение
        __m128i a_mul_lo = _mm_mullo_epi16(a_lo, v_alpha);
        __m128i a_mul_hi = _mm_mullo_epi16(a_hi, v_alpha);
        __m128i b_mul_lo = _mm_mullo_epi16(b_lo, v_beta);
        __m128i b_mul_hi = _mm_mullo_epi16(b_hi, v_beta);
        
        __m128i sum_lo = _mm_add_epi16(a_mul_lo, b_mul_lo);
        __m128i sum_hi = _mm_add_epi16(a_mul_hi, b_mul_hi);
        
        // деление на 256 (сдвиг) и упаковка с насыщением
        __m128i shifted_lo = _mm_srai_epi16(sum_lo, 8);
        __m128i shifted_hi = _mm_srai_epi16(sum_hi, 8);
        __m128i result = _mm_packs_epi16(shifted_lo, shifted_hi);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&C[i]), result);
    }
}

// 4. AVX2→AVX→AVX2 (каскад) - ключевая реализация по варианту 21
void blend_avx2_avx(const int8_t* A, const int8_t* B, int8_t* C, int n, float alpha, float beta) {
    // Используем float для большей точности (как в задании)
    __m256 v_alpha = _mm256_set1_ps(alpha);
    __m256 v_beta = _mm256_set1_ps(beta);
    
    for (int i = 0; i < n; i += 32) {
        // Загружаем 32 байта (256 бит) через AVX2
        __m256i a_i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[i]));
        __m256i b_i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[i]));
        
        // --- КАСКАД: AVX2 -> AVX (разбиваем на две 128-битные половины) ---
        __m128i a_lo128 = _mm256_castsi256_si128(a_i);      // младшие 128 бит
        __m128i a_hi128 = _mm256_extracti128_si256(a_i, 1); // старшие 128 бит
        __m128i b_lo128 = _mm256_castsi256_si128(b_i);
        __m128i b_hi128 = _mm256_extracti128_si256(b_i, 1);
        
        // Расширяем 8-bit -> 32-bit float (используем SSE/AVX инструкции)
        // Для этого сначала 8-bit -> 16-bit, затем 16-bit -> 32-bit float
        __m128i a_lo16 = _mm_cvtepi8_epi16(a_lo128);
        __m128i a_hi16 = _mm_cvtepi8_epi16(a_hi128);
        __m128i b_lo16 = _mm_cvtepi8_epi16(b_lo128);
        __m128i b_hi16 = _mm_cvtepi8_epi16(b_hi128);
        
        // 16-bit int -> 32-bit float
        __m128 a_lo_f = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_lo16));
        __m128 a_hi_f = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(a_hi16, 0x00)));
        __m128 b_lo_f = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(b_lo16));
        __m128 b_hi_f = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_shuffle_epi32(b_hi16, 0x00)));
        
        // Смешивание в float: C = A*alpha + B*beta
        __m128 c_lo_f = _mm_add_ps(_mm_mul_ps(a_lo_f, _mm256_castps256_ps128(v_alpha)),
                                    _mm_mul_ps(b_lo_f, _mm256_castps256_ps128(v_beta)));
        __m128 c_hi_f = _mm_add_ps(_mm_mul_ps(a_hi_f, _mm256_castps256_ps128(v_alpha)),
                                    _mm_mul_ps(b_hi_f, _mm256_castps256_ps128(v_beta)));
        
        // Округление и преобразование обратно в int8_t
        __m128i c_lo_i = _mm_cvtps_epi32(c_lo_f);
        __m128i c_hi_i = _mm_cvtps_epi32(c_hi_f);
        
        // Упаковка 32-bit -> 16-bit -> 8-bit
        __m128i c_lo_i16 = _mm_packs_epi32(c_lo_i, c_lo_i);
        __m128i c_hi_i16 = _mm_packs_epi32(c_hi_i, c_hi_i);
        __m128i c_lo_i8 = _mm_packs_epi16(c_lo_i16, c_lo_i16);
        __m128i c_hi_i8 = _mm_packs_epi16(c_hi_i16, c_hi_i16);
        
        // --- Обратный каскад: AVX -> AVX2 (собираем обратно в 256 бит) ---
        __m256i result = _mm256_set_m128i(c_hi_i8, c_lo_i8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[i]), result);
    }
}

// Функция замера времени
template<typename Func>
double measure_time(Func f, const int8_t* A, const int8_t* B, int8_t* C, int n, 
                     float alpha, float beta, int iterations) {
    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        f(A, B, C, n, alpha, beta);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)iterations;
}

int main() {
    alignas(32) int8_t A[N], B[N], C_cpp[N], C_mmx[N], C_sse2[N], C_avx[N];
    
    // Инициализация
    for (int i = 0; i < N; ++i) {
        A[i] = (i * 7) % 256 - 128;
        B[i] = (i * 13) % 256 - 128;
    }
    
    const float alpha = 0.3f;
    const float beta = 0.7f;
    
    // Проверка корректности
    blend_cpp(A, B, C_cpp, N, alpha);
    blend_mmx(A, B, C_mmx, N, alpha, beta);
    blend_sse2(A, B, C_sse2, N, alpha, beta);
    blend_avx2_avx(A, B, C_avx, N, alpha, beta);
    
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (C_cpp[i] != C_mmx[i] || C_cpp[i] != C_sse2[i] || C_cpp[i] != C_avx[i]) {
            cout << "Mismatch at " << i << ": C++=" << (int)C_cpp[i] 
                 << " MMX=" << (int)C_mmx[i] 
                 << " SSE2=" << (int)C_sse2[i] 
                 << " AVX=" << (int)C_avx[i] << endl;
            ok = false;
            break;
        }
    }
    if (ok) cout << "[OK] All results match!" << endl;
    
    // Замеры производительности
    cout << "\n--- Performance (ns per call, N=1024, iterations=" << ITERATIONS << ") ---" << endl;
    double t_cpp = measure_time(blend_cpp, A, B, C_cpp, N, alpha, beta, ITERATIONS);
    double t_mmx = measure_time(blend_mmx, A, B, C_mmx, N, alpha, beta, ITERATIONS);
    double t_sse2 = measure_time(blend_sse2, A, B, C_sse2, N, alpha, beta, ITERATIONS);
    double t_avx = measure_time(blend_avx2_avx, A, B, C_avx, N, alpha, beta, ITERATIONS);
    
    cout << "C++          : " << t_cpp << " ns\n";
    cout << "MMX          : " << t_mmx << " ns\n";
    cout << "SSE2         : " << t_sse2 << " ns\n";
    cout << "AVX2→AVX→AVX2: " << t_avx << " ns\n";
    
    cout << "\n--- Speedup vs C++ ---\n";
    cout << "MMX          : " << (t_cpp / t_mmx) << "x\n";
    cout << "SSE2         : " << (t_cpp / t_sse2) << "x\n";
    cout << "AVX2→AVX→AVX2: " << (t_cpp / t_avx) << "x\n";
    
    return 0;
}
