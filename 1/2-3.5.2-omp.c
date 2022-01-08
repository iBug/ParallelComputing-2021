#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for i = 1 to 999 do // 循环 2
//   A[i] = B[i] + C[i];                  // S1
//   D[i] = ( A[i] + A[ 999-i+1 ] ) / 2 ; // S2
// endfor

// A[1:500] = B[1:500] + C[1:500]
// D[1:500] = (A[1:500] + A[999:500]) / 2
// A[501:999] = B[501:999] + C[501:999]
// D[501:999] = (A[501:999] + A[499:1]) / 2

void f_ref(int *a, int *b, int *c, int *d) {
    for (int i = 1; i <= 999; i++) {
        a[i] = b[i] + c[i];
        d[i] = (a[i] + a[999 - i + 1]) / 2;
    }
}

void f_opt(int *a, int *b, int *c, int *d) {
#pragma omp simd
    for (int i = 1; i <= 500; i++) {
        a[i] = b[i] + c[i];
    }
#pragma omp simd
    for (int i = 1; i <= 500; i++) {
        d[i] = (a[i] + a[1000 - i]) / 2;
    }
#pragma omp simd
    for (int i = 501; i <= 999; i++) {
        a[i] = b[i] + c[i];
    }
#pragma omp simd
    for (int i = 501; i <= 999; i++) {
        d[i] = (a[i] + a[1000 - i]) / 2;
    }
}

void init(int *a, int size) {
    for (int i = 0; i < size; i++)
        a[i] = rand();
}

int main(void) {
    srand((unsigned)time(NULL));
    double start_time, end_time;

    int a[1000], b[1000], c[1000], d[1000];
    int a1[1000], b1[1000], c1[1000], d1[1000];
    int a2[1000], b2[1000], c2[1000], d2[1000];
    init(a, 1000);
    init(b, 1000);
    init(c, 1000);
    init(d, 1000);

    memcpy(a1, a, sizeof a);
    memcpy(b1, b, sizeof b);
    memcpy(c1, c, sizeof c);
    memcpy(d1, d, sizeof d);
    start_time = omp_get_wtime();
    f_ref(a1, b1, c1, d1);
    end_time = omp_get_wtime();
    printf("Original time: %.6lf\n", end_time - start_time);

    memcpy(a2, a, sizeof a);
    memcpy(b2, b, sizeof b);
    memcpy(c2, c, sizeof c);
    memcpy(d2, d, sizeof d);
    start_time = omp_get_wtime();
    f_opt(a2, b2, c2, d2);
    end_time = omp_get_wtime();
    printf("Optimized time: %.6lf\n", end_time - start_time);

    if (memcmp(a1, a2, sizeof a) || memcmp(b1, b2, sizeof b) || memcmp(c1, c2, sizeof c) || memcmp(c1, c2, sizeof c))
        printf("Invalid result\n");
    return 0;
}
