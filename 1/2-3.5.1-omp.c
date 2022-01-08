#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for i = 1 to 100 do //循环 1
//   A[i] = A[i] + B[i-1]; // S1
//   B[i] = C[i-1] * 2 ;  // S2
//   C[i] = 1 / B[i] ;    // S3
//   D[i] = C[i] * C[i] ; // S4
// endfor

// for i = 1 to 100 do //循环 1
//   B[i] = C[i-1] * 2;
//   C[i] = 1 / B[i];
// endfor
// A[1:100] = A[1:100] + B[0:99]
// D[1:100] = C[1:100] * C[1:100]

void f_ref(double *a, double *b, double *c, double *d) {
    for (int i = 1; i <= 100; i++) {
        a[i] = a[i] + b[i - 1];
        b[i] = c[i - 1] * 2;
        c[i] = 1 / b[i];
        d[i] = c[i] * c[i];
    }
}

void f_opt(double *a, double *b, double *c, double *d) {
    for (int i = 1; i <= 100; i++) {
        b[i] = c[i - 1] * 2;
        c[i] = 1 / b[i];
    }

#pragma omp simd
    for (int i = 1; i <= 100; i++) {
        a[i] = a[i] + b[i - 1];
    }

#pragma omp simd
    for (int i = 1; i <= 100; i++) {
        d[i] = c[i] * c[i];
    }
}

void init(double *a, int size) {
    for (int i = 0; i < size; i++)
        a[i] = rand();
}

int main(void) {
    srand((unsigned)time(NULL));
    double start_time, end_time;

    double a[101], b[101], c[101], d[101];
    double a1[101], b1[101], c1[101], d1[101];
    double a2[101], b2[101], c2[101], d2[101];
    init(a, 101);
    init(b, 101);
    init(c, 101);
    init(d, 101);

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
