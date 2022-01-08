#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for I=1 to N do
// A(I) = B(I) + C(I+1);
// C(I) = A(I) * D(I);
// end for

// A(1:N) = B(1:N) + C(2:N + 1)
// C(1:N) = A(1:N) * D(1:N)

#define N 1024

void f_ref(int *a, int *b, int *c, int *d) {
    for (int i = 1; i <= N; i++) {
        a[i] = b[i] + c[i + 1];
        c[i] = a[i] * d[i];
    }
}

void f_vec(int *a, int *b, int *c, int *d) {
#pragma omp simd
    for (int i = 1; i <= N; i++) {
        a[i] = b[i] + c[i + 1];
    }

#pragma omp simd
    for (int i = 1; i <= N; i++) {
        c[i] = a[i] * d[i];
    }
}

void init(int *a, int size) {
    for (int i = 0; i < size; i++)
        a[i] = rand();
}

int main(void) {
    srand((unsigned)time(NULL));
    double start_time, end_time;

    int a[N + 1], b[N + 1], c[N + 2], d[N + 1];
    int a1[N + 1], b1[N + 1], c1[N + 2], d1[N + 1];
    int a2[N + 1], b2[N + 1], c2[N + 2], d2[N + 1];
    init(a, sizeof a / sizeof *a);
    init(b, sizeof b / sizeof *b);
    init(c, sizeof c / sizeof *c);
    init(d, sizeof d / sizeof *d);

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
    f_vec(a2, b2, c2, d2);
    end_time = omp_get_wtime();
    printf("Vector time: %.6lf\n", end_time - start_time);

    if (memcmp(a1, a2, sizeof a) || memcmp(c1, c2, sizeof c))
        printf("Invalid result\n");
    return 0;
}
