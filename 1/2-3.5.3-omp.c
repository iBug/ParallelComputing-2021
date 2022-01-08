#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for i = 1 to 100 do // 循环 3
//   for j = 1 to 100 do
//     A[3*i+2*j, 2*j] = C[i,j] * 2 ; // S1
//     D[i,j] = A[i-j+6, i+j] ;       // S2
//   endfor
// endfor

// doall i = 1 to 100 do
//   A[3*i+2:3*i+200:2, 2:200:2] = C[i, 1:100] * 2
//   D[i, 1:100] = A[i+5:i-94, i+1:i+100]
// enddoall

#define OFFSET 93

void f_ref(int a[][201], int c[][101], int d[][201]) {
    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 100; j++) {
            a[3 * i + 2 * j][2 * j] = c[i][j] * 2;
            d[i][j] = a[i - j + 6][i + j];
        }
    }
}

void f_opt(int a[][201], int c[][101], int d[][201]) {
#pragma omp parallel for
    for (int i = 1; i <= 100; i++) {
#pragma omp simd
        for (int j = 1; j <= 100; j++) {
            a[3 * i + 2 * j][2 * j] = c[i][j] * 2;
        }
#pragma omp simd
        for (int j = 1; j <= 100; j++) {
            d[i][j] = a[i - j + 6][i + j];
        }
    }
}

void init(int *a, int size) {
    for (int i = 0; i < size; i++)
        a[i] = rand();
}

int main(void) {
    srand((unsigned)time(NULL));
    double start_time, end_time;

    int a[501 + OFFSET][201], c[101][101], d[101][201];
    int a1[501 + OFFSET][201], c1[101][101], d1[101][201];
    int a2[501 + OFFSET][201], c2[101][101], d2[101][201];
    init((int *)a, sizeof(a)/sizeof(a[0][0]));
    init((int *)c, sizeof(c)/sizeof(c[0][0]));
    init((int *)d, sizeof(d)/sizeof(d[0][0]));

    memcpy(a1, a, sizeof a);
    memcpy(c1, c, sizeof c);
    memcpy(d1, d, sizeof d);
    start_time = omp_get_wtime();
    f_ref(a1 + OFFSET, c1, d1);
    end_time = omp_get_wtime();
    printf("Original time: %.6lf\n", end_time - start_time);

    memcpy(a2, a, sizeof a);
    memcpy(c2, c, sizeof c);
    memcpy(d2, d, sizeof d);
    start_time = omp_get_wtime();
    f_opt(a2 + OFFSET, c2, d2);
    end_time = omp_get_wtime();
    printf("Optimized time: %.6lf\n", end_time - start_time);

    if (memcmp(a1, a2, sizeof a) || memcmp(d1, d2, sizeof d))
        printf("Invalid result\n");
    return 0;
}
