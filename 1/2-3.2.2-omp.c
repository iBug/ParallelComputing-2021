#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for i = 1 to 100 do // 循环 2 N 是常量
//   X[i] = Y[i] + 10; // 语句 S1
//   for j = 1 to 100 do
//     B[j] = A[j, N]; // 语句 S2
//     for k = 1 to 100 do
//       A[j+1, k] = B[j] + C[j, k]; // 语句 S3
//     endfor // loop-k
//     Y[i+j] = A[j+1, N]; // 语句 S4
//   endfor // loop-j
// endfor // loop-i

// for j = 1 to 100 do
//   B[j] = A[j, N]
//   A[j+1, 1:100] = B[j] + C[j, 1:100]
// endfor
// Y[2:100] = A[2, N]
// Y[101:200] = A[2:101, N]
// X[1:100] = Y[1:100] + 10

#define N 99

void f_ref(int a[][101], int b[], int c[][101], int x[], int y[]) {
    for (int i = 1; i <= 100; i++) {
        x[i] = y[i] + 10;
        for (int j = 1; j <= 100; j++) {
            b[j] = a[j][N];
            for (int k = 1; k <= 100; k++) {
                a[j + 1][k] = b[j] + c[j][k];
            }
            y[i + j] = a[j + 1][N];
        }
    }
}

void f_opt(int a[][101], int b[], int c[][101], int x[], int y[]) {
    for (int j = 1; j <= 100; j++) {
        b[j] = a[j][N];
#pragma omp simd
        for (int i = 1; i <= 100; i++) {
            a[j + 1][i] = b[j] + c[j][i];
        }
    }
#pragma omp simd
    for (int i = 2; i <= 100; i++) {
        y[i] = a[2][N];
    }
#pragma omp simd
    for (int i = 2; i <= 101; i++) {
        y[i + 99] = a[i][N];
    }
#pragma omp simd
    for (int i = 1; i <= 100; i++) {
        x[i] = y[i] + 10;
    }
}

void init(int *a, int size) {
    for (int i = 0; i < size; i++)
        a[i] = rand();
}

int main(void) {
    srand((unsigned)time(NULL));
    double start_time, end_time;

    int a[102][101], b[101], c[101][101], x[101], y[201];
    int a1[102][101], b1[101], c1[101][101], x1[101], y1[201];
    int a2[102][101], b2[101], c2[101][101], x2[101], y2[201];
    init((int *)a, sizeof(a) / sizeof(a[0][0]));
    init(b, sizeof(b) / sizeof(b[0]));
    init((int *)c, sizeof(c) / sizeof(c[0][0]));
    init(x, sizeof(x) / sizeof(x[0]));
    init(y, sizeof(y) / sizeof(y[0]));

    memcpy(a1, a, sizeof a);
    memcpy(b1, b, sizeof b);
    memcpy(c1, c, sizeof c);
    memcpy(x1, x, sizeof x);
    memcpy(y1, y, sizeof y);
    start_time = omp_get_wtime();
    f_ref(a1, b1, c1, x1, y1);
    end_time = omp_get_wtime();
    printf("Original time: %.6lf\n", end_time - start_time);

    memcpy(a2, a, sizeof a);
    memcpy(b2, b, sizeof b);
    memcpy(c2, c, sizeof c);
    memcpy(x2, x, sizeof x);
    memcpy(y2, y, sizeof y);
    start_time = omp_get_wtime();
    f_opt(a2, b2, c2, x2, y2);
    end_time = omp_get_wtime();
    printf("Optimized time: %.6lf\n", end_time - start_time);

    if (memcmp(a1, a2, sizeof a) || memcmp(b1, b2, sizeof b) || memcmp(x1, x2, sizeof x) || memcmp(y1, y2, sizeof y))
        printf("Invalid result\n");
    return 0;
}
