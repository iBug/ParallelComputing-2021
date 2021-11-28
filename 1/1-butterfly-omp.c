#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int popcnt(uint32_t i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = (i + (i >> 4)) & 0x0F0F0F0F;
    return (i * 0x01010101) >> 24;
}

int main() {
    int n;
    scanf("%d", &n);
    if (popcnt(n) != 1) {
        fprintf(stderr, "Invalid number %d\n", n);
        return 1;
    }
    int *A = malloc(n * sizeof(*A)),
        *B = malloc(n * sizeof(*B));
    for (int i = 0; i < n; i++) {
        scanf("%d", &A[i]);
    }

    // Reduce sum
    // Unfortunately we have to use two buffers,
    // otherwise it's impossible to perform "send/recv and add" without blocking
    for (int i = 1; i < n; i <<= 1) {
#pragma omp parallel for
        for (int j = 0; j < n; j++) {
            B[j] = A[j] + A[j ^ i];
        }
        int *T = A;
        A = B;
        B = T;
    }

    for (int i = 0; i < n; i++) {
        printf("A[%d] = %d\n", i, A[i]);
    }
    free(A);
    free(B);
    return 0;
}