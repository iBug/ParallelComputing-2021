#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

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
        fprintf(stderr, "Invalid number\n");
        return 1;
    }
    int *A = malloc(n * sizeof(*A));
    for (int i = 0; i < n; i++) {
        scanf("%d", &A[i]);
    }

    // Reduce sum
    for (int i = 1; i < n; i <<= 1) {
        #pragma omp parallel for
        for (int j = 0; j < n; j += i << 1) {
            A[j] += A[j + i];
        }
    }

    // Broadcast
    for (int i = n << 1; i > 0; i >>= 1) {
        #pragma omp parallel for
        for (int j = 0; j < n; j += i << 1) {
            A[j + i] = A[j];
        }
    }

    for (int i = 0; i < n; i++) {
        printf("A[%d] = %d\n", i, A[i]);
    }
    free(A);
    return 0;
}
