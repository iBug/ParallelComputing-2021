#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int popcnt(uint32_t i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = (i + (i >> 4)) & 0x0F0F0F0F;
    return (i * 0x01010101) >> 24;
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (popcnt(size) != 1) {
        MPI_Finalize();
        if (rank == 0)
            fprintf(stderr, "Invalid MPI size %d\n", size);
        return 1;
    }

    // Initialize n and sanity check
    int n;
    if (rank == 0) {
        scanf("%d", &n);
        if (popcnt(n) != 1) {
            fprintf(stderr, "Invalid number %d\n", n);
            n = 0;
        } else if (n < size) {
            fprintf(stderr, "Number %d must not be smaller than MPI size %d\n", n, size);
            n = 0;
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return 0;
    }

    // Read data and scatter
    const int local_n = n / size;
    int *A = malloc(local_n * sizeof(*A)), *data = NULL;
    if (rank == 0) {
        data = malloc(n * sizeof(*data));
        for (int i = 0; i < n; i++)
            scanf("%d", &data[i]);
    }
    MPI_Scatter(data, local_n, MPI_INT, A, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local sum
    int sum = 0;
    for (int i = 0; i < local_n; i++)
        sum += A[i];
    free(A);

    // Butterfly exchange & add
    for (int i = 1; i < size; i <<= 1) {
        int value;
        MPI_Sendrecv(&sum, 1, MPI_INT, rank ^ i, 0, &value, 1, MPI_INT, rank ^ i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum += value;
    }

    // Print result
    MPI_Gather(&sum, 1, MPI_INT, data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < size; i++)
            printf("Sum[%d] = %d\n", i, data[i]);
        free(data);
    }
    MPI_Finalize();
    return 0;
}
