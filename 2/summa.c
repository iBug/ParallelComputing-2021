#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Need row size and column size\n");
        MPI_Finalize();
        return 1;
    }

    const int row_size = atoi(argv[1]);
    const int col_size = atoi(argv[2]);
    if (row_size * col_size != size) {
        if (rank == 0)
            fprintf(stderr, "Need %d process%s\n", row_size * col_size, row_size * col_size > 1 ? "es" : "");
        MPI_Finalize();
        return 1;
    }
    if (row_size < 1 || col_size < 1) {
        if (rank == 0)
            fprintf(stderr, "Invalid row or column size\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm comm_cart, comm_row, comm_col;
    {
        const int dims[2] = {row_size, col_size},
                  dims_row[2] = {1, 0},
                  dims_col[2] = {0, 1},
                  periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
        MPI_Cart_sub(comm_cart, dims_row, &comm_row);
        MPI_Cart_sub(comm_cart, dims_col, &comm_col);
    }

    float *A, *B, *C;
    int M = 0, N = 0, K = 0;
    {
        const char *filename;
        if (argc > 5) {
            filename = argv[5];
        } else {
            filename = "dataIn.txt";
        }
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            fprintf(stderr, "Cannot open file \"%s\"\n", filename);
            return 1;
        } else {
            fscanf(fp, "%d%d%d", &M, &K, &N);
            if (M < 1 || N < 1 || K < 1 || M % row_size || N % col_size || K % row_size || K % col_size) {
                fprintf(stderr, "Invalid matrix size\n");
                M = N = K = 0;
            }
            for (int i = 0; i < M * K; i++)
                fscanf(fp, "%f", A + i);
            for (int i = 0; i < K * N; i++)
                fscanf(fp, "%f", B + i);
            fclose(fp);
        }
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!M || !N || !K) {
        MPI_Finalize();
        return 1;
    }
    const int chunkAH = M / row_size, chunkAW = K / col_size, chunkBH = K / row_size, chunkBW = N / col_size;

    int loops = 1;
    if (argc > 4) {
        loops = atoi(argv[4]);
        if (loops < 1) {
            fprintf(stderr, "Invalid number of loops\n");
            return 1;
        }
    }
    fprintf(stderr, "Running %d iteration%s\n", loops, loops == 1 ? "" : "s");

    double total_time = 0.;
    for (int loop = 0; loop < loops; loop++) {
        // Reset A and B
        memcpy(A, A, M * K * sizeof *A);
        memcpy(B, B, K * N * sizeof *B);
        const double start = MPI_Wtime();

        for (int j = 0; j < N; j++) {
            for (int i = j + 1; i < N; i++) {
            }
        }

        const double end = MPI_Wtime();
        total_time += end - start;
    }
    if (rank == 0)
        fprintf(stderr, "Average processing time: %.3lfs\n", total_time / loops);

    const char *filename;
    if (argc > 6) {
        filename = argv[6];
    } else {
        filename = "dataOut.txt";
    }
    FILE *fp = fopen(filename, "w");
    if (fp) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%f", C[i * N + j]);
                if (j < N - 1)
                    fputc('\t', fp);
            }
            fputc('\n', fp);
        }
        fclose(fp);
    } else {
        fprintf(stderr, "Cannot open file \"%s\"\n", filename);
    }
    free(A);
    MPI_Finalize();
    return 0;
}
