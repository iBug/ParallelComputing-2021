#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N; // Matrix size
    float *A = NULL;
    if (rank == 0) {
        const char *filename;
        if (argc >= 2) {
            filename = argv[1];
        } else {
            filename = "dataIn.txt";
        }
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            fprintf(stderr, "Cannot open file \"%s\"\n", filename);
            N = 0;
        } else {
            int N2;
            fscanf(fp, "%d%d", &N, &N2);
            if (N != N2 || N <= 1) {
                fprintf(stderr, "Invalid matrix size\n");
                N = 0;
            } else if (N < size) {
                fprintf(stderr, "Matrix size is too small for %d processes\n", size);
                N = 0;
            }
            A = malloc(N * N * sizeof *A);
            for (int i = 0; i < N * N; i++)
                fscanf(fp, "%f", A + i);
            fclose(fp);
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N == 0) {
        MPI_Finalize();
        return 1;
    }

    int loops = 1;
    if (argc > 3) {
        loops = atoi(argv[3]);
        if (loops < 1) {
            if (rank == 0)
                fprintf(stderr, "Invalid number of loops\n");
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0)
        fprintf(stderr, "Running %d iteration%s\n", loops, loops == 1 ? "" : "s");

    double total_time = 0.;
    const int m = N / size + (N % size > 0);
    float *a = malloc(N * m * sizeof *a);
    float *buf = malloc(N * sizeof *buf);

    MPI_Datatype Mtype; // for packed MPI transmission
    MPI_Type_contiguous(N, MPI_FLOAT, &Mtype);
    MPI_Type_commit(&Mtype);

    for (int loop = 0; loop < loops; loop++) {
        const double start = MPI_Wtime();

        const int scatter_unit = N * size;
        for (int i = 0; i < N / size; i++) {
            MPI_Scatter(A + i * scatter_unit, 1, Mtype, a + i * N, 1, Mtype, 0, MPI_COMM_WORLD);
        }
        if (N % size && rank == 0)
            memcpy(a + (N / size) * N, A + (N / size) * scatter_unit, N * sizeof(*a));
        if (N % size > 1) {
            if (rank == 0) {
#pragma omp parallel for
                for (int i = 1; i < N % size; i++) {
                    MPI_Send(A + (N / size) * scatter_unit + i * N, 1, Mtype, i, 0, MPI_COMM_WORLD);
                }
            } else if (rank < (N / size)) {
                MPI_Recv(a + (N / size) * N, 1, Mtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Main computation
        for (int i = 0; i < N; i++) {
            const int round = i / size; // original "i"
            const int mr = i % size;    // main row
            float *f;
            if (rank == mr) {
                f = a + round * N;
            } else {
                f = buf;
            }
            MPI_Bcast(f + i, N - i, MPI_FLOAT, mr, MPI_COMM_WORLD);

            if (rank <= mr) {
#pragma omp parallel for
                for (int k = round + 1; k < m; k++) {
                    a[k * N + i] /= f[i];
                    // #pragma omp simd
                    for (int w = i + 1; w < N; w++)
                        a[k * N + w] -= f[w] * a[k * N + i];
                }
            } else {
#pragma omp parallel for
                for (int k = round; k < m; k++) {
                    a[k * N + i] /= f[i];
                    // #pragma omp simd
                    for (int w = i + 1; w < N; w++)
                        a[k * N + w] -= f[w] * a[k * N + i];
                }
            }
        }

        for (int i = 0; i < N / size; i++) {
            MPI_Gather(a + i * N, 1, Mtype, A + i * scatter_unit, 1, Mtype, 0, MPI_COMM_WORLD);
        }
        if (N % size && rank == 0)
            memcpy(A + (N / size) * scatter_unit, a + (N / size) * N, N * sizeof(*a));
        if (N % size > 1) {
            if (rank == 0) {
                for (int i = 1; i < N % size; i++) {
                    MPI_Recv(A + (N / size) * scatter_unit + i * N, 1, Mtype, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else {
                MPI_Send(a + (N / size) * N, 1, Mtype, 0, 0, MPI_COMM_WORLD);
            }
        }
        const double end = MPI_Wtime();
        total_time += end - start;
    }
    MPI_Type_free(&Mtype);

    if (rank == 0) {
        fprintf(stderr, "Average processing time: %.6lfs\n", total_time / loops);

        const char *filename;
        if (argc >= 3) {
            filename = argv[2];
        } else {
            filename = "dataOut.txt";
        }
        FILE *fp = fopen(filename, "w");

        // Output L
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i > j)
                    fprintf(fp, "%f", A[i * N + j]);
                else if (i == j)
                    fprintf(fp, "%f", 1.F);
                else
                    fprintf(fp, "%f", 0.F);
                if (j < N - 1)
                    fputc('\t', fp);
            }
            fputc('\n', fp);
        }
        fputc('\n', fp);

        // Output U
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i > j)
                    fprintf(fp, "%f", 0.F);
                else
                    fprintf(fp, "%f", A[i * N + j]);
                if (j < N - 1)
                    fputc('\t', fp);
            }
            fputc('\n', fp);
        }

        fclose(fp);
        free(A);
    }
    free(a);
    free(buf);

    MPI_Finalize();
    return 0;
}
