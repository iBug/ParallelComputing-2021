#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int N;
    float *A = NULL, *R = NULL;
    {
        const char *filename;
        if (argc >= 2) {
            filename = argv[1];
        } else {
            filename = "dataIn.txt";
        }
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            fprintf(stderr, "Cannot open file \"%s\"\n", filename);
            return 1;
        } else {
            int N2;
            fscanf(fp, "%d%d", &N, &N2);
            if (N != N2 || N <= 1) {
                fprintf(stderr, "Invalid matrix size\n");
                return 1;
            }
            A = malloc(2 * N * N * sizeof *A);
            R = malloc(2 * N * N * sizeof *R);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    fscanf(fp, "%f", R + 2 * i * N + j);
                    R[2 * i * N + N + j] = (float)(i == j);
                }
            }
            fclose(fp);
        }
    }

    int loops = 1;
    if (argc > 3) {
        loops = atoi(argv[3]);
        if (loops < 1) {
            fprintf(stderr, "Invalid number of loops\n");
            return 1;
        }
    }
    fprintf(stderr, "Running %d iteration%s\n", loops, loops == 1 ? "" : "s");

    double total_time = 0.;
    for (int loop = 0; loop < loops; loop++) {
        // Reset A and Q
        memcpy(A, R, 2 * N * N * sizeof *A);
        const double start = omp_get_wtime();

        for (int j = 0; j < N; j++) {
            for (int i = j + 1; i < N; i++) {
                float sq = sqrtf(A[2 * j * N + j] * A[2 * j * N + j] + A[2 * i * N + j] * A[2 * i * N + j]);
                float c = A[2 * j * N + j] / sq;
                float s = A[2 * i * N + j] / sq;
                for (int k = 0; k < 2 * N; k++) {
                    float aj = c * A[2 * j * N + k] + s * A[2 * i * N + k];
                    float ai = -s * A[2 * j * N + k] + c * A[2 * i * N + k];
                    A[2 * j * N + k] = aj;
                    A[2 * i * N + k] = ai;
                }
            }
        }

        // Transpose Q
        for (int j = 0; j < N; j++) {
            for (int i = j + 1; i < N; i++) {
                float t = A[2 * i * N + N + j];
                A[2 * i * N + N + j] = A[2 * j * N + N + i];
                A[2 * j * N + N + i] = t;
            }
        }

        const double end = omp_get_wtime();
        total_time += end - start;
    }
    fprintf(stderr, "Average processing time: %.3lfs\n", total_time / loops);

    const char *filename;
    if (argc >= 3) {
        filename = argv[2];
    } else {
        filename = "dataOut.txt";
    }
    FILE *fp = fopen(filename, "w");

    // Output R
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%f", A[2 * i * N + j]);
            if (j < N - 1)
                fputc('\t', fp);
        }
        fputc('\n', fp);
    }
    fputc('\n', fp);

    // Output Q
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%f", A[2 * i * N + N + j]);
            if (j < N - 1)
                fputc('\t', fp);
        }
        fputc('\n', fp);
    }

    fclose(fp);
    free(A);
    free(R);
    return 0;
}
