#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int N;
    float *A = NULL, *R = NULL, *Q = NULL;
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
            A = malloc(N * N * sizeof *A);
            R = malloc(N * N * sizeof *R);
            Q = malloc(N * N * sizeof *Q);
            for (int i = 0; i < N * N; i++)
                fscanf(fp, "%f", R + i);
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
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                Q[i * N + j] = (float)(i == j);
            }
        }
        memcpy(A, R, N * N * sizeof *A);
        const double start = omp_get_wtime();

        for (int j = 0; j < N; j++) {
            for (int i = j + 1; i < N; i++) {
                float sq = sqrtf(A[j * N + j] * A[j * N + j] + A[i * N + j] * A[i * N + j]);
                float c = A[j * N + j] / sq;
                float s = A[i * N + j] / sq;
                for (int k = 0; k < N; k++) {
                    float aj = c * A[j * N + k] + s * A[i * N + k];
                    float qj = c * Q[j * N + k] + s * Q[i * N + k];
                    float ai = -s * A[j * N + k] + c * A[i * N + k];
                    float qi = -s * Q[j * N + k] + c * Q[i * N + k];
                    A[j * N + k] = aj;
                    Q[j * N + k] = qj;
                    A[i * N + k] = ai;
                    Q[i * N + k] = qi;
                }
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
    if (fp) {
        // Output R
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%f", A[i * N + j]);
                if (j < N - 1)
                    fputc('\t', fp);
            }
            fputc('\n', fp);
        }
        fputc('\n', fp);

        // Output Q
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fp, "%f", Q[j * N + i]);
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
    free(Q);
    free(R);
    return 0;
}
