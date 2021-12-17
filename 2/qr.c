#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int N;
    float *A, *Q, *R, *S; // S for reading input
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
        }
        int N2;
        fscanf(fp, "%d%d", &N, &N2);
        if (N != N2 || N <= 1) {
            fprintf(stderr, "Invalid matrix size\n");
            return 1;
        }
        A = malloc(N * N * sizeof *A);
        Q = malloc(N * N * sizeof *Q);
        R = malloc(N * N * sizeof *R);
        S = malloc(N * N * sizeof *S);
        for (int i = 0; i < N * N; i++)
            fscanf(fp, "%f", S + i);
        fclose(fp);
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

    omp_lock_t *lock = malloc(N * sizeof(*lock));
    for (int i = 0; i < N; i++)
        omp_init_lock(&lock[i]);

    double total_time = 0.;
    for (int loop = 0; loop < loops; loop++)
#pragma omp parallel
    {
        // Reset A and R
#pragma omp for schedule(static, 1)
        for (int i = 0; i < N; i++)
            omp_set_lock(&lock[i]);
#pragma omp single
        {
            memcpy(A, S, N * N * sizeof(*A));
            memset(Q, 0, N * N * sizeof(*Q));
            memset(R, 0, N * N * sizeof(*R));
        }
        const double start = omp_get_wtime();

        // First column of ( Q[][0] )
        if (omp_get_thread_num() == 0) {
            // Get ||A||
            float sum = 0.F;
            for (int i = 0; i < N; i++)
                sum += A[0 * N + i] * A[0 * N + i];
            R[0 * N + 0] = sqrtf(sum);
            for (int i = 0; i < N; i++)
                Q[0 * N + i] = A[0 * N + i] / R[0 * N + 0];
            omp_unset_lock(&lock[0]);
        }

        for (int i = 1; i < N; i++) {
            // Check if Q[][i-1] (the previous column) is computed.
            omp_set_lock(&lock[i - 1]);
            omp_unset_lock(&lock[i - 1]);

#pragma omp for schedule(static, 1) nowait
            for (int j = 0; j < N; j++) {
                if (j < i)
                    continue;

                for (int k = 0; k < N; k++)
                    R[j * N + (i - 1)] += Q[(i - 1) * N + k] * A[j * N + k];
                for (int k = 0; k < N; k++)
                    A[j * N + k] -= R[j * N + (i - 1)] * Q[(i - 1) * N + k];

                if (i == j) {
                    float sum = 0.F;
                    for (int k = 0; k < N; k++)
                        sum += A[i * N + k] * A[i * N + k];
                    R[i * N + i] = sqrtf(sum);
                    for (int k = 0; k < N; k++)
                        Q[i * N + k] = A[i * N + k] / R[i * N + i];
                    omp_unset_lock(&lock[i]);
                }
            }
        }

        const double end = omp_get_wtime();
#pragma omp single
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
                fprintf(fp, "%f", R[j * N + i]);
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
    free(S);
    free(lock);
    return 0;
}
