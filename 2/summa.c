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
    int loops = 1;
    if (argc > 4) {
        loops = atoi(argv[4]);
        if (loops < 1) {
            if (rank == 0)
                fprintf(stderr, "Invalid number of loops\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm comm_cart, comm_row, comm_col;
    int coords[2];
    {
        const int dims[2] = {row_size, col_size},
                  dims_row[2] = {0, 1},
                  dims_col[2] = {1, 0},
                  periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
        MPI_Cart_sub(comm_cart, dims_row, &comm_row);
        MPI_Cart_sub(comm_cart, dims_col, &comm_col);
        MPI_Cart_coords(comm_cart, rank, 2, coords);
    }

    float *A = NULL, *B = NULL, *C = NULL;
    int M = 0, N = 0, K = 0;
    if (rank == 0) {
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
            } else {
                // W = height, H = width, A = Area
                const int chunkAH = M / row_size, chunkAW = K / col_size,
                          chunkBH = K / row_size, chunkBW = N / col_size,
                          chunkAA = chunkAH * chunkAW, chunkBA = chunkBH * chunkBW;
                A = malloc(M * K * sizeof *A);
                B = malloc(K * N * sizeof *B);
                for (int i = 0; i < M * K; i++) {
                    int x = i % K, y = i / K,
                        cx = x / chunkAW, cy = y / chunkAH,         // coordinates of the chunk
                        chunkX = x % chunkAW, chunkY = y % chunkAH, // coordinates of item in the chunk
                        ci = cx + cy * col_size, chunkI = chunkX + chunkY * chunkAW;
                    fscanf(fp, "%f", A + ci * chunkAA + chunkI);
                }
                for (int i = 0; i < K * N; i++) {
                    // B is stored in column-major order
                    int x = i % N, y = i / N,
                        cx = x / chunkBW, cy = y / chunkBH,         // coordinates of the chunk
                        chunkX = x % chunkBW, chunkY = y % chunkBH, // coordinates of item in the chunk
                        ci = cx + cy * col_size, chunkI = chunkX * chunkBH + chunkY;
                    fscanf(fp, "%f", B + ci * chunkBA + chunkI);
                }
            }
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
    const int chunkAH = M / row_size, chunkAW = K / col_size,
              chunkBH = K / row_size, chunkBW = N / col_size,
              chunkCH = chunkAH, chunkCW = chunkBW,
              chunkAA = chunkAH * chunkAW, chunkBA = chunkBH * chunkBW, chunkCA = chunkCH * chunkCW;

    // Now scatter A and B
    float *myA = malloc(chunkAA * sizeof *myA), *myB = malloc(chunkBA * sizeof *myB), *myC = malloc(chunkCA * sizeof *myC);
    MPI_Scatter(A, chunkAA, MPI_FLOAT, myA, chunkAA, MPI_FLOAT, 0, comm_cart);
    MPI_Scatter(B, chunkBA, MPI_FLOAT, myB, chunkBA, MPI_FLOAT, 0, comm_cart);
    if (rank == 0) {
        free(A);
        free(B);
        A = B = NULL;
        fprintf(stderr, "Running %d iteration%s\n", loops, loops == 1 ? "" : "s");
    }

    float *localA = malloc(chunkAA * sizeof *localA),
          *localB = malloc(chunkBA * sizeof *localB);
    C = myC;
    double total_time = 0.;
    for (int loop = 0; loop < loops; loop++) {
        // Reset C
        memset(C, 0, chunkCA * sizeof *C);
        const double start = MPI_Wtime();
        int kstart = 0, iA = 0, iB = 0, haveA = -1, haveB = -1;
        // start of [k], index of A's and B's chunks, which chunks we currently have in localA and localB

        while (kstart < K) {
            A = iA == coords[1] ? myA : localA;
            B = iB == coords[0] ? myB : localB;

            if (haveA != iA) {
                MPI_Bcast(A, chunkAA, MPI_FLOAT, iA, comm_row);
                haveA = iA;
            }
            if (haveB != iB) {
                MPI_Bcast(B, chunkBA, MPI_FLOAT, iB, comm_col);
                haveB = iB;
            }
            int Aend = (1 + iA) * chunkAW,
                Bend = (1 + iB) * chunkBH,
                kend = Aend < Bend ? Aend : Bend,
                offsetAk = kstart - iA * chunkAW,
                offsetBk = kstart - iB * chunkBH;

            for (int i = 0; i < chunkCH; i++) {
                for (int j = 0; j < chunkCW; j++) {
                    for (int k = 0; k < kend - kstart; k++) {
                        C[i * chunkCW + j] += A[i * chunkAW + offsetAk + k] * B[j * chunkBH + offsetBk + k];
                    }
                }
            }
            kstart = kend;
            if (kstart >= Aend)
                iA++;
            if (kstart >= Bend)
                iB++;
        }

        const double end = MPI_Wtime();
        total_time += end - start;
    }
    if (rank == 0)
        C = malloc(M * N * sizeof *C);
    MPI_Gather(myC, chunkCA, MPI_FLOAT, C, chunkCA, MPI_FLOAT, 0, comm_cart);
    if (rank == 0)
        fprintf(stderr, "Average processing time: %.6lfs\n", total_time / loops);

    if (rank == 0) {
        const char *filename;
        if (argc > 6) {
            filename = argv[6];
        } else {
            filename = "dataOut.txt";
        }
        FILE *fp = fopen(filename, "w");
        if (fp) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    int cx = j / chunkCW,
                        cy = i / chunkCH,                           // coordinates of the chunk
                        chunkX = j % chunkCW, chunkY = i % chunkCH, // coordinates of item in the chunk
                        ci = cx + cy * col_size, chunkI = chunkX + chunkY * chunkCW;
                    fprintf(fp, "%f", C[ci * chunkCA + chunkI]);
                    if (j < N - 1)
                        fputc('\t', fp);
                }
                fputc('\n', fp);
            }
            fclose(fp);
        } else {
            fprintf(stderr, "Cannot open file \"%s\"\n", filename);
        }
        free(C);
    }
    MPI_Comm_free(&comm_cart);
    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_col);
    free(localA);
    free(localB);
    free(myA);
    free(myB);
    free(myC);
    MPI_Finalize();
    return 0;
}
