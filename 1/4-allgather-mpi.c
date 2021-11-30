#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int My_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size, sendtypesize, recvtypesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &sendtypesize);
    MPI_Type_size(recvtype, &recvtypesize);

    int sendchunk = sendcount * sendtypesize,
        recvchunk = recvcount * recvtypesize;
    memcpy(recvbuf + rank * recvchunk, sendbuf, sendchunk); // send to self
    int i;
    for (i = 1; (i << 1) <= size; i <<= 1) {
        int idx_start = rank & ~(i - 1);
        MPI_Sendrecv(recvbuf + idx_start * sendchunk, sendcount * i, sendtype, rank ^ i, 0, recvbuf + (idx_start ^ i) * recvchunk, recvcount * i, recvtype, rank ^ i, 0, comm, MPI_STATUS_IGNORE);
    }
    if (i != size) {
        // Special handling for last round
    }
    return 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    int iters = 50, count = 4096;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters < 1) {
            iters = 50;
        }
    }
    if (argc > 2) {
        count = atoi(argv[2]);
        if (count < 1) {
            count = 4096;
        }
    }

    // Prepare data
    int *sendbuf = malloc(count * sizeof(*sendbuf)),
        *recvbuf = malloc(count * size * sizeof(*recvbuf));
    for (int i = 0; i < count; i++)
        sendbuf[i] = rank;

    // Correctness check
    My_Allgather(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        int idx = i * count + rand() % count;
        if (recvbuf[idx] != i) {
            fprintf(stderr, "%d: Got %d, expected %d\n", rank, recvbuf[idx], i);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        fprintf(stderr, "Data validated\n");

    // Performance check
    double start, end;

#define TO_STRING(x) #x
#define TO_STRING_V(x) TO_STRING(x)
#define RUN_TEST(func)                                                                                                                \
    {                                                                                                                                 \
        start = MPI_Wtime();                                                                                                          \
        for (int i = 0; i < iters; i++)                                                                                               \
            func(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD);                                                   \
        end = MPI_Wtime();                                                                                                            \
        if (rank == 0)                                                                                                                \
            fprintf(stderr, TO_STRING(func) ": %d iterations in %lf seconds, %lf op/s\n", iters, end - start, iters / (end - start)); \
    }

    RUN_TEST(MPI_Allgather);
    RUN_TEST(My_Allgather);

    MPI_Finalize();
    return 0;
}
