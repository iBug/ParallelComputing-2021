#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int My_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size, sendtypesize, recvtypesize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &sendtypesize);
    MPI_Type_size(recvtype, &recvtypesize);

    int sendchunk = sendcount * sendtypesize,
        recvchunk = recvcount * recvtypesize;
    memcpy(recvbuf + rank * recvchunk, sendbuf + rank * sendchunk, sendchunk); // send to self
    for (int i = 1; i < size; i++) {
        int sendT = (rank + i) % size,
            recvT = (rank + size - i) % size;
        MPI_Sendrecv(sendbuf + sendT * sendchunk, sendcount, sendtype, sendT, 0, recvbuf + recvT * recvchunk, recvcount, recvtype, recvT, 0, comm, MPI_STATUS_IGNORE);
    }
    return 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    int iters = 100, count = 4096;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters < 1) {
            iters = 100;
        }
    }
    if (argc > 2) {
        count = atoi(argv[2]);
        if (count < 1) {
            count = 4096;
        }
    }

    // Prepare data
    int *sendbuf = malloc(count * size * sizeof(*sendbuf)),
        *recvbuf = malloc(count * size * sizeof(*recvbuf));
    for (int i = 0; i < size; i++)
        sendbuf[i * count] = rank * size + i;

    // Correctness check
    My_Alltoall(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        if (recvbuf[i * count] != i * size + rank) {
            fprintf(stderr, "%d: Got %d, expected %d\n", rank, recvbuf[i], i * size + rank);
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
#define RUN_TEST(func) \
{ \
    start = MPI_Wtime(); \
    for (int i = 0; i < iters; i++) \
        func(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD); \
    end = MPI_Wtime(); \
    if (rank == 0) \
        fprintf(stderr, TO_STRING(func) ": %d iterations in %lf seconds, %lf op/s\n", iters, end - start, iters / (end - start)); \
}

    RUN_TEST(MPI_Alltoall);
    RUN_TEST(My_Alltoall);

    MPI_Finalize();
    return 0;
}
