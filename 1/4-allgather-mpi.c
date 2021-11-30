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
    for (int i = 1; i < size; i <<= 1) {
        int peer = rank ^ i,
            start = rank & ~(i - 1),
            rstart = start ^ i,
            blocks = (start + i > size) ? (size - start) : i,
            rblocks = (rstart + i > size) ? (size - rstart) : i;
        if (rstart >= size)
            continue;

        if (1) {
            fprintf(stderr, "%d: <%d>, S[%d:%d], R[%d:%d]\n", rank, peer, start, start + blocks - 1, rstart, rstart + rblocks - 1);
        }
        int j = 1;
        if (peer < size) {
            MPI_Sendrecv(recvbuf + start * sendchunk, sendcount * blocks, sendtype, peer, 0, recvbuf + rstart * recvchunk, recvcount * rblocks, recvtype, peer, 0, comm, MPI_STATUS_IGNORE);
        } else {
            for (; j < i; j <<= 1) {
                int sibling = rank ^ j,
                    rsibling = peer ^ j,
                    s_partialrank = sibling & (i - 1);
                if (rsibling < size || s_partialrank < j) {
                    fprintf(stderr, "%d: no peer %d, <%d> R[%d:%d]\n", rank, peer, sibling, rstart, rstart + rblocks - 1);
                    MPI_Recv(recvbuf + rstart * recvchunk, recvcount * rblocks, recvtype, sibling, 0, comm, MPI_STATUS_IGNORE);
                    j <<= 1;
                    break;
                }
            }
        }
        for (; j < i; j <<= 1) {
            int sibling = rank ^ j,
                rsibling = peer ^ j,
                s_partialrank = size & (j - 1);
            if (rank == 1) {
                int A = rsibling >= size, B = s_partialrank == 0;
                fprintf(stderr, "%d: i=%d, j=%d, test peer %d of %d (A:%d, B:%d)\n", rank, i, j, rsibling, sibling, A, B);
            }
            if (rsibling >= size && s_partialrank == 0) {
                fprintf(stderr, "%d: no peer %d, <%d> S[%d:%d]\n", rank, rsibling, sibling, rstart, rstart + rblocks - 1);
                MPI_Send(recvbuf + rstart * recvchunk, recvcount * rblocks, recvtype, sibling, 0, comm);
            }
        }
    }
    fprintf(stderr, "%d: done\n", rank);
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
        if (iters < 0) {
            iters = 0;
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

    if (iters > 0) {
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
    }

    MPI_Finalize();
    return 0;
}
