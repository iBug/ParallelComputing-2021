#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int MPI_MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, int splits) {
    // Split comm_N
    int rank;
    MPI_Comm_rank(comm, &rank);
    const int this_comm = rank % splits;
    const int root_comm = root % splits,
              root_rank_N = root / splits;
    MPI_Comm comm_N;
    MPI_Comm_split(comm, this_comm, rank, &comm_N);

    // Group comm_H
    int rank_N;
    MPI_Comm_rank(comm_N, &rank_N);
    MPI_Comm comm_H;
    MPI_Comm_split(comm, rank_N == 0 ? 0 : MPI_UNDEFINED, rank, &comm_H);

    // Send data to rank_N 0
    if (root_rank_N != 0) {
        if (this_comm == root_comm) {
            if (rank_N == 0) {
                MPI_Recv(buffer, count, datatype, root_rank_N, 0, comm_N, MPI_STATUS_IGNORE);
            } else if (rank_N == root_rank_N) {
                MPI_Send(buffer, count, datatype, 0, 0, comm_N);
            }
        }
    }

    // Broadcast data in comm_H
    if (comm_H != MPI_COMM_NULL)
        MPI_Bcast(buffer, count, datatype, root_comm, comm_H);

    // Broadcast data in comm_N
    MPI_Bcast(buffer, count, datatype, 0, comm_N);

    // No error handling
    return 0;
}

/* Usage: $0 [splits [root [value]]]
 *   splits: number of splits (default: 2)
 *   root: root rank (default: random)
 *   value: value to broadcast (default: random)
 */
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 4) {
        MPI_Finalize();
        if (rank == 0)
            fprintf(stderr, "At least 4 processes required\n");
        return 1;
    }

    // Generate parameters and process CLI arguments
    srand(time(NULL));
    int splits = 2, root = rand() % size, value = rand();
    if (argc > 1) {
        splits = atoi(argv[1]);
        if (splits < 2) {
            MPI_Finalize();
            if (rank == 0)
                fprintf(stderr, "Invalid split count %d\n", splits);
            return 1;
        } else if (splits > size / 2) {
            MPI_Finalize();
            if (rank == 0)
                fprintf(stderr, "Split count %d is too large\n", splits);
            return 1;
        }
    }
    if (argc > 2) {
        root = atoi(argv[2]);
        if (root < 0 || root >= size) {
            MPI_Finalize();
            if (rank == 0)
                fprintf(stderr, "Invalid root rank %d\n", root);
            return 0;
        }
    }
    if (argc > 3) {
        value = atoi(argv[3]);
    }
    if (argc > 4 && rank == 0) {
        fprintf(stderr, "Warning: Up to 3 arguments are understood\n");
    }

    // Sanity guard: Separate MPI data from command line arguments
    int data = 0;
    if (rank == root)
        data = value;

    // Broadcast data
    MPI_MyBcast(&data, 1, MPI_INT, root, MPI_COMM_WORLD, splits);

    // Print result
    int *buf = NULL;
    if (rank == root)
        buf = malloc(size * sizeof(*buf));
    MPI_Gather(&data, 1, MPI_INT, buf, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank == root) {
        for (int i = 0; i < size; i++)
            printf("data[%d] = %d\n", i, buf[i]);
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
