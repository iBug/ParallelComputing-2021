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
    int splits = 2, root = rand() % size, value = rand(), count = 1;
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
    if (argc > 4) {
        count = atoi(argv[4]);
        if (count < 1)
            count = 1;
    }
    if (argc > 5 && rank == 0) {
        fprintf(stderr, "Warning: Up to 4 arguments are understood\n");
    }

    // Deploy input value
    int *data = malloc(count * sizeof(*data));
    if (rank == root) {
        for (int i = 0; i < count; i++) {
            data[i] = value;
        }
    }

    // Reference implementation
    const double ref_start_time = MPI_Wtime();
    MPI_Bcast(data, count, MPI_INT, root, MPI_COMM_WORLD);
    const double ref_end_time = MPI_Wtime(),
          ref_run_time = ref_end_time - ref_start_time;

    // Broadcast data
    const double my_start_time = MPI_Wtime();
    MPI_MyBcast(data, count, MPI_INT, root, MPI_COMM_WORLD, splits);
    const double my_end_time = MPI_Wtime(),
          my_run_time = my_end_time - my_start_time;

    // Print result
    int *buf = NULL;
    if (rank == root)
        buf = malloc(size * count * sizeof(*buf));
    MPI_Gather(data, count, MPI_INT, buf, count, MPI_INT, root, MPI_COMM_WORLD);
    double ref_time, my_time;
    MPI_Reduce(&ref_run_time, &ref_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_run_time, &my_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == root) {
        printf("MPI time: %.6lf\n", ref_time);
        printf("My time:  %.6lf\n", my_time);
        for (int i = 0; i < size; i++)
            printf("data[%d] = %d\n", i, buf[i * count]);
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
