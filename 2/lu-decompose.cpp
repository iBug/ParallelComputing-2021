#include <mpi.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using MPI::COMM_WORLD;

class MPIWrapper {
   public:
    MPIWrapper(int &argc, char **&argv) {
        MPI::Init(argc, argv);
    }
    ~MPIWrapper() {
        MPI::Finalize();
    }
};

int main(int argc, char **argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const std::vector<std::string> args(argv, argv + argc);
    MPIWrapper wrapper(argc, argv);
    const auto rank = COMM_WORLD.Get_rank();
    const auto size = COMM_WORLD.Get_size();

    int N; // Matrix size
    float *A = nullptr;
    if (rank == 0) {
        std::string filename;
        if (args.size() >= 2) {
            filename = args[1];
        } else {
            filename = "dataIn.txt";
        }
        std::ifstream fin(filename);
        if (!fin) {
            std::cerr << "Cannot open file \"" << filename << "\"" << std::endl;
            N = 0;
        } else {
            int N2;
            fin >> N >> N2;
            if (N != N2 || N <= 1) {
                std::cerr << "Invalid matrix size" << std::endl;
                N = 0;
            } else if (N < size) {
                std::cerr << "Matrix size is too small for " << size << "processes" << std::endl;
                N = 0;
            }
            A = new float[N * N];
            for (int i = 0; i < N * N; i++)
                fin >> A[i];
            fin.close();
        }
    }
    COMM_WORLD.Bcast(&N, 1, MPI::INT, 0);
    if (N == 0)
        return 1;

    const auto m = N / size + (N % size > 0);
    float *a = new float[N * m];
    float *buf = new float[N];

    const auto scatter_unit = N * size;
    for (int i = 0; i < N / size; i++) {
        COMM_WORLD.Scatter(A + i * scatter_unit, N, MPI::FLOAT, a + i * N, N, MPI::FLOAT, 0);
    }
    if (N % size && rank == 0)
        std::memcpy(a + (N / size) * N, A + (N / size) * scatter_unit, N * sizeof(*a));
    if (N % size > 1) {
        if (rank == 0) {
            for (int i = 1; i < N % size; i++) {
                COMM_WORLD.Send(A + (N / size) * scatter_unit + i * N, N, MPI::FLOAT, i, 0);
            }
        } else if (rank < (N / size)) {
            COMM_WORLD.Recv(a + (N / size) * N, N, MPI::FLOAT, 0, 0);
        }
    }

    for (int i = 0; i < N; i++) {
        const auto round = i / size; // original "i"
        const auto mr = i % size;    // main row
        float *f;
        if (rank == mr) {
            f = a + round * N;
        } else {
            f = buf;
        }
        COMM_WORLD.Bcast(f, N, MPI::FLOAT, mr);

        if (rank <= mr) {
            for (int k = round + 1; k < m; k++) {
                a[k * N + i] /= f[i];
                for (int w = i + 1; w < N; w++)
                    a[k * N + w] -= f[w] * a[k * N + i];
            }
        } else {
            for (int k = round; k < m; k++) {
                a[k * N + i] /= f[i];
                for (int w = i + 1; w < N; w++)
                    a[k * N + w] -= f[w] * a[k * N + i];
            }
        }
    }

    for (int i = 0; i < N / size; i++) {
        COMM_WORLD.Gather(a + i * N, N, MPI::FLOAT, A + i * scatter_unit, N, MPI::FLOAT, 0);
    }
    if (N % size && rank == 0)
        std::memcpy(A + (N / size) * scatter_unit, a + (N / size) * N, N * sizeof(*a));
    if (N % size > 1) {
        if (rank == 0) {
            for (int i = 1; i < N % size; i++) {
                COMM_WORLD.Recv(A + (N / size) * scatter_unit + i * N, N, MPI::FLOAT, i, 0);
            }
        } else {
            COMM_WORLD.Send(a + (N / size) * N, N, MPI::FLOAT, 0, 0);
        }
    }

    if (rank == 0) {
        std::string filename;
        if (args.size() >= 3) {
            filename = args[2];
        } else {
            filename = "dataOut.txt";
        }
        std::ofstream fout(filename);

        // Output L
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i > j)
                    fout << A[i * N + j];
                else if (i == j)
                    fout << 1.F;
                else
                    fout << 0.F;
                if (j < N - 1)
                    fout << "\t";
            }
            fout << std::endl;
        }
        fout << std::endl;

        // Output U
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i > j)
                    fout << 0.F;
                else
                    fout << A[i * N + j];
                if (j < N - 1)
                    fout << "\t";
            }
            fout << std::endl;
        }

        delete[] A;
    }
    delete[] a;
    delete[] buf;

    return 0;
}
