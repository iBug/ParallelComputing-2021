#include <mpi.h>

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
    float *A;
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
            }
            A = new float[N * N];
            for (int i = 0; i < N * N; ++i)
                fin >> A[i];
            fin.close();
        }
    }
    COMM_WORLD.Bcast(&N, 1, MPI::INT, 0);
    if (N == 0)
        return 1;

    float *a = new float[N];
    float *f = new float[N];

    return 0;
}
