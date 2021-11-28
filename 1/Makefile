CC := gcc
CXX := g+=
MPICC := mpicc
MPICXX := mpic++
CFLAGS := -O2 -Wall -Wno-unused-result
CXXFLAGS := $(CFLAGS)

OMP_SRCS := $(wildcard *-omp.c)
MPI_SRCS := $(wildcard *-mpi.c)
SRCS := $(OMP_SRCS) $(MPI_SRCS)
OMP_BINS := $(patsubst %.c, %, $(OMP_SRCS))
MPI_BINS := $(patsubst %.c, %, $(MPI_SRCS))
BINS := $(OMP_BINS) $(MPI_BINS)

# Extra .gitignore entries
IGNORES := *.o input.txt output.txt


.PHONY: all clean .gitignore

all: .gitignore $(BINS)

$(OMP_BINS): %: %.c
	$(CC) $(CFLAGS) -o $@ $^ -fopenmp

$(MPI_BINS): %: %.c
	$(MPICC) $(CFLAGS) -o $@ $^

.gitignore: $(SRCS)
	@echo "# Generated from Makefile" > $@
	@for i in $(IGNORES) $(BINS); do echo "$$i" >> $@; done

clean:
	rm -f $(BINS)
