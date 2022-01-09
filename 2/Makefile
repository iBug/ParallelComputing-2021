CC := gcc
MPICC := mpicc
CFLAGS := -O3 -Wall -Wno-unused-result -march=native -fopenmp
LDFLAGS := -lm

SRCS := $(sort $(wildcard *.c))
BINS := $(patsubst %.c, %, $(SRCS))

# Extra .gitignore entries
IGNORES := "*.o" "*.txt"


.PHONY: all clean .gitignore

all: .gitignore $(BINS)

ludep summa psrs: %: %.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

qr qsort: %: %.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.gitignore: $(SRCS)
	@echo "# Generated from Makefile" > $@
	@for i in $(IGNORES) $(BINS); do echo "$$i" >> $@; done

clean:
	rm -f $(BINS)
