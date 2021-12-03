CC := gcc
MPICC := mpicc
CFLAGS := -O3 -Wall -Wno-unused-result -march=native

SRCS := $(sort $(wildcard *.c))
BINS := $(patsubst %.c, %, $(SRCS))

# Extra .gitignore entries
IGNORES := "*.o" "*.txt"


.PHONY: all clean .gitignore

all: .gitignore $(BINS)

$(BINS): %: %.c
	$(MPICC) $(CFLAGS) -o $@ $^

.gitignore: $(SRCS)
	@echo "# Generated from Makefile" > $@
	@for i in $(IGNORES) $(BINS); do echo "$$i" >> $@; done

clean:
	rm -f $(BINS)
