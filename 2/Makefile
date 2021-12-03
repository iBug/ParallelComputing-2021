CC := gcc
CXX := g+=
MPICC := mpicc
MPICXX := mpic++
CFLAGS := -O2 -Wall -Wno-unused-result
CXXFLAGS := $(CFLAGS)

CXXEXT := cpp
CSRCS := $(sort $(wildcard *.c))
CXXSRCS := $(sort $(wildcard *.$(CXXEXT)))
CBINS := $(patsubst %.c, %, $(CSRCS))
CXXBINS := $(patsubst %.$(CXXEXT), %, $(CXXSRCS))
SRCS := $(CSRCS) $(CXXSRCS)
BINS := $(CBINS) $(CXXBINS)

# Extra .gitignore entries
IGNORES := *.o *.txt


.PHONY: all clean .gitignore

all: .gitignore $(BINS)

$(CBINS): %: %.c
	$(MPICC) $(CFLAGS) -o $@ $^

$(CXXBINS): %: %.$(CXXEXT)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

.gitignore: $(SRCS)
	@echo "# Generated from Makefile" > $@
	@for i in $(IGNORES) $(BINS); do echo "$$i" >> $@; done

clean:
	rm -f $(BINS)
