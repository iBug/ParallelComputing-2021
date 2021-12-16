#!/usr/bin/python3

import sys
import random


INPUT = "dataIn.txt"
OUTPUT = "dataOut.txt"


def gen_input(size, scale):
    with open(INPUT, "w") as f:
        f.write(f"{size} {size}\n")
        for i in range(size):
            f.write("{}\n".format("\t".join([str(2 * scale * random.random() - scale) for _ in range(size)])))


def check_output(rtol, atol):
    import numpy as np
    with open(INPUT, "r") as f:
        size = int(f.readline().split()[0])
        original = np.matrix([[float(x) for x in f.readline().split()] for _ in range(size)])
    with open(OUTPUT, "r") as f:
        r_matrix = np.matrix([[float(x) for x in f.readline().split()] for _ in range(size)])
        f.readline()
        q_matrix = np.matrix([[float(x) for x in f.readline().split()] for _ in range(size)])
    result = np.allclose(q_matrix.T * q_matrix, np.identity(size), rtol=rtol, atol=atol)
    if not result:
        print((q_matrix.T * q_matrix - np.identity(size)).max())
        print("Bad Q matrix")
        return 1
    l_result = q_matrix * r_matrix
    result = np.allclose(l_result, original, rtol=rtol, atol=atol)
    if not result:
        print("FAIL")
        l_diff = l_result - original
        #print(l_diff)
        print(l_diff.max(), l_diff.min())
        return 1
    print("OK")


def main(argv):
    if len(argv) < 2:
        return gen_input(16, 10.0)
    if argv[1].startswith("g"):
        size = 16
        if len(argv) >= 3:
            size = int(argv[2])
        scale = float(size)
        if len(argv) >= 4:
            scale = float(argv[3])
        return gen_input(size, scale)
    elif argv[1].startswith("c"):
        rtol = 1e-3
        if len(argv) >= 3:
            rtol = float(argv[2])
        atol = 2e-5
        if len(argv) >= 4:
            atol = float(argv[3])
        return check_output(rtol, atol)
    else:
        raise ValueError(f"Unknown command {argv[1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
