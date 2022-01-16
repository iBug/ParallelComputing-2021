1. compile:
mpicc qr.c -o qr -lm

2. run:
mpirun -np 4 qr

3. result:
Whole running time    = 0.002906 seconds
Distribute data time  = 0.001895 seconds
Parallel compute time = 0.001011 seconds
Input of file "dataIn.txt"
3        3
0.000000        3.000000        4.000000
3.000000        1.000000        2.000000
4.000000        2.000000        1.000000

Output of QR operation
Matrix R:
0.547723        -0.365148       2.745626
-0.000000       2.731582        3.097670
-0.000000       0.000000        1.966209
Matrix Q:
0.196116        0.901140        0.340810
0.588348        -0.225285       0.550539
0.784465        -0.056321       -0.498106
