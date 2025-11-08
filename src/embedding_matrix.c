#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// generate one standard normal sample using Box–Muller
double randn_one() {
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// generate an array of size m with N(0, 1) samples
void randn(double *out, int m) {
    for (int i = 0; i < m; i++) {
        out[i] = randn_one();
    }
}

// build C embedding matrix
double* embedding_matrix(int V, int m) {
    double *C = malloc(V * m * sizeof(double));

    randn(C, V * m);

    return C;
}