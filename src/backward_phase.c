// built-in files
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external files
# include "model.h"

void backward(Model *mo, int ids[], int next_id) {
    // seed
    srand(time(NULL));

    // hyperparameters
    int V = 512;
    int m = 32;
    int h = 16;
    int n = 2;
    double lr = 0.01;

    // main variables
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double* local_p = malloc((V/comm_sz) * sizeof(double));
    double* local_gradient_Ly = malloc((V/comm_sz) * sizeof(double));
    double* gradient_La = malloc(h * sizeof(double));
    double* gradient_Lx = malloc(h * sizeof(double));
    double* local_gradient_La = malloc(h * sizeof(double));
    double* local_gradient_Lo = malloc(h * sizeof(double));
    double* local_U = embedding_matrix(V / comm_sz, h);
    double* local_b = malloc((V/comm_sz) * sizeof(double));
    double* x_flat = malloc(n * m * sizeof(double));
    double* a = malloc(h * sizeof(double));

    // perform backward gradient for output units in i-th block
    for (int i = 0; i < h; i++) {
        local_gradient_La[i] = 0;
    }

    for (int i = 0; i < V / comm_sz; i++) {
        if (i + rank*(V/comm_sz) == next_id) {
            local_gradient_Ly[i] = 1 - local_p[i];
        } else {
            local_gradient_Ly[i] = -local_p[i];
        }

        local_b[i] += lr*local_gradient_Ly[i];

        for (int j = 0; j < h; j++) {            
            local_gradient_La[j] += lr * local_gradient_Ly[i * h + j];
        }

        for (int j = 0; j < h; j++) {            
            local_U[i * h + j] += lr * local_gradient_Ly[i * h + j];
        }
    }

    // share dL/da among all processors
    MPI_Allreduce(&local_gradient_La, &gradient_La, h, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // backpropagate through and update hidden layer weights
    for (int k = 0; k < h; k++) {
        local_gradient_Lo[k] = (1.0 - a[k] * a[k]) * gradient_La[k];
    }

    for (int i = 0; i < m*n; i++) {
        for (int k = 0; k < h; k++) {
            gradient_Lx[i] += H[k * m*n + i] * local_gradient_Lo[k];
        }
    }

    for (int k = 0; k < h; k++) {
        mo->d[k] += lr * local_gradient_Lo[k];

        for (int i = 0; i < m*n; i++) {
            mo->H[k * m*n + i] += lr * local_gradient_Lo[k] * x_flat[i];
        }
    }

    // update word feature vectors for the input words: loop over k between 1 and n - 1
    for (int k = 0; k < n - 1; k++) {
        int word_id = ids[k]; 

        for (int j = 0; j < m; j++) {
            mo->C[word_id * m + j] += lr * gradient_Lx[k * m + j];
        }
    }

    MPI_Finalize();

    return 0;
}


