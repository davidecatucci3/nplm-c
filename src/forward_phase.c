// built-in files
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external files
#include "model.h"

void forward(Model *mo, int ids[]) {
    // seed
    srand(time(NULL));

    // hyperparameters
    int V = 512;
    int m = 32;
    int h = 16;
    int n = 2;

    // perform forward computation for the word features layer
    double* x_flat = malloc(n * m * sizeof(double)); //input vector neural network that has been flattened 

    for (int i = 0; i < n; i++) {
        int id = ids[i];
        
        for (int j = 0; j < m; j++) {
            x_flat[i * m + j] = mo->C[id * m + j];  
        }
    }

    // perform forward computation for the hidden layer
    double* o = malloc(h * sizeof(double)); // output vector first layer

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,    
        CblasNoTrans,    
        h, n*m,
        1.0,
        mo->H, n*m,           
        x_flat, 1,         
        0.0,
        o, 1               
    );

    for (int i = 0; i < h; i++) {
        o[i] = tanh(o[i] + mo->d[i]);
    }

    // perform forward computation for output units in the i-th block
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double S = 0;
    double local_s = 0;
    double* local_U = malloc((V/comm_sz) * h * sizeof(double)); // weights second layer
    double* local_b = malloc((V/comm_sz) * sizeof(double)); // bias second layer
    double* local_y = malloc((V/comm_sz) * sizeof(double)); // output second layer (logits)
    double* local_p = malloc((V/comm_sz) * sizeof(double)); // output second layer (probs)
    double* p = malloc(V * sizeof(double));

    for (int r = 0; r < V/comm_sz; r++) {
        int global_row = (rank * (V/comm_sz)) + r;

        for (int c = 0; c < h; c++) {
            local_U[r * h + c] = mo->U[global_row * h + c];
        }
    }

    for (int r = 0; r < V/comm_sz; r++) {
        int global_idx = (V/comm_sz) + r;

        local_b[r] = mo->b[global_idx];
    }

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,     
        CblasNoTrans,      
        V/comm_sz, h,
        1.0,
        local_U, h,           
        o, 1,         
        0.0,
        local_y, 1               
    );

    for (int i = 0; i < V/comm_sz; i++) {
        local_y[i] = local_y[i] + local_b[i];
        local_p[i] = exp(local_y[i]);
            
        local_s += local_p[i];
    } 

    // compute and share S among the processors
    MPI_Allreduce(&local_s, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // normalize the probabilities
    for (int i = 0; i < V/comm_sz; i++) {
        local_p[i] /= S;
    }

    MPI_Allgather(local_p, V/comm_sz, MPI_DOUBLE, p, V/comm_sz, MPI_DOUBLE, MPI_COMM_WORLD);

    // compute loss
    double L = 0; // total loss

    if (rank == 0) {
        for (int i = 0; i < V; i++) {
            double li = log(p[mo->vocab[i]]); // loss of wi
             
            L += li;
        }

        L = L / V;
    }

    MPI_Finalize();
}

