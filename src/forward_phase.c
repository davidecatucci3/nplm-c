// built-in files
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external files
#include "embedding_matrix.h"

int forward_phase(int rank, int block_size, int n, int m, int h, int ids, double* C, double* H, double* x, double *o, double *local_y, double* U, double* b) {
    // FORWARD PHASE
    // perform forward computation for the word features layer  
    for (int i = 0; i < n; i++) {
        int id = ids[i];
                
        for (int j = 0; j < m; j++) {
            x[i * m + j] = C[id * m + j];  
        }
    }
            
    // perform forward computation for the hidden layer
    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,    
        CblasNoTrans,    
        h, n*m,
        1.0,
        H, n*m,                    
        x, 1,         
        0.0,
        o, 1               
    );

    for (int i = 0; i < h; i++) {
        o[i] = tanh(o[i] + d[i]);
    }

    // perform forward computation for output units in the i-th block
    double S = 0.0;                                  // total sum of exponential for softmax
    double local_s = 0.0;                            // local exponential for softmax
    double* local_U = U + rank * block_size * h;     // take a block of U for parallelize it over all ranks
    double* local_b = b + rank * block_size;         // take a block of b for parallelize it over all ranks

    cblas_dgemv( // BLAS faster matrix mul
        CblasRowMajor,     
        CblasNoTrans,      
        block_size, h,
        1.0,
        local_U, h,           
        o, 1,         
        0.0,
        local_y, 1               
    );

    for (int i = 0; i < block_size; i++) {
        local_y[i] += local_b[i];
    }

    // softmax stability 
    double local_max = -INFINITY;
    double global_max = 0.0;

    for (int i = 0; i < block_size; ++i) {
        if (local_y[i] > local_max) {
            local_max = local_y[i];
        }
    }

    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (int i = 0; i < block_size; ++i) {
        local_p[i] = exp(local_y[i] - global_max);

        local_s += local_p[i];
    }

    // compute and share S among the processors
    MPI_Allreduce(&local_s, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
    // normalize the probabilities
    for (int i = 0; i < block_size; i++) {
        local_p[i] /= S;
    }
}