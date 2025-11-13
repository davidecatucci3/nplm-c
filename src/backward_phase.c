// built-in files
#include <cblas.h>
#include <stdio.h>
#include <mpi.h>

void backward_phase(int rank, int* ids, int next_id, int block_size, int m, int n, int h, int V, double lr, double wd, double* local_gradient_La, double* local_gradient_Lo, double* C, double* x, double* H, double* o, double* d, double* local_U, double* local_gradient_Ly, double* gradient_Lx, double* gradient_La, double* local_b, double* local_p) {
    // perform backward gradient for output units in i-th block
            for (int i = 0; i < h; i++) {
                local_gradient_La[i] = 0.0;
            }

            for (int i = 0; i < n*m; ++i) {
                gradient_Lx[i] = 0.0;
            }
            
            for (int i = 0; i < block_size; i++) {
                if (i + rank*(block_size) == next_id) {
                    local_gradient_Ly[i] = 1 - local_p[i];
                } else {
                    local_gradient_Ly[i] = -local_p[i];
                }

                local_b[i] += lr*local_gradient_Ly[i];

                for (int j = 0; j < h; j++) {            
                    local_gradient_La[j] += local_gradient_Ly[i] * local_U[i*h + j];
            
                    local_U[i * h + j] += lr * local_gradient_Ly[i] * o[j];
                }
            }
            
            // share dL/da among all processors
            MPI_Allreduce(local_gradient_La, gradient_La, h, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // backpropagate through and update hidden layer weights
            for (int k = 0; k < h; k++) {
                local_gradient_Lo[k] = (1.0 - o[k] * o[k]) * gradient_La[k];
            }

            for (int i = 0; i < m*n; i++) {
                for (int k = 0; k < h; k++) {
                    gradient_Lx[i] += H[k * m*n + i] * local_gradient_Lo[k];
                }
            }

            for (int k = 0; k < h; k++) {
                d[k] += lr * local_gradient_Lo[k];

                for (int i = 0; i < m*n; i++) {
                    H[k * m*n + i] += lr * local_gradient_Lo[k] * x[i];
                }
            }
            
            // update word feature vectors for the input words: loop over k between 1 and n
            for (int k = 0; k < n; k++) {
                int word_id = ids[k]; 

                for (int j = 0; j < m; j++) {
                    C[word_id * m + j] += lr * gradient_Lx[k * m + j];
                }
            }

            // weight decay regularization
            for (int i = 0; i < V*m; ++i) C[i] *= (1.0 - lr * wd);
            for (int i = 0; i < h*n*m; ++i) H[i] *= (1.0 - lr * wd);
            for (int i = 0; i < block_size*h; ++i) local_U[i] *= (1.0 - lr * wd);
}