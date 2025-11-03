// built-in files
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// external files
#include "embedding_matrix.h"
#include "get_data.h"

int main() {
    MPI_Init(NULL, NULL);

    // seed
    srand(time(NULL));

    // hyperparameters
    int epochs = 50;
    int V = 512;
    int m = 64;
    int h = 32;
    int n = 2;

    // parameters
    double* C = embedding_matrix(V, m);
    double* H = embedding_matrix(h, n*m);
    double* d = embedding_matrix(h, 1);
    double* U = embedding_matrix(V, h);
    double* b = embedding_matrix(V, 1);

    // vocabulary
    int* vocab = malloc(V * sizeof(int));

    for (int epoch = 0; epoch < epochs; epoch++) {
        // FORWARD PHASE
        // perform forward computation for the word features layer
        int* ids_full = get_data();
        int ids[2] = {ids_full[0], ids_full[1]};
        int next_id = ids_full[2];
        double* x_flat = malloc(n * m * sizeof(double)); //input vector neural network that has been flattened 

        for (int i = 0; i < n; i++) {
            int id = ids[i];
            
            for (int j = 0; j < m; j++) {
                x_flat[i * m + j] = C[id * m + j];  
            }
        }

        // perform forward computation for the hidden layer
        double* o = malloc(h * sizeof(double)); // output vector first layer

        cblas_dgemv( // BLAS faster matrix mul
            CblasRowMajor,    
            CblasNoTrans,    
            h, n*m,
            1.0,
            H, n*m,           
            x_flat, 1,         
            0.0,
            o, 1               
        );

        for (int i = 0; i < h; i++) {
            o[i] = tanh(o[i] + d[i]);
        }

        // perform forward computation for output units in the i-th block
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

        for (int i = 0; i < V / comm_sz; i++) {
            for (int j = 0; j < h; j++) {
                local_U[i * h + j] = U[(rank * (V / comm_sz) + i) * h + j];
            }

            local_b[i] = b[rank * (V / comm_sz) + i];
        }

        for (int r = 0; r < V/comm_sz; r++) {
            int global_row = (rank * (V/comm_sz)) + r;

            for (int c = 0; c < h; c++) {
                local_U[r * h + c] = U[global_row * h + c];
            }
        }

        for (int r = 0; r < V/comm_sz; r++) {
            int global_idx = rank * (V/comm_sz) + r;

            local_b[r] = b[global_idx];
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
                double li = log(p[vocab[i]]); // loss of wi
                
                L += li;
            }

            L = L / V;

            printf("Epoch: %d | Loss: %lf \n", epoch, L);
        }
        
        // BACKWARD PHASE 
        double* local_gradient_Ly = malloc((V/comm_sz) * sizeof(double));
        double* gradient_La = malloc(h * sizeof(double));
        double* gradient_Lx = malloc(n * m * sizeof(double));
        double* local_gradient_La = malloc(h * sizeof(double));
        double* local_gradient_Lo = malloc(h * sizeof(double));
        double lr = 0.01;

        // perform backward gradient for output units in i-th block
        for (int i = 0; i < h; i++) {
            local_gradient_La[i] = 0;
        }

        for (int i = 0; i < n*m; i++) {
            gradient_Lx[i] = 0;
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
                H[k * m*n + i] += lr * local_gradient_Lo[k] * x_flat[i];
            }
        }
        
        // update word feature vectors for the input words: loop over k between 1 and n - 1
        for (int k = 0; k < n - 1; k++) {
            int word_id = ids[k]; 

            for (int j = 0; j < m; j++) {
                C[word_id * m + j] += lr * gradient_Lx[k * m + j];
            }
        }
        
        free(x_flat);
        free(o);
        free(local_U);
        free(local_b);
        free(local_y);
        free(local_p);
        free(p);
        free(local_gradient_Ly);
        free(gradient_La);
        free(gradient_Lx);
        free(local_gradient_La);
        free(local_gradient_Lo);
    }
    
    MPI_Finalize();

    return 0;
}