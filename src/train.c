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
    // initialize MPI and set rank and comm_sz
    // initialize MPI and set rank and comm_sz
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // process id       // process id 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);   // total number of processes   // total number of processes

    // hyperparameters
    int epochs = 500;
    int V = vocab.size;
    int m = 64;              // embedding size
    int h = 32;              // hidde layer units
    int n = 2;               // input elements
    double lr = 1e-3;        // start learning rate
    double r = 1e-8;         // decreasor rate
    double wd = 1e-4;        // weight decay

    // parameters
    double* C = embedding_matrix(V, m);       // embedding weights
    double* H = embedding_matrix(h, n*m);     // weights first layer
    double* d = embedding_matrix(h, 1);       // bias first layer
    double* U = embedding_matrix(V, h);       // weights second layer
    double* b = embedding_matrix(V, 1);       // bias second layer

    // other valriables
    long long t = 0;

    // traininig loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // seed
        srand(time(NULL) + epoch);

        // FORWARD PHASE
        // perform forward computation for the word features layer
        int x1, x2, y;
        get_chunk(&x1, &x2, &y);        
        int ids[2] = {x1, x2}; 
        int next_id =y;
        double* x = malloc(n * m * sizeof(double)); //input vector neural network (flattened)

        for (int i = 0; i < n; i++) {
            int id = ids[i];
            
            for (int j = 0; j < m; j++) {
                x[i * m + j] = C[id * m + j];  
                x[i * m + j] = C[id * m + j];  
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
            x, 1,         
            x, 1,         
            0.0,
            o, 1               
        );

        for (int i = 0; i < h; i++) {
            o[i] = tanh(o[i] + d[i]);
        }

        // perform forward computation for output units in the i-th block
        double S = 0.0;
        double local_s = 0.0;
        double* local_U = malloc((V/comm_sz) * h * sizeof(double)); // weights second layer
        double* local_b = malloc((V/comm_sz) * sizeof(double)); // bias second layer
        double* local_y = malloc((V/comm_sz) * sizeof(double)); // output second layer (logits)
        double* local_p = malloc((V/comm_sz) * sizeof(double)); // output second layer (probs)
        double* p = malloc(V * sizeof(double));
        double local_max = local_y[0];
        double global_max = 0.0;

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

        for (int i = 1; i < V / comm_sz; ++i) {
            if (local_y[i] > local_max) local_max = local_y[i];
        }

        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < V/comm_sz; ++i) {
            double ex = exp(local_y[i] - global_max);

            local_p[i] = ex;

            local_s += ex;
        }

        // compute and share S among the processors
        MPI_Allreduce(&local_s, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // normalize the probabilities
        for (int i = 0; i < V/comm_sz; i++) {
            local_p[i] /= S;
        }

        MPI_Allgather(local_p, V/comm_sz, MPI_DOUBLE, p, V/comm_sz, MPI_DOUBLE, MPI_COMM_WORLD);


        // compute loss
        if (rank == 0) {
            double L = log(p[next_id] + 1e-12);

            printf("Epoch %d | Loss=%.2f | lr=%.6f \n", epoch, L, lr);
        }
        
        // BACKWARD PHASE 
        double* local_gradient_Ly = malloc((V/comm_sz) * sizeof(double));
        double* gradient_La = malloc(h * sizeof(double));
        double* gradient_Lx = malloc(n * m * sizeof(double));
        double* local_gradient_La = malloc(h * sizeof(double));
        double* local_gradient_Lo = malloc(h * sizeof(double));
        
        
        // perform backward gradient for output units in i-th block
        for (int i = 0; i < h; i++) {
            local_gradient_La[i] = 0.0;
        }

        for (int i = 0; i < V / comm_sz; i++) {
            if (i + rank*(V/comm_sz) == next_id) {
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

        for (int i = 0; i < n*m; ++i) {
            gradient_Lx[i] = 0.0;
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
                H[k * m*n + i] += lr * local_gradient_Lo[k] * x[i];
            }
        }
        
        // update word feature vectors for the input words: loop over k between 1 and n
        for (int k = 0; k < n; k++) {
            int word_id = ids[k]; 
        }

        for (int k = 0; k < n; k++) {
            int word_id = inpu_ids[k]; 

            for (int j = 0; j < m; j++) {
                C[word_id * m + j] += lr * gradient_Lx[k * m + j];
            }
        }

        // weight decay regularization
        for (int i = 0; i < V*m; ++i) C[i] *= (1.0 - lr * wd);
        for (int i = 0; i < h*n*m; ++i) H[i] *= (1.0 - lr * wd);
        for (int i = 0; i < (V/comm_sz)*h; ++i) local_U[i] *= (1.0 - lr * wd);

        t++;
        
        lr = lr / (1.0 + r * t);
           
        free(local_gradient_Ly);
        free(local_gradient_La);
        free(local_gradient_Lo);
        free(gradient_La);
    }
    
    MPI_Finalize();

    return 0;
}