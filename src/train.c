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
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // process id       
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);   // total number of processes 

    // create vocabulary
    Vocab vocab;

    init_vocab(&vocab);
    build_vocab("data/brown.csv", &vocab);

    // hyperparameters
    int epochs = 500;
    int V = 6408;            // vocab.size is 6402 but to use 8 cores and be divisible i need 6408
    int m = 64;              // embedding size
    int h = 32;              // hidde layer units
    int n = 2;               // input elements
    double lr = 1e-3;        // start learning rate
    double r = 1e-8;         // decreasor rate
    double wd = 1e-4;        // weight decay

    // sanity
    if (V % comm_sz != 0) {
        if (rank == 0) fprintf(stderr, "V (%d) must be divisible by comm_sz (%d)\n", V, comm_sz);

        MPI_Finalize();

        return 0;
    }

    int block = V / comm_sz; // rows per rank

    // parameters
    double* C = embedding_matrix(V, m);       // embedding weights
    double* H = embedding_matrix(h, n*m);     // weights first layer
    double* d = embedding_matrix(h, 1);       // bias first layer
    double* U = embedding_matrix(V, h);       // weights second layer
    double* b = embedding_matrix(V, 1);       // bias second layer

    double* local_U = malloc((V/comm_sz) * h * sizeof(double));     // weights second layer
    double* local_b = malloc((V/comm_sz) * sizeof(double));         // bias second layer
    double* local_y = malloc((V/comm_sz) * sizeof(double));         // output second layer (logits)
    double* local_p = malloc((V/comm_sz) * sizeof(double));         // output second layer (probs)
    double* p = malloc(V * sizeof(double));

    double* local_gradient_Ly = malloc((V/comm_sz) * sizeof(double));
    double* gradient_La = malloc(h * sizeof(double));
    double* gradient_Lx = malloc(n * m * sizeof(double));
    double* local_gradient_La = malloc(h * sizeof(double));
    double* local_gradient_Lo = malloc(h * sizeof(double));

    // other valriables
    long long t = 0;

    // seed
    srand(time(NULL) + rank);

    // initialize local_U and local_b once (parameter-parallel: each rank owns a chunk)
    for (int r = 0; r < block; ++r) {
        int global_row = rank * block + r;

        local_b[r] = b[global_row];

        for (int c = 0; c < h; ++c) {
            local_U[r * h + c] = U[global_row * h + c];
        }
    }

    // traininig loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        int x1, x2, y;
        int count = 0;
        long long local_count = 0;
        double local_loss_sum = 0.0;

        reset_get_chunk();
        
        double start_time = MPI_Wtime();

        while (1) {
            // FORWARD PHASE
            // perform forward computation for the word features layer  
            get_chunk(&x1, &x2, &y);
            
            if (x1 == -1) break;  // end of file
            if (count == 1000) break; // break

            int ids[2] = {x1, x2}; 
            int next_id = y;
            double* x = malloc(n * m * sizeof(double)); //input vector neural network (flattened)

            for (int i = 0; i < n; i++) {
                int id = ids[i];
                
                for (int j = 0; j < m; j++) {
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
                0.0,
                o, 1               
            );

            for (int i = 0; i < h; i++) {
                o[i] = tanh(o[i] + d[i]);
            }

            // perform forward computation for output units in the i-th block
            double S = 0.0;
            double local_s = 0.0;

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

            double local_max = -INFINITY;
            double global_max = 0.0;

            for (int i = 0; i < V / comm_sz; ++i) {
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
            double prob_target = p[next_id];
            double nll = log(prob_target + 1e-12);

            local_loss_sum += nll;
            local_count += 1;
            
            // BACKWARD PHASE         
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
            for (int i = 0; i < (V/comm_sz)*h; ++i) local_U[i] *= (1.0 - lr * wd);

            t++;
            count++;
            
            lr = lr / (1.0 + r * t);
        }

        // data ata each epoch
        double global_loss_sum = 0.0;
        long long global_count = 0;

        MPI_Reduce(&local_loss_sum, &global_loss_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        double end_time = MPI_Wtime();
        double epoch_time = end_time - start_time;

        if (rank == 0) {
            double avg_nll = (global_count > 0) ? (global_loss_sum / (double)global_count) : 0.0;

            printf("Epoch %d | Avg loss = %.6f | lr = %.6e | time = %.3f \n", epoch, avg_nll, lr, epoch_time);
        }
    }

    // free up memory
    free(C); free(H); free(d); free(U); free(b);
    free(local_U); free(local_b); free(local_y); free(local_p); free(p);
    free(local_gradient_Ly); free(gradient_La); free(gradient_Lx);
    free(local_gradient_La); free(local_gradient_Lo);
    
    MPI_Finalize();

    return 0;
}

// T_serial(n) = 1.3
// T_parallel(n, 8) = 2.5
// T_parallel(n, 1) = 1.4
// Sp(n, 8) = 1.3 / 2.5 = 0.52
// Sc(n, 8) = 1.4 / 2.5 = 0.5
