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
#include "generate_tokens.h"

int main() {
    // initialize MPI
    MPI_Init(NULL, NULL);

    int rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // process number       
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);     // number of processes 

    // seed (for randomicity of parameters values)
    srand(time(NULL) + rank);

    // create vocabulary
    Vocab vocab;

    init_vocab(&vocab);
    build_vocab("data/brown.csv", &vocab);

    // hyperparameters
    int epochs = 500;   
    int B = 64;           // batch size  
    int V = 6408;         // vocab.size is 6402 but to use 8 cores and be divisible I need 6408
    int m = 64;           // embedding size
    int h = 32;           // hidde layer units
    int n = 2;            // input size
    double lr = 1e-3;     // start learning rate
    double r = 1e-8;      // decrease factor
    double wd = 1e-4;     // weight decay

    // check if vocabulary size if divisible by number of processes
    if (V % comm_sz != 0) {
        if (rank == 0) fprintf(stderr, "V (%d) must be divisible by comm_sz (%d)\n", V, comm_sz);

        MPI_Finalize();

        return 0;
    }

    int block_size = V / comm_sz; // each local parameters has this number of rows (rows per rank)

    // parameters
    double* C = embedding_matrix(V, m);       // embedding weights
    double* H = embedding_matrix(h, n*m);     // weights first layer
    double* d = embedding_matrix(h, 1);       // bias first layer
    double* U = embedding_matrix(V, h);       // weights second layer
    double* b = embedding_matrix(V, 1);       // bias second layer

    //input and output variables
    double* x = malloc(n*m * sizeof(double));                  // input vector neural network (flattened)
    double* o = malloc(h * sizeof(double));                    // output vector first layer
    double* local_y = malloc(block_size * sizeof(double));     // output second layer logits (each rank has its own block and it's parallelized, when is local that's the meaning)
    double* local_p = malloc(block_size * sizeof(double));     // output second layer probs

    // gradients matrix
    double* local_gradient_Ly = malloc(block_size * sizeof(double));     
    double* local_gradient_La = malloc(h * sizeof(double));
    double* local_gradient_Lo = malloc(h * sizeof(double));
    double* gradient_La = malloc(h * sizeof(double));
    double* gradient_Lx = malloc(n * m * sizeof(double));

    double t = 0; // used for learning rate decay

    // traininig loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        int x1, x2, y;             // two inputs (x1 and x2) and a third value to predict (y)                 
        int count = 0;             // count samples (chunks) elaborated
        double loss_sum = 0.0;     // sum loss for each sample
        
        reset_get_chunk();
        
        double start_time = MPI_Wtime(); // start timer to calculate time spent for one epoch
 
        while (1) {
            // get train sample of data
            get_chunk("data/train_ids.txt", &x1, &x2, &y);

            //if (x1 == -1) break;        // all samples in train data per epoch
            if (count == 1000) break;     // 1000 samples per epoch

            int ids[2] = {x1, x2};
            int next_id = y;

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

            // update log-likelihood, if wt falls in the block of CPU i > 0, then CPU i sends pwt to CPU 0. CPU 0 computes L = log(pwt) and keeps track of the total log-likelihood
            int local_start = rank * block_size;
            int local_end = local_start + block_size;
            double prob_target = 0.0;

            if (next_id >= local_start && next_id < local_end) {
                int local_index = next_id - local_start;

                prob_target = local_p[local_index];
            }
            
            MPI_Allreduce(MPI_IN_PLACE, &prob_target, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            double ll = log(prob_target + 1e-12); // log-likelihood

            loss_sum += ll;
            count += 1;
            
            // BACKWARD PHASE         
            // perform backward gradient for output units in i-th block
            for (int i = 0; i < h; i++) {
                local_gradient_La[i] = 0.0;
            }

            for (int i = 0; i < n*m; ++i) {
                gradient_Lx[i] = 0.0;
            }
            
            for (int i = 0; i < block_size; i++) {
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
            
            lr = lr / (1.0 + r * t);
        }

        t += 1.0;

        // important at each epoch
        double end_time = MPI_Wtime(); // end timer to calculate time spent for one epoch
        double epoch_time = end_time - start_time;
    
        if (rank == 0) {
            double avg_ll = (loss_sum / count); // avergae loss among all samples

            printf("Epoch %d | train loss = %.6f | lr = %.6e | dt = %.3f \n", epoch, avg_ll, lr, epoch_time);
        } 

        double* tx = malloc(n*m * sizeof(double));
        double* to = malloc(h * sizeof(double));                    
        double* tlocal_y = malloc(block_size * sizeof(double));     
        double* tlocal_p = malloc(block_size * sizeof(double)); 

        // each 25 epochs test
        if (epoch % 25 == 0) {
            int tx1, tx2, ty;           // two inputs (x1 and x2) and a third value to predict (y)                 
            int tcount = 0;             // count samples (chunks) elaborated
            double tloss_sum = 0.0;     // sum loss for each sample   
            
            double tstart_time = MPI_Wtime(); // start timer to calculate time spent for one epoch

            while (1) {
                // get test sample of data
                get_chunk("data/test_ids.txt", &tx1, &tx2, &ty);

                //if (tx1 == -1) break;        // all samples in train data per epoch
                if (tcount == 1000) break;     // 1000 samples per epoch

                int tids[2] = {tx1, tx2};
                int tnext_id = ty;

                // FORWARD PHASE
                // perform forward computation for the word features layer  
                for (int i = 0; i < n; i++) {
                    int tid = tids[i];
                    
                    for (int j = 0; j < m; j++) {
                        tx[i * m + j] = C[tid * m + j];  
                    }
                }
                
                // perform forward computation for the hidden layer
                cblas_dgemv( // BLAS faster matrix mul
                    CblasRowMajor,    
                    CblasNoTrans,    
                    h, n*m,
                    1.0,
                    H, n*m,                    
                    tx, 1,         
                    0.0,
                    to, 1               
                );

                for (int i = 0; i < h; i++) {
                    to[i] = tanh(to[i] + d[i]);
                }

                // perform forward computation for output units in the i-th block
                double tS = 0.0;                                  // total sum of exponential for softmax
                double tlocal_s = 0.0;                            // local exponential for softmax
                double* tlocal_U = U + rank * block_size * h;     // take a block of U for parallelize it over all ranks
                double* tlocal_b = b + rank * block_size;         // take a block of b for parallelize it over all ranks

                cblas_dgemv( // BLAS faster matrix mul
                    CblasRowMajor,     
                    CblasNoTrans,      
                    block_size, h,
                    1.0,
                    tlocal_U, h,           
                    to, 1,         
                    0.0,
                    tlocal_y, 1               
                );

                for (int i = 0; i < block_size; i++) {
                    tlocal_y[i] += tlocal_b[i];
                }

                // softmax stability 
                double tlocal_max = -INFINITY;
                double tglobal_max = 0.0;

                for (int i = 0; i < block_size; ++i) {
                    if (tlocal_y[i] > tlocal_max) {
                        tlocal_max = tlocal_y[i];
                    }
                }

                MPI_Allreduce(&tlocal_max, &tglobal_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

                for (int i = 0; i < block_size; ++i) {
                    tlocal_p[i] = exp(tlocal_y[i] - tglobal_max);

                    tlocal_s += tlocal_p[i];
                }

                // compute and share S among the processors
                MPI_Allreduce(&tlocal_s, &tS, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                // normalize the probabilities
                for (int i = 0; i < block_size; i++) {
                    tlocal_p[i] /= tS;
                }

                // update log-likelihood, if wt falls in the block of CPU i > 0, then CPU i sends pwt to CPU 0. CPU 0 computes L = log(pwt) and keeps track of the total log-likelihood
                int tlocal_start = rank * block_size;
                int tlocal_end = tlocal_start + block_size;
                double tprob_target = 0.0;

                if (tnext_id >= tlocal_start && tnext_id < tlocal_end) {
                    int tlocal_index = tnext_id - tlocal_start;

                    tprob_target = tlocal_p[tlocal_index];
                }
                
                MPI_Allreduce(MPI_IN_PLACE, &tprob_target, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                double tll = log(tprob_target + 1e-12); // log-likelihood

                tloss_sum += tll;
                tcount += 1;
            }

            // important at each epoch
            double tend_time = MPI_Wtime(); // end timer to calculate time spent for one epoch
            double tepoch_time = tend_time - tstart_time;
        
            if (rank == 0) {
                double tavg_ll = (tloss_sum / tcount); // avergae loss among all samples

                printf("Epoch %d | test loss = %.6f | lr = %.6e | dt = %.3f \n", epoch, tavg_ll, lr, epoch_time);
            } 
        }

        // each 50 epochs generate tokens
        if (epoch % 50 == 0) {
            generate_tokens(rank, 30, n, m, h, V, &vocab, C, H, d, U, b);
        }

        // free up memory
        free(tx); free(to); free(tlocal_y); free(tlocal_p);
    }

    // free memory
    free(C); free(H); free(d); free(U); free(b);
    free(x); free(o); free(local_y); free(local_p);
    free(local_gradient_Ly); free(gradient_La); free(gradient_Lx);
    free(local_gradient_La); free(local_gradient_Lo);

    MPI_Finalize();

    return 0;
}
