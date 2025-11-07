// built-in files
#include <stdbool.h>
#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// external files
#include "embedding_matrix.h"
#include "get_data.h"

int main() {
    // create vocabulary
    Vocab vocab;
    
    init_vocab(&vocab);
    build_vocab("data/brown.csv", &vocab);

    // hyperparameters
    int epochs = 500;
    int V = 6408;           // padded for divisibility
    int m = 64;             // embedding size
    int h = 32;             // hidden layer units
    int n = 2;              // input elements
    double lr = 1e-3;       // learning rate
    double r = 1e-8;        // learning rate decay factor
    double wd = 1e-4;       // weight decay

    // parameters
    double* C = embedding_matrix(V, m);       // embeddings
    double* H = embedding_matrix(h, n*m);     // hidden weights
    double* d = embedding_matrix(h, 1);       // hidden bias
    double* U = embedding_matrix(V, h);       // output weights
    double* b = embedding_matrix(V, 1);       // output bias

    double* y_logits = malloc(V * sizeof(double));
    double* p = malloc(V * sizeof(double));

    double* grad_Ly = malloc(V * sizeof(double));
    double* grad_La = malloc(h * sizeof(double));
    double* grad_Lx = malloc(n * m * sizeof(double));
    double* grad_Lo = malloc(h * sizeof(double));

    srand(time(NULL));
    long long t = 0;

    // training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        int x1, x2, y;
        int count = 0;
        double loss_sum = 0.0;

        reset_get_chunk();

        clock_t start_time = clock();

        while (1) {
            get_chunk(&x1, &x2, &y);
            if (x1 == -1) break;      // end of file
            if (count == 1000) break; // sample limit per epoch

            int ids[2] = {x1, x2};
            int next_id = y;

            double* x = malloc(n * m * sizeof(double));
            for (int i = 0; i < n; i++) {
                int id = ids[i];
                for (int j = 0; j < m; j++) {
                    x[i * m + j] = C[id * m + j];
                }
            }

            double* o = malloc(h * sizeof(double));

            cblas_dgemv(CblasRowMajor, CblasNoTrans, h, n*m,
                        1.0, H, n*m, x, 1, 0.0, o, 1);

            for (int i = 0; i < h; i++)
                o[i] = tanh(o[i] + d[i]);

            // output logits: y_logits = U * o + b
            cblas_dgemv(CblasRowMajor, CblasNoTrans, V, h,
                        1.0, U, h, o, 1, 0.0, y_logits, 1);
            for (int i = 0; i < V; i++) y_logits[i] += b[i];

            // softmax
            double max_y = y_logits[0];
            for (int i = 1; i < V; i++)
                if (y_logits[i] > max_y) max_y = y_logits[i];

            double sum_exp = 0.0;
            for (int i = 0; i < V; i++) {
                p[i] = exp(y_logits[i] - max_y);
                sum_exp += p[i];
            }
            for (int i = 0; i < V; i++)
                p[i] /= sum_exp;

            // loss
            double prob_target = p[next_id];
            double nll = -log(prob_target + 1e-12);
            loss_sum += nll;
            count++;

            // BACKWARD PHASE
            for (int i = 0; i < V; i++)
                grad_Ly[i] = p[i];
            grad_Ly[next_id] -= 1.0; // dL/dy

            for (int i = 0; i < h; i++) grad_La[i] = 0.0;
            for (int i = 0; i < V; i++) {
                double g = grad_Ly[i];
                b[i] -= lr * g;
                for (int j = 0; j < h; j++) {
                    grad_La[j] += g * U[i*h + j];
                    U[i*h + j] -= lr * g * o[j];
                }
            }

            for (int k = 0; k < h; k++)
                grad_Lo[k] = (1.0 - o[k]*o[k]) * grad_La[k];

            for (int i = 0; i < n*m; i++) {
                grad_Lx[i] = 0.0;
                for (int k = 0; k < h; k++)
                    grad_Lx[i] += H[k*n*m + i] * grad_Lo[k];
            }

            for (int k = 0; k < h; k++) {
                d[k] -= lr * grad_Lo[k];
                for (int i = 0; i < n*m; i++)
                    H[k*n*m + i] -= lr * grad_Lo[k] * x[i];
            }

            for (int k = 0; k < n; k++) {
                int word_id = ids[k];
                for (int j = 0; j < m; j++)
                    C[word_id*m + j] -= lr * grad_Lx[k*m + j];
            }

            // weight decay
            for (int i = 0; i < V*m; i++) C[i] *= (1.0 - lr * wd);
            for (int i = 0; i < h*n*m; i++) H[i] *= (1.0 - lr * wd);
            for (int i = 0; i < V*h; i++) U[i] *= (1.0 - lr * wd);

            t++;
            lr = lr / (1.0 + r * t);

            free(x);
            free(o);
        }

        clock_t end_time = clock();

        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        double avg_loss = (count > 0) ? (loss_sum / count) : 0.0;

        printf("Epoch %d | Avg loss = %.6f | lr = %.6e | time = %.3f s\n", epoch, avg_loss, lr, epoch_time);
    }

    free(C); free(H); free(d); free(U); free(b);
    free(y_logits); free(p);
    free(grad_Ly); free(grad_La); free(grad_Lx); free(grad_Lo);

    return 0;
}
