// built-in files
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

// external files
#include "get_data.h"

void generate_tokens(int rank, int max_tokens, int n, int m, int h, int V, Vocab* vocab, double* C, double* H, double* d, double* U, double* b) {
    int tokens[max_tokens]; // store generated tokens
    int token_count = 0;

    // initialize input tokens with zeros (empty space)
    int ids[2] = {0, 0}; 

    double* x = malloc(n*m * sizeof(double));
    double* o = malloc(h * sizeof(double));
    double* y = malloc(V * sizeof(double));
    double* p = malloc(V * sizeof(double));

    while (token_count < max_tokens) {
        // build input embedding
        for (int i = 0; i < n; i++) {
            int id = ids[i];
            for (int j = 0; j < m; j++) {
                x[i*m + j] = C[id*m + j];
            }
        }

        // hidden layer
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            h, n*m,
            1.0,
            H, n*m,
            x, 1,
            0.0,
            o, 1
        );
        for (int i = 0; i < h; i++) o[i] = tanh(o[i] + d[i]);

        // output layer
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            V, h,
            1.0,
            U, h,
            o, 1,
            0.0,
            y, 1
        );

        for (int i = 0; i < V; i++) y[i] += b[i];

        // softmax
        double max_y = -INFINITY;
        for (int i = 0; i < V; i++) if (y[i] > max_y) max_y = y[i];
        double sum_exp = 0.0;

        for (int i = 0; i < V; i++) {
            p[i] = exp(y[i] - max_y);
            sum_exp += p[i];
        }

        for (int i = 0; i < V; i++) p[i] /= sum_exp;

        // sample next token
        double r = ((double)rand() / RAND_MAX);
        double cumulative = 0.0;
        int next_id = 0;

        for (int i = 0; i < V; i++) {
            cumulative += p[i];

            if (r < cumulative) {
                next_id = i;

                break;
            }
        }

        tokens[token_count++] = next_id;

        ids[0] = ids[1];
        ids[1] = next_id;
    }

    // print generated tokens
    printf("rank %d: ", rank);

    for (int i = 0; i < token_count; i++) {
        printf("%s ", vocab->words[tokens[i]]);
    }

    printf("\n");

    // free memory
    free(x); free(o); free(y); free(p);
}
