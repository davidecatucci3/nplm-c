// built-in files
#include <stdlib.h>

// external files
#include "embedding_matrix.h"
#include "model.h"

void model_init(Model *mo, int V, int m, int h, int n) {
    mo->C = embedding_matrix(V, m); // embedding matrix 
    mo->vocab = malloc(V * sizeof(int)); // vocabulary map
    mo->H = embedding_matrix(h, n*m); // weights first layer
    mo->d = embedding_matrix(h, 1); // bias first layer
    mo->U = embedding_matrix(V, h); // weights second layer
    mo->b = embedding_matrix(V, 1); // bias second layer
}
