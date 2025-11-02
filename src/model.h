#ifndef MODEL_H
#define MODEL_H

typedef struct {
    double* C; // C embedding matrix 
    int* vocab; // C embedding matrix 
    double* H; // weights first layer
    double* d; // bias first layer
    double* U; // weights second layer
    double* b; // bias second layer
    
} Model;

void model_init(Model *mo, int V, int m, int h, int n);

void forward(Model *mo, int ids[]);

#endif