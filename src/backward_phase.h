#ifndef BACKWARD_PHASE
#define BACKWARD_PHASE

void backward_phase(int rank, int* ids, int next_id, int block_size, int m, int n, int h, int V, double lr, double wd, double* local_gradient_La, double* local_gradient_Lo, double* C, double* x, double* H, double* o, double* d, double* local_U, double* local_gradient_Ly, double* gradient_Lx, double* gradient_La, double* local_b, double* local_p);

#endif