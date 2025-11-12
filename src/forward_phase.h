#ifndef FORWARD_PHASE
#define FORWARD_PHASE

void forward_phase(int rank, int block_size, int n, int m, int h, int* ids, double* C, double* H, double* d, double* local_p, double* x, double *o, double *local_y, double* U, double* b, double* local_U, double* local_b);

#endif