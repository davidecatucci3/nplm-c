# nplm-c

A Neural Probabilistic Language Model (NPLM) implementation in C, designed for high performance and parallelized training using MPI (Message Passing Interface).

## Description

This project implements a neural network-based language model. It is written in C and leverages MPI to distribute the training process across multiple processes, allowing for efficient training on large datasets. The model architecture includes an embedding layer, a hidden layer, and an output layer, trained using stochastic gradient descent with backpropagation.

## Prerequisites

To build and run this project, you need the following installed on your system:

*   **C Compiler**: GCC or Clang.
*   **MPI Implementation**: OpenMPI, MPICH, or similar (e.g., `mpicc`).
*   **OpenBLAS**: For optimized linear algebra operations.

## Installation & Compilation

The project can be compiled with or without MPI support.

### With MPI (Recommended)

Use `mpicc` to compile the project. Ensure you link against OpenBLAS.

```bash
mpicc -I./src -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    -o train src/*.c -lopenblas -lm
```

### Without MPI (Single Process)

If you do not have MPI or wish to run on a single process:

```bash
gcc -I./src -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    -o train src/*.c -lopenblas -lm
```

*Note: Adjust the OpenBLAS paths (`-I` and `-L`) according to your system's installation location.*

## Usage

To train the model using MPI, use the `mpirun` command.

```bash
mpirun -n <number_of_processes> ./train
```

**Example:** Run with 8 processes:

Due to overhead (frequently all_reduce operations) using an high number of process is not faster

```bash
mpirun -n 8 ./train
```

## Data

The project expects the following data files in the `data/` directory:

*   `data/brown.csv`: The vocabulary source file.
*   `data/train_ids.txt`: Training data (token IDs).
*   `data/test_ids.txt`: Test data (token IDs).

## Configuration

Hyperparameters are currently hardcoded in `src/train.c`. You can modify the following variables in the `main` function to tune the model:

*   `epochs`: Number of training epochs (default: 10).
*   `V`: Vocabulary size (default: 6408). **Must be divisible by the number of MPI processes.**
*   `m`: Embedding size (default: 64).
*   `h`: Hidden layer units (default: 32).
*   `n`: Input context size (default: 2).
*   `lr`: Initial learning rate (default: 1e-3).

## Project Structure

The source code is located in the `src/` directory:

*   `train.c`: Main entry point containing the training loop and MPI orchestration.
*   `embedding_matrix.c/h`: Functions for initializing embedding matrices.
*   `forward_phase.c/h`: Implementation of the forward propagation pass.
*   `backward_phase.c/h`: Implementation of the backward propagation (gradient computation).
*   `get_data.c/h`: Utilities for reading data chunks.
*   `generate_tokens.c/h`: Functions for generating text from the trained model.
