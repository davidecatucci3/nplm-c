# nplm-c
⚠️ **Important:** This project is in progress (not finished) until there is this disclaimer, even if there are files uploaded

## 🧠 Overview

This project is a **C-language reimplementation** of **“A Neural Probabilistic Language Model” (Bengio et al., 2003)** — one of the first works to introduce **neural networks for language modeling**. 

The model learns to **predict the next word** in a sequence by mapping discrete word indices into **continuous embeddings**, feeding their concatenated vectors through a **tanh-activated hidden layer**, and computing a **softmax output** over the vocabulary.  
During training, it jointly learns both:
- **Word embeddings**, capturing semantic similarity between words.  
- **Neural network parameters**, modeling how words combine to form meaningful contexts.  

To speed-up the computation, this implementation uses **MPI-based parallelization**:
- The **output layer (softmax)** is **distributed across MPI processes**, each handling a slice of the vocabulary.
- Processes compute **local logits and probabilities**, then synchronize global results using `MPI_Allreduce` and `MPI_Allgather`.
- Gradients and parameter updates are shared across processes for consistent learning.

I created this project for **two main reasons**.  

First, out of **interest in the research**: I wanted to implement this paper because it was one of the first to explore language modeling using modern neural networks—a topic that genuinely fascinated me.  

Second, as a **university project requirement**: My university required me to complete a project and implement it using MPI and CUDA, so I thought this would be a good one
  
If you want to learn more, **read the official paper**: 📄 [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## 📂 Project Structure

```
├── src/
│   ├── train.c               # Main training and MPI logic
│   ├── embedding_matrix.c    # Embedding matrix implementation
│   ├── embedding_matrix.h    # Embedding matrix header
│   ├── get_data.c            # Data loading and processing implementation
│   └── get_data.h            # Data loading header
├── data/
│   └── brown.csv             # Input text corpus
├── docs/                     # Documentation files 
```

## 🚀 MPI Parallelization

This implementation uses **parameter parallelization** — the vocabulary and its output weights are divided among MPI processes, but **each process maintains its own copy of the model parameters**.  

- Each MPI process holds a **different subset of the output layer (U, b)**.  
- All processes perform **forward and backward computations independently**,  
  but **only share partial results needed for softmax normalization** (e.g., local exponentials and their sums).  
- The global probability distribution is reconstructed via collective MPI operations such as `MPI_Allreduce` and `MPI_Allgather`.  

This means the training is **loosely coupled**:
- Parameters are **not synchronized** at each iteration,  
- Only the **final softmax outputs** are shared across ranks.  

Such a design makes it easy to scale across vocabularies or test distributed efficiency without the complexity of full gradient synchronization — demonstrating **a lightweight form of parallelism** suitable for research, experimentation, or educational exploration of distributed neural networks.
