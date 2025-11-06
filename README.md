# nplm-c
⚠️ **Important:** This project is in progress (not finished) until there is this disclaimer, even if there are files uploaded

## 🧠 Overview
This project is a **C-language reimplementation** of **“A Neural Probabilistic Language Model” (Bengio et al., 2003)** — one of the first works to introduce **neural networks for language modeling**. 

This project was realized for two main reasons **(in order)**:
1. I read this paper about 2 years ago but didn’t fully understand it at the time. Since then, I’ve gained more experience and decided to finally implement it to deepen my understanding.  
2. I also used this project as part of a **university exam on multicore programming**, where I explored how MPI, CUDA and other libraries can be used to parallelize neural network computations.

The model learns to **predict the next word** in a sequence by mapping discrete word indices into **continuous embeddings**, feeding their concatenated vectors through a **tanh-activated hidden layer**, and computing a **softmax output** over the vocabulary.  
During training, it jointly learns both:
- **Word embeddings**, capturing semantic similarity between words, and  
- **Neural network parameters**, modeling how words combine to form meaningful contexts.  

To scale the computation to large vocabularies, this implementation uses **MPI-based parallelization**:
- The **output layer (softmax)** is **distributed across MPI processes**, each handling a slice of the vocabulary.
- Processes compute **local logits and probabilities**, then synchronize global results using `MPI_Allreduce` and `MPI_Allgather`.
- Gradients and parameter updates are shared across processes for consistent learning.

If you want to know more, read the official paper:  
📄 [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
