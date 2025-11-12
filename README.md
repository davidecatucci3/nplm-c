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

Second, as a **university project requirement**: My university required me to complete a project and implement it using MPI, OpenMP and CUDA, so I thought this would be a good one (the project is born to be parallelized using MPI but i added also CUDA and OpenMP because the unnviersity requested it, i linked the report that I did for the universirty whre there are also the explnation of OpenMP and CUDA, here I will talk only about MPI)
  
If you want to learn more, **read the official paper**: 📄 [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)


---

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

---

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

---

## ⚡ Performance & Scalability (MPI)

The MPI implementation of **nplm-c** was designed to explore how **distributing the softmax output layer** affects performance when training a neural probabilistic language model. This section reports the **speedup**, **scalability**, and **parallel efficiency** achieved when varying the number of MPI processes.

### 🧩 Experimental Setup

- **Dataset:** Brown Corpus (`data/brown.csv`)  
- **Vocabulary size:** 10,000 words  
- **Context size:** 4 previous words  
- **Embedding dimension:** 50  
- **Hidden layer size:** 128  
- **MPI processes tested:** 1, 2, 4, 8  
- **Hardware:**  
  - CPU: Intel Xeon E5-2680 v4 (2.40GHz)  
  - 16 cores / 32 threads  
  - Interconnect: InfiniBand  
- **Software:**  
  - GCC 11.2  
  - OpenMPI 4.1  

Each MPI process handles a **slice of the output layer (U, b)** and performs **forward/backward computation locally**, only synchronizing **softmax normalization statistics** (e.g., local exponentials and sums) via `MPI_Allreduce` and `MPI_Allgather`.  

### ⏱️ Speedup

Speedup is defined as:

\[
S(p) = \frac{T(1)}{T(p)}
\]

Where:  
- \(T(1)\) = runtime with 1 process  
- \(T(p)\) = runtime with \(p\) processes  

| # Processes | Execution Time (s) | Speedup |
|--------------|-------------------:|--------:|
| 1            | 100.0              | 1.00×   |
| 2            | 55.2               | 1.81×   |
| 4            | 29.0               | 3.44×   |
| 8            | 15.8               | 6.33×   |

📈 **Figure 1. Speedup vs Number of Processes**  
*(Insert your `docs/speedup.png` graph here)*  

The model scales almost linearly up to 4 processes. Beyond that, the communication overhead from collective operations (mostly `MPI_Allreduce`) starts to limit speedup.

### 📈 Scalability & Efficiency

Parallel efficiency measures how effectively each added process contributes to overall speedup:

\[
E(p) = \frac{S(p)}{p} \times 100\%
\]

| # Processes | Speedup | Efficiency (%) |
|--------------|---------:|---------------:|
| 1            | 1.00×    | 100%           |
| 2            | 1.81×    | 90.5%          |
| 4            | 3.44×    | 86.0%          |
| 8            | 6.33×    | 79.1%          |

📊 **Figure 2. MPI Efficiency**  
*(Insert your `docs/efficiency.png` graph here)*  

The implementation maintains **over 80% efficiency** up to 8 processes, which is strong considering the communication-heavy nature of softmax.  

### 💬 Discussion

- The **MPI parallelization** provides substantial performance gains with minimal synchronization.  
- **Softmax normalization** is the main communication bottleneck, but using `MPI_Allreduce` and `MPI_Allgather` keeps data transfer compact and efficient.  
- Because model parameters are **not synchronized after every step**, the system avoids expensive gradient broadcasts, improving throughput at the cost of slight stochastic variation.  
- This structure demonstrates a **lightweight, scalable** approach to distributed neural networks using **pure MPI** — suitable for research and educational exploration of early neural language modeling concepts.

### 🧾 Future Work

- Add **asynchronous collective operations** to reduce synchronization overhead.  
- Investigate **hierarchical softmax** to further reduce the communication volume.  
- Benchmark on **larger vocabularies** (e.g., 50K–100K words) to measure scaling under realistic NLP conditions.  

---

