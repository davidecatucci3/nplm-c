#ifndef GET_DATA
#define GET_DATA

typedef struct {
    char **words;
    int size;
    int capacity;
} Vocab;

void init_vocab(Vocab *vocab);
void build_vocab(const char *filename, Vocab *vocab);
void get_chunk(int *x1, int *x2, int *y);

#endif