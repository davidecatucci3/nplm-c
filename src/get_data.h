#ifndef GET_DATA
#define GET_DATA

typedef struct {
    char *chars;
    int size;
    int capacity;
} Vocab;

void init_vocab(Vocab *vocab);
void build_vocab(const char *filename, Vocab *vocab);
int char_to_id(Vocab *vocab, char c);
int* get_data(Vocab *vocab);

#endif