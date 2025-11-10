#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

// ---------------- Vocabulary Structure ----------------

static FILE *fp = NULL;
static int eof_reached = 0;

typedef struct {
    char **words;   // dynamically allocated array of unique words
    int size;       // number of unique words
    int capacity;   // allocated capacity
} Vocab;

void reset_get_chunk(void) {
    if (fp) {
        fclose(fp);
        fp = NULL;
    }
    eof_reached = 0;
}

void get_chunk(const char *filename, int *x1, int *x2, int *y) {
    if (eof_reached) {
        *x1 = *x2 = *y = -1;
        return;
    }

    if (fp == NULL) {
        fp = fopen(filename, "r");
        if (!fp) {
            perror("fopen");
            *x1 = *x2 = *y = -1;
            eof_reached = 1;
            return;
        }
    }

    // Skip blank lines and check EOF explicitly
    int items = fscanf(fp, "%d %d %d", x1, x2, y);

    if (items != 3) {
        *x1 = *x2 = *y = -1;
        eof_reached = 1;
        fclose(fp);
        fp = NULL;
        return;
    }
}

// ---------------- Helper Functions ----------------

// Initialize vocab
void init_vocab(Vocab *vocab) {
    vocab->size = 0;
    vocab->capacity = 16;
    vocab->words = malloc(vocab->capacity * sizeof(char *));
    if (!vocab->words) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

// Add a word if not already present
int add_word(Vocab *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++) {
        if (strcmp(vocab->words[i], word) == 0)
            return i; // already exists
    }

    if (vocab->size >= vocab->capacity) {
        vocab->capacity *= 2;
        char **tmp = realloc(vocab->words, vocab->capacity * sizeof(char *));
        if (!tmp) {
            fprintf(stderr, "Realloc failed\n");
            exit(EXIT_FAILURE);
        }
        vocab->words = tmp;
    }

    vocab->words[vocab->size] = strdup(word);
    if (!vocab->words[vocab->size]) {
        fprintf(stderr, "strdup failed\n");
        exit(EXIT_FAILURE);
    }

    return vocab->size++;
}

// Get ID of a word (returns -1 if not found)
int word_to_id(Vocab *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++) {
        if (strcmp(vocab->words[i], word) == 0)
            return i;
    }
    return -1;
}

// ---------------- Vocabulary Builder ----------------

// Normalize word (lowercase + strip punctuation)
void normalize_word(char *word) {
    for (int i = 0, j = 0; word[i]; i++) {
        if (isalnum((unsigned char)word[i]) || word[i] == '\'' || word[i] == '_')
            word[j++] = tolower((unsigned char)word[i]);
        word[j] = '\0';
    }
}

// Build vocabulary from file
void build_vocab(const char *filename, Vocab *vocab) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char word[256];
    while (fscanf(fp, "%255s", word) == 1) {
        normalize_word(word);
        if (strlen(word) > 0)
            add_word(vocab, word);
    }

    fclose(fp);
}

// ---------------- Data Sampler ----------------

// Return 3 random word IDs from a random line
int* get_data(Vocab *vocab) {
    static int ids[3];
    FILE *fp = fopen("data/brown.csv", "r");
    if (!fp) {
        perror("Error opening file");
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    // Count total lines
    int line_count = 0;
    char line[8192];
    while (fgets(line, sizeof(line), fp))
        line_count++;

    if (line_count == 0) {
        fclose(fp);
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    // Choose a random line
    int target = rand() % line_count;

    rewind(fp);
    int current = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (current++ == target) break;
    }
    fclose(fp);

    // Tokenize the selected line
    char *words[512];
    int count = 0;
    char *token = strtok(line, " \t\r\n,.;:!?\"()");
    while (token && count < 512) {
        normalize_word(token);
        if (strlen(token) > 0)
            words[count++] = token;
        token = strtok(NULL, " \t\r\n,.;:!?\"()");
    }

    if (count < 3) {
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    // Pick a random starting index for 3 consecutive words
    int start = rand() % (count - 2);

    ids[0] = word_to_id(vocab, words[start]);
    ids[1] = word_to_id(vocab, words[start + 1]);
    ids[2] = word_to_id(vocab, words[start + 2]);

    // Handle case where any word isn’t in vocab
    for (int i = 0; i < 3; i++) {
        if (ids[i] == -1) {
            ids[0] = ids[1] = ids[2] = -1;
            break;
        }
    }

    return ids;
}