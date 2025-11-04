#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Dynamic vocabulary structure
typedef struct {
    char *chars;   // dynamically allocated array of unique chars
    int size;      // number of unique characters
    int capacity;  // allocated capacity
} Vocab;

// Initialize vocab
void init_vocab(Vocab *vocab) {
    vocab->size = 0;
    vocab->capacity = 16;
    vocab->chars = (char *)malloc(vocab->capacity * sizeof(char));
    if (!vocab->chars) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
}

// Add new character to vocab if not already present
int add_char(Vocab *vocab, char c) {
    // check if already exists
    for (int i = 0; i < vocab->size; i++) {
        if (vocab->chars[i] == c)
            return i; // already exists
    }

    // grow array if needed
    if (vocab->size >= vocab->capacity) {
        vocab->capacity *= 2;
        vocab->chars = (char *)realloc(vocab->chars, vocab->capacity * sizeof(char));
        if (!vocab->chars) {
            fprintf(stderr, "Realloc failed\n");
            exit(1);
        }
    }

    vocab->chars[vocab->size] = c;
    return vocab->size++;
}

// Get character ID (returns -1 if not found)
int char_to_id(Vocab *vocab, char c) {
    for (int i = 0; i < vocab->size; i++)
        if (vocab->chars[i] == c)
            return i;
    return -1;
}

// Build vocabulary dynamically by reading the whole file
void build_vocab(const char *filename, Vocab *vocab) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        exit(1);
    }

    char c;
    while ((c = fgetc(fp)) != EOF) {
        add_char(vocab, c);
    }

    fclose(fp);
}

// Example get_data() returning a few character IDs
int* get_data(Vocab *vocab) {
    static int ids[3];

    FILE *fp = fopen("data/brown.csv", "r");
    if (!fp) {
        perror("Error opening file");
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    // Read a random line
    int line_count = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp))
        line_count++;

    if (line_count <= 1) {
        fclose(fp);
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    srand(time(NULL));
    int target = (rand() % (line_count - 1)) + 1;

    rewind(fp);
    int current = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (current++ == target) break;
    }
    fclose(fp);

    line[strcspn(line, "\r\n")] = 0;

    if (strlen(line) < 3) {
        ids[0] = ids[1] = ids[2] = -1;
        return ids;
    }

    ids[0] = char_to_id(vocab, line[0]);
    ids[1] = char_to_id(vocab, line[1]);
    ids[2] = char_to_id(vocab, line[2]);
    return ids;
}