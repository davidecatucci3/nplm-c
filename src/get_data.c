#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE 4096
#define MAX_TOKENS 512

// Dummy token → ID mapping (replace with your real tokenizer)
int token_to_id(const char *token) {
    int id = 0;
    for (int i = 0; token[i]; i++)
        id += token[i];
    return id % 1000; // simple hash
}

// Helper: split string by delimiter
int split_tokens(char *str, char *delim, char *tokens[], int max_tokens) {
    int count = 0;
    char *tok = strtok(str, delim);
    while (tok != NULL && count < max_tokens) {
        tokens[count++] = tok;
        tok = strtok(NULL, delim);
    }
    return count;
}

// get_data() → returns pointer to static int[2] with token IDs
int* get_data() {
    static int ids[3];  // static so we can return pointer safely

    FILE *fp = fopen("data/brown.csv", "r");
    
    if (!fp) {
        perror("Error opening file");
        ids[0] = ids[1] = -1;
        return ids;
    }

    // Count lines to pick a random one
    int line_count = 0;
    char line[MAX_LINE];
    while (fgets(line, sizeof(line), fp))
        line_count++;

    if (line_count <= 1) { // header only
        fclose(fp);
        ids[0] = ids[1] = -1;
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

    // Split CSV line (simple version, no quoted commas)
    char *fields[8];
    int n = split_tokens(line, ",", fields, 8);
    if (n < 7) {
        ids[0] = ids[1] = -1;
        return ids;
    }

    char *tokenized_text = fields[4];
    char temp_text[MAX_LINE];
    strncpy(temp_text, tokenized_text, sizeof(temp_text));
    temp_text[sizeof(temp_text)-1] = '\0';

    // Tokenize by spaces
    char *tokens[MAX_TOKENS];
    int num_tokens = split_tokens(temp_text, " ", tokens, MAX_TOKENS);

    if (num_tokens < 2) {
        ids[0] = ids[1] = -1;
        
        return ids;
    }

    // Pick random 2-token chunk
    int start = rand() % (num_tokens - 1);

    ids[0] = token_to_id(tokens[start]);
    ids[1] = token_to_id(tokens[start + 1]);
    ids[2] =  token_to_id(tokens[start + 2]);

    return ids;
}