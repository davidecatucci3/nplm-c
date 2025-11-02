// built-in files
#include <stdio.h>

// external files
#include "model.h"

int main() {
    Model mo;

    model_init(&mo, 512, 32, 16, 2);
 
    int ids[] = {33, 15};

    forward(&mo, ids);
}