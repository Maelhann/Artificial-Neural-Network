#include "layer.h"

/* The sigmoid function and derivative. */
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidprime(double x) {
    return x * (1 - x);
}

/* Creates a single layer. */
layer_t *layer_create() {// all the pointers within a layer are initialized to NULL,
    // all the integer values are 0.
    layer_t *newLayer = malloc(sizeof(layer_t));
    if (newLayer == NULL) {
        printf("error, layer allocation failed");
        return NULL;
    } else {
        newLayer->prev = NULL;
        newLayer->next = NULL;
        newLayer->biases = NULL;
        newLayer->weights = NULL;
        newLayer->deltas = NULL;
        newLayer->outputs = NULL;
        newLayer->num_inputs = 0;
        newLayer->num_outputs = 0;
        return newLayer;
    }
}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev) {
    layer = layer_create();
    layer->num_outputs = num_outputs;
    layer->outputs = malloc(num_outputs * sizeof(double_t));
    if (prev != NULL) {
        layer->prev = prev;
        layer->num_inputs = prev->num_outputs;
        layer->biases = malloc(layer->num_inputs * sizeof(double_t));
        layer->deltas = malloc(layer->num_inputs * sizeof(double_t));
        layer->weights = malloc(layer->num_inputs * sizeof(double_t *));
        for (int i = 0; i < sizeof(layer->weights) / sizeof(double_t *); i++) {
            layer->weights[i] = malloc(layer->num_inputs * sizeof(double_t));
            for (int j = 0; j < sizeof(layer->num_inputs) / sizeof(double_t); j++) {
                layer->weights[i][j] = ANN_RANDOM();
            }
          }
        }
     
    return false;

}

/* Frees a given layer. */
void layer_free(layer_t *layer) {
    /**** PART 1 - QUESTION 4 ****/

    /* 2 MARKS */
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer) {
    /**** PART 1 - QUESTION 5 ****/
    /* objective: compute layer->outputs */

    /* 3 MARKS */
}

/* Computes the delta errors for this layer. */
void layer_compute_deltas(layer_t const *layer) {
    /**** PART 1 - QUESTION 6 ****/
    /* objective: compute layer->deltas */

    /* 2 MARKS */
}

/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate) {
    /**** PART 1 - QUESTION 7 ****/
    /* objective: update layer->weights and layer->biases */

    /* 1 MARK */
}
