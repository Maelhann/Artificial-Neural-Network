#include "layer.h"

/* The sigmoid function and derivative. */
double sigmoid(double x)
{
  return 1 / (1 + exp(-x)) ;  
}

double sigmoidprime(double x)
{
  return x*(1 - x);
}

/* Creates a single layer. */
layer_t *layer_create()
{
   layer_t* ptr = malloc(sizeof(layer_t)); 
   ptr->outputs = NULL; 
   ptr->prev = NULL; 
   ptr->next = NULL; 
   ptr->weights = NULL; 
   ptr->biases = NULL; 
   ptr->deltas = NULL;  
   ptr->num_inputs = 0; 
   ptr->num_outputs = 0;
}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev)
{
  
  layer->num_outputs = num_outputs; 
  layer->outputs = calloc(num_outputs,sizeof(double));	
  for(int i = 0 ; i < num_outputs;i++){
  layer->outputs[i] = 0 ; 
  }
  if(prev != NULL){
  layer->prev = prev ;
  layer->num_inputs = prev->num_outputs;  
  layer->biases = calloc(prev->num_outputs,sizeof(double));
  layer->deltas = calloc(prev->num_outputs,sizeof(double)); 
  // initialize biases and deltas to zero. 
  for(int j = 0; j < prev->num_outputs;j++){
   layer->biases[j] = 0 ;
   layer->biases[j] = 0 ; 
   }
  layer->weights = calloc(prev->num_outputs,sizeof(double*)); 
  for(int k = 0 ; k < prev->num_outputs ; k++){
     layer->weights[k] = calloc(prev->num_outputs,sizeof(double)); 
     for(int l = 0 ; l < prev->num_outputs; l++){
      layer->weights[k][l] = ANN_RANDOM(); 
     }
  }
  
  }
  return false;  
  
}

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
  /**** PART 1 - QUESTION 4 ****/

  /* 2 MARKS */
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
  /**** PART 1 - QUESTION 5 ****/
  /* objective: compute layer->outputs */

  /* 3 MARKS */
}

/* Computes the delta errors for this layer. */
void layer_compute_deltas(layer_t const *layer)
{
  /**** PART 1 - QUESTION 6 ****/
  /* objective: compute layer->deltas */

  /* 2 MARKS */
}

/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{
  /**** PART 1 - QUESTION 7 ****/
  /* objective: update layer->weights and layer->biases */

  /* 1 MARK */
}
