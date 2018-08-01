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
   layer->deltas[j] = 0 ; 
   }
  layer->weights = calloc(prev->num_outputs,sizeof(double*)); 
  for(int k = 0 ; k < prev->num_outputs ; k++){
     layer->weights[k] = calloc(prev->num_outputs,sizeof(double)); 
     for(int l = 0 ; l < prev->num_outputs; l++){
       layer->weights[k][l] = ANN_RANDOM(); 
     }
  }
   if(layer->weights == NULL ||
      layer->biases == NULL  ||
      layer->deltas == NULL){
   return true;
    }
  }
  
  if(layer->outputs == NULL){
   return true ; 
  }
 return false;  
}

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
  free(layer->next); 
  free(layer->prev); 
  free(layer->outputs);
  free(layer->biases);
  free(layer->deltas); 
  for(int i = 0 ; i < layer->num_inputs ;i++){
    free(layer->weights[i]); 
  }
  free(layer->weights);  
  free(layer);   
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
  double sum = 0 ;
  for(int j = 0 ; j < layer->num_outputs ; j++){
    sum = 0; 
       for(int i = 0 ; i < layer->prev->num_outputs ; i++){
     sum += layer->prev->outputs[i] * layer->weights[i][j]; 
       }
   layer->outputs[j] = sigmoid(layer->biases[j] + sum); 
  }
}

/* Computes the delta errors for this layer
 * uses the back-propagation algorithm */
void layer_compute_deltas(layer_t const *layer)
{   assert(layer != NULL);
    double sum = 0 ; 
    for(int i = 0 ; i < layer->num_inputs; i++){
     sum = 0 ; 
      for(int j = 0 ;  j < layer->num_outputs; j++){
  	 sum += layer->weights[i][j] * layer->next->deltas[j];     
      }
     layer->deltas[i] = sigmoidprime(layer->outputs[i])*sum ;
    }
}

/* Updates weights and biases according
 * to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{ 
  assert(layer != NULL);
  for(int i = 0 ; i < layer->prev->num_inputs ; i++){
    for(int j = 0 ; j < layer->num_inputs ; j++){
    layer->weights[i][j] = layer->weights[i][j] + (l_rate *
	    layer->outputs[i] * layer->deltas[j]);  
    }    
    
  }
  
}
