#include "layer.h"

/* The sigmoid function and derivative. */
double sigmoid(double x)
{
  return (1 / 1 + (1/exp(x))); 
}

double sigmoidprime(double x)
{
  return x*(1 - x);
}

/* Creates a single layer. */
layer_t *layer_create()
{
  layer_t* layer = malloc(sizeof(layer_t)); 
  if(layer == NULL){
   // if alloc fails 
   return NULL ; 
  }else{
  //init all integer vals to zero and all pointers to NULL 
  layer->num_outputs = 0 ; 
  layer->num_inputs = 0 ;
  
  layer->outputs = NULL; 
  layer->prev = NULL ; 
  layer->next = NULL ; 
  layer->weights = NULL ;
  layer->biases = NULL ; 
  layer->deltas = NULL ;

  return layer ;
  } 
}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev)
{ 
  layer->num_outputs = num_outputs ; 
  layer->outputs = malloc(num_outputs*sizeof(double)); 
   if(layer->outputs != NULL){ 
      for(int i = 0 ; i < num_outputs ; i++){
      layer->outputs[i] = 0 ; 
    }	
   }else{
    return true ; 
   }

   if(prev != NULL){
   // if we are not dealing with input layer, then allocate arrays. 
   layer->num_inputs = prev->num_outputs;
   layer->prev = prev ;  
   layer->biases = malloc(layer->num_inputs*sizeof(double)); 
   layer->deltas = malloc(layer->num_inputs*sizeof(double)); 
     if(layer->biases != NULL && layer->deltas != NULL){
      for(int i = 0 ; i < layer->num_inputs ; i++){
      	layer->biases[i] = 0 ; 
        layer->deltas[i] = 0 ;   
      }
     }else{
  	return true;
     }

   layer->weights = malloc(layer->num_inputs * sizeof(double*));
    if(layer->weights != NULL){
     for(int i = 0 ; i < layer->num_inputs ; i++){
       layer->weights[i] = malloc(layer->num_inputs*sizeof(double));
       if(layer->weights[i] == NULL){
 	 return true ; 
       }
        for(int j = 0 ; j < layer->num_inputs ; j++){
         layer->weights[i][j] = ANN_RANDOM();
        } 
     }
    }else{
     return true ; 
    } 
    
  }
  return false; 
}

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
 free(layer->outputs);
 
 if(layer->prev != NULL){
 free(layer->deltas);
 free(layer->biases); 
 
  for(int i = 0 ; i < layer->num_inputs ; i++){
  free(layer->weights[i]); 
  } 
  
  free(layer->weights); 
  layer->prev = NULL ; 
 }
 
 free(layer);
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
   double sum  ;
   for(int j = 0 ; j < layer->num_outputs ; j++){
     sum = 0 ;     
      for(int i = 0 ; i < layer->prev->num_outputs ; i++){
        sum += layer->weights[i][j] * layer->prev->outputs[i];    
      }
     layer->outputs[j] = sigmoid(layer->biases[j] + sum);  
    } 
}

/* Computes the delta errors for this layer. */
void layer_compute_deltas(layer_t const *layer)
{
  double sum ; 
  for(int i = 0 ; i < layer->num_inputs ; i++){
    sum = 0; 
     for(int j = 0 ; j < layer->next->num_inputs;j++){
      sum += layer->next->weights[i][j] * layer->next->deltas[j] ;  
     }
    layer->deltas[i] = sigmoidprime(layer->outputs[i])
			*sum ;   
   }
}

/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{
  for(int j = 0 ; j < layer->num_inputs ; j++){
    layer->biases[j] += l_rate * layer->deltas[j];
  }

  for(int i = 0 ; i < layer->prev->num_inputs ; i++){
   for(int j = 0 ; j < layer->num_inputs ; j++){
    layer->weights[i][j] += l_rate 
              * layer->prev->outputs[i] * layer->deltas[j];   
    }
  }
}
