#include "ann.h"

/* Creates and returns a new artificial neural net */
ann_t *ann_create(int num_layers, int *layer_outputs)
{ 
  ann_t *network = malloc(sizeof(ann_t)) ;
  network->input_layer = layer_create() ;
  layer_init(network->input_layer,layer_outputs[1],NULL) ; 
  int i =  2 ;
  layer_t* prev = network->input_layer ; 
  layer_t* curr = layer_create() ; 
     while(i < num_layers){	  
      layer_init(curr,layer_outputs[i],prev);
      prev->next = curr; 
       if(i == num_layers-1){
        network->output_layer = curr ;
        break; 
       }else{
        prev = curr ; 
        curr = layer_create(); 
        i++ ;
   }  
  }
  return network;
}

/* Frees the space allocated to ann. */ 
void ann_free(ann_t *ann)
{
  layer_t* curr = ann->input_layer->next;  
  while(curr->next != NULL){
    layer_free(curr->prev);
    curr = curr->next;  
  }
  free(ann);
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
 
}

/* Trains the ann with single backprop update. */
void ann_train(ann_t const *ann, double const *inputs, double const *targets, double l_rate)
{
  /* Sanity checks. */
  assert(ann != NULL);
  assert(inputs != NULL);
  assert(targets != NULL);
  assert(l_rate > 0);

  /* Run forward pass. */
  ann_predict(ann, inputs);

  /**** PART 2 - QUESTION 4 ****/

  /* 3 MARKS */
}
