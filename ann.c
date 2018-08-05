#include "ann.h"

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int *layer_outputs)
{
   ann_t* network = malloc(sizeof(ann_t));
   if(network == NULL){
   return NULL ; 
   }
   network->input_layer = layer_create() ;

   layer_init(network->input_layer,layer_outputs[1],NULL); 
   network->input_layer->num_inputs = layer_outputs[0];  
   
   layer_t* curr = layer_create() ;
   layer_t* prev = network->input_layer;    

   int i = 1 ; 
   while(i < num_layers){
     layer_init(curr,layer_outputs[i],prev); 
     prev->next = curr ; 
     curr->next = layer_create();
     prev = curr;
     curr = curr->next ; 
     i++;     
   } 
  network->output_layer = prev ; 
  return network ;

 }

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{ 
   layer_t* curr = ann->input_layer->next ; 
   while(curr->next != NULL){
   layer_free(curr->prev); 
   curr = curr->next;
   } 
   layer_free(curr); 
   free(ann); 
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
  for(int i = 0 ; i < ann->input_layer->num_outputs ;i++){
   ann->input_layer->outputs[i] = inputs[i]; 
  }
  layer_t* curr = ann->input_layer->next ; 
  while(curr->next != NULL){
   layer_compute_outputs(curr); 
   curr = curr->next ; 
  }
  layer_compute_outputs(ann->output_layer);

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
  for(int j = 0 ; j < ann->output_layer->num_outputs ;j++){
  ann->output_layer->deltas[j] = sigmoidprime(ann->output_layer
  ->outputs[j])	*(targets[j] - ann->output_layer->outputs[j]) ;
  }
  
  layer_t* curr = ann->output_layer->prev ; 
  while(curr->prev != NULL){
     layer_compute_deltas(curr); 
     curr = curr->prev ;  
  } 
  curr = ann->input_layer->next; 
  while(curr->next != NULL){
   layer_update(curr,l_rate); 
   curr = curr->next;
  } 
  layer_update(ann->output_layer,l_rate);  
}
