from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
import tensorflow as tf
import sonnet as snt


EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}



class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, latent_sizes_edge, latent_sizes_node, latent_sizes_global, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    
    self.edge_fun = lambda : snt.Sequential([
      snt.nets.MLP(latent_sizes_edge, activate_final=True),
      snt.LayerNorm()
      ])
    self.node_fun = lambda : snt.Sequential([
      snt.nets.MLP(latent_sizes_node, activate_final=True),
      snt.LayerNorm()
      ])
    self.global_fun = lambda : snt.Sequential([
      snt.nets.MLP(latent_sizes_global, activate_final=False)
      ])
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(self.edge_fun,
      									self.node_fun,
                                        self.global_fun, 
                                           edge_block_opt=EDGE_BLOCK_OPT,
                                           node_block_opt=NODE_BLOCK_OPT,
                                           global_block_opt=GLOBAL_BLOCK_OPT
                                           )

  def _build(self, inputs):
    return self._network(inputs)


class EncodeProcess(snt.AbstractModule):
  """Full encode-process-decode model.
  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  
            |         |  |  |      |  |  
  Input --->| Encoder |  *->| Core |--*-> Output(t)
            |         |---->|      |     
            *---------*     *------*     
  """

  def __init__(self, latent_sizes_edge, latent_sizes_node, latent_sizes_global, name="EncodeProcess"):
    super(EncodeProcess, self).__init__(name=name)
    self._encoder = MLPGraphNetwork(latent_sizes_edge, latent_sizes_node, latent_sizes_global)
    self._core = MLPGraphNetwork(latent_sizes_edge, latent_sizes_node, latent_sizes_global)
 	
  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
    return latent



class Q_network(snt.AbstractModule):

    def __init__(self,hidden_size, name="Q_network" ):
        super(Q_network, self).__init__(name="Q_network")
        self.hidden_size = hidden_size
        with self._enter_variable_scope():
          self.network = snt.Sequential([
          snt.nets.MLP([self.hidden_size,int(self.hidden_size/2)] , activate_final=True),
          snt.nets.MLP([1])
          ])

    def _build(self, state, action):
        return self.network(tf.concat([state,action],axis=1))
        
    
    def map_(self, state, action):
        st = utils_tf.repeat(state,[tf.shape(action)[0]])
        return self.network(tf.concat([st,action],axis=1))
    	#st = utils_tf.repeat(state,n_node)
    	#return self.network(tf.concat([st,action],axis=1))

    	
