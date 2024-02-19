## Network Catalogue

| Network   | Description                                      |
|-----------|--------------------------------------------------|
| SimpleNN  | A simple PyTorch neural network with four fully connected layers. The forward method applies the ReLU activation function to the output of each fully connected layer, except for the last layer where it applies the sigmoid activation function. This ensures that the output values are between 0 and 1, representing the probability of a binary classification. |
| nn2       | A PyTorch neural network architecture consisting of four fully connected layers with dropout regularisation. It uses the leaky ReLU activation function for intermediate layers and the sigmoid activation function for the final layer.                     |
| nn3       | A PyTorch neural network consisting of six fully connected layers with both batch normalisation and dropout regularisation to stabilise training. Each fully connected layer is followed by a leaky ReLU activation function. The final output is passed through a sigmoid activation function                   |
| nn4       | A PyTorch deep neural network with seven fully connected layers. Each layer is followed by batch normalization and dropout for stable training. A leaky ReLU activation function is applied after each layer. The network also includes a residual connection after the second layer to facilitate backpropagation. The final output is passed through a sigmoid activation function.                   |
| nn5       | A PyTorch deep neural network similar to nn4, with the addition of a self-attention mechanism[1]. Notably, nn5 integrates a self-attention layer after the second fully connected layer, enhancing the network's ability to focus on relevant features in the input data. The network maintains a residual connection after the second layer to aid in backpropagation. The final output is passed through a sigmoid activation function|
| nn6          | A PyTorch-based deep neural network, ResidualComplexNNwith_MH_attention, extends upon its predecessors by incorporating a multi-headed self-attention mechanism[2]. The architecture also employs residual connections to facilitate effective backpropagation, ensuring that both direct and attention-enhanced representations contribute to the learning process. Following the self-attention layer, a feed-forward network (FFN) block and layer normalisation are applied. |
| GNN1            | A PyTorch Graph Neural Network (GNN) with six graph convolution layers (GCNConv). Each layer is followed by batch normalization and dropout. A leaky ReLU activation function is applied after each layer. The network includes a residual connection after the second layer. The network also incorporates global features, which are processed through two fully connected layers. The outputs of the GNN and the global features are then concatenated and passed through a final fully connected layer. The final output is passed through a sigmoid activation function. This architecture is designed for graph-structured data.|
|    LENN       |  A PyTorch Graph Neural Network (GNN) designed for processing graph-structured data in the context of Lorentz-invariant systems, which are common in physics. The network consists of three main components: LorentzEdgeBlock, LorentzNodeBlock, and GlobalBlock. LorentzEdgeBlock processes edge attributes of the graph. It uses a multilayer perceptron (MLP) and a special function psi to transform the edge attributes. It also uses a Minkowski metric for calculating the inner product of source and destination node features. LorentzNodeBlock processes node features of the graph. It uses two MLPs and the Minkowski metric to transform the node features. It also aggregates edge attributes for each node using a mean operation. GlobalBlock processes the global features of the graph. It uses an MLP to transform the global features. The LorentzInteractionNetwork uses a MetaLayer from PyTorch Geometric, which allows for flexible combination of different types of layers (edge, node, and global). The output of the network is the transformed global features.           |


----

## Reference Notes

### 1. Self-Attention
Self-attention is a mechanism that allows a neural network to focus on different parts of the input sequence when making predictions. It calculates attention weights for each element in the sequence based on its relationship with other elements. This attention mechanism has been widely used in natural language processing tasks, such as machine translation and text summarization, to capture long-range dependencies and improve performance.

To implement self-attention in a neural network, a self-attention layer is added after one or more layers. This layer calculates attention weights by comparing the similarity between each element and all other elements in the sequence. The attention weights are then used to weight the contributions of each element when making predictions.

The procedure is the following:
First, the input x is linearly transformed to create queries (Q), keys (K) and values (V):


```math
  Q = xW^{q}     \, \, \, , \,
  K = xW^{K}      \, \, \, , \,
  V = xW^{V}      \, \, \,
```

Attention scores are computed by taking the dot product of these queries and keys, then followed by some scaling factor:

```math
  \text{Attention Scores} = \frac{QK^{T}}{\sqrt{d_{k}}}
```

This scaling is used to avoid large values during softmax.

The attention scors are then normalised using a softmax to get the softmax scores.

The output of the attention mechanism is computed as a weighted sum of the values, with weights given by softmax scores:

```math
  \text{Output} = \text{Softmax Scores} \times V
```

### 2. Multi-headed self-attention

Multi-headed self-attention is an extension of the self-attention mechanism that allows the model to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function, multi-headed self-attention runs through several attention processes in parallel. This design enables the model to capture a richer array of information and integrate diverse perspectives from the input data.

The multi-headed self-attention mechanism operates as follows:

The input x is linearly transformed N times with different, learned linear projections to N sets of queries (Q), keys (K), and values (V). Each set corresponds to a different "head".
For each head, attention scores are computed by taking the dot product of queries and keys, followed by scaling:

```math
\text{Attenion Scores}_{i} = \frac{Q_{i}K_{i}^{T}}{\sqrt{d_{k}}}
```
where \(d_{k}\) is the dim. of the keys, which ensures the dot procuts do not grow large.

The scores are then normalised using softmax to obtain weights on the values:

```math

\text{Softmax Scores}_{i} = \text{softmax(Attention Scores)}_{i}
```
The output for each head is then computed as the wieghted sum of the values, based on these weights

```math

\text{Output}_{i} = \text{Softmax Scores}_{i} \times V_{i}
```
The outputs of all heads are concatenated and linearly transformed to produce the final ouput of the multi-headed self-attention layers:

```math

\text{Multi-Head Output} = [\text{Output}_{1}; \text{Output}_{2};...;\text{Output}_{N}]W^{O}

```
where \(W^{O}\) is the learned linear projection.

## Performance Metrics

## Glossary
| Term      | Description                                      |
|-----------|--------------------------------------------------|
|||
|||
|||
