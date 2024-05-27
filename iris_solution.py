from typing import Tuple, List, Dict
import random

from sklearn.datasets import load_iris
import jax
from jax.dtypes import prng_key
import jax.numpy as jnp

LEARNING_RATE = 0.003

def load_iris_dataset(shuffle: bool = True) -> Tuple[jax.Array, jax.Array]:
    """
    Load the Iris dataset and return the data and labels as JAX arrays.

    Args:
        shuffle (bool): Whether to shuffle the data. Default is True.
    
    Returns:
        data (jax.Array): The input data as a JAX array with shape [150, 4].
        label (jax.Array): The labels as a JAX array with shape [150]. 3 categories(0, 1, 2).
    """
    iris = load_iris()
    data = jnp.array(iris.data)
    label = jnp.array(iris.target)
    label_one_hot = jax.nn.one_hot(label, 3)

    if shuffle:
        shuffle_idx = [i for i in range(data.shape[0])]
        random.shuffle(shuffle_idx)
        data = data[jnp.array(shuffle_idx)]
        label_one_hot = label_one_hot[jnp.array(shuffle_idx)]
    return data, label_one_hot

def create_model(rng: prng_key) -> List[Dict[str, jax.Array]]:
    rng_keys = jax.random.split(rng, 4)
    model = [
        {'weight': jax.random.normal(rng_keys[0], (4, 32)), 'bias': jax.random.normal(rng_keys[1], (32,))},
        {'weight': jax.random.normal(rng_keys[2], (32, 3)), 'bias': jax.random.normal(rng_keys[3], (3,))}
    ]
    return model

@jax.jit
def forward(
    model: List[Dict[str, jax.Array]],
    x: jax.Array
    ) -> jax.Array:
    output = None
    for layer in model[:-1]:
        output = jnp.dot(x, layer['weight']) + layer['bias']
        output = jax.nn.relu(output)
    
    output = jnp.dot(output, model[-1]['weight']) + model[-1]['bias']
    return output

@jax.jit
def cross_entropy(
    model: List[Dict[str, jax.Array]],
    x: jax.Array,
    y: jax.Array,
    ) -> float:
  preds = forward(model, x)
  preds = jax.nn.log_softmax(preds)
  return -jnp.mean(preds * y)

@jax.jit
def update(
    params: List[Dict[str, jax.Array]],
    x: jax.Array,
    y: jax.Array,
    ) -> Tuple[float, List[Dict[str, jax.Array]]]:
    loss, grads = jax.value_and_grad(cross_entropy)(params, x, y)
    return loss, jax.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


if __name__=='__main__':
    num_epoch = 5
    batch_size = 30

    data, one_hot_label = load_iris_dataset(shuffle=True)

    subrng = jax.random.key(0)
    rng, subrng = jax.random.split(subrng, 2)
    model = create_model(rng)
    batch_forward = jax.vmap(forward, in_axes=[None, 0])

    batch_indices = [jnp.arange(i, i+batch_size) for i in range(data.shape[0]//batch_size)]
    for epoch in range(num_epoch):
        epoch_loss = 0
        for batch_index in batch_indices:
            batch_data, batch_one_hot_label = data[batch_index], one_hot_label[batch_index]
            loss, model = update(model, batch_data, batch_one_hot_label)
            epoch_loss += loss
        print(f"epoch: {epoch}/{num_epoch}\t loss: {loss}")


