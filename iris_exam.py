from typing import Tuple
import random

from sklearn.datasets import load_iris
import jax
import jax.numpy as jnp


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


if __name__=='__main__':
    num_epoch = 5
    batch_size = 30

    data, one_hot_label = load_iris_dataset(shuffle=True)

    batch_indices = [jnp.array(i, i+batch_size) for i in range(data.shape[0]//batch_size)]
    for epoch in range(num_epoch):
        for batch_index in batch_indices:
            batch_data, batch_one_hot_label = data[batch_index], one_hot_label[batch_index]
            # make your logic here
            pass


