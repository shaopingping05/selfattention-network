
```bash
pip install keras-tcn
pip install keras-tcn --no-dependencies  # without the dependencies if you already have TF/Numpy.
```

For MacOS M1 users: `pip install --no-binary keras-tcn keras-tcn`. The `--no-binary` option will force pip to download the sources (tar.gz) and re-compile them locally. Also make sure that `grpcio` and `h5py` are installed correctly. There are some tutorials on how to do that online.

## Why TCN (Temporal Convolutional Network) instead of LSTM/GRU?

- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Performs better than LSTM/GRU on long time series (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism (convolutional layers), flexible receptive field size (how far the model can see), stable gradients (compared to backpropagation through time, vanishing gradients)...

<p align="center">
  <img src="misc/Dilated_Conv.png">
  <b>Visualization of a stack of dilated causal convolutional layers (Wavenet, 2016)</b><br><br>
</p>

## TCN Layer

### TCN Class

```python
TCN(
    nb_filters=64,
    kernel_size=3,
    nb_stacks=1,
    dilations=(1, 2, 4, 8, 16, 32),
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.0,
    return_sequences=False,
    activation='relu',
    kernel_initializer='he_normal',
    use_batch_norm=False,
    use_layer_norm=False,
    use_weight_norm=False,
    go_backwards=False,
    return_state=False,
    **kwargs
)
```


### Input shape

3D tensor with shape `(batch_size, timesteps, input_dim)`.

`timesteps` can be `None`. This can be useful if each sequence is of a different length: [Multiple Length Sequence Example](tasks/multi_length_sequences.py).

### Output shape

- if `return_sequences=True`: 3D tensor with shape `(batch_size, timesteps, nb_filters)`.
- if `return_sequences=False`: 2D tensor with shape `(batch_size, nb_filters)`.









### Non-causal TCN

Making the TCN architecture non-causal allows it to take the future into consideration to do its prediction as shown in the figure below.

However, it is not anymore suitable for real-time applications.

<p align="center">
  <img src="misc/Non_Causal.png">
  <b>Non-Causal TCN - ks = 3, dilations = [1, 2, 4, 8], 1 block</b><br><br>
</p>

To use a non-causal TCN, specify `padding='valid'` or `padding='same'` when initializing the TCN layers.

## Run

Once `keras-tcn` is installed as a package, you can take a glimpse of what is possible to do with TCNs. Some tasks examples are available in the repository for this purpose:

```bash
cd adding_problem/
python main.py # run adding problem task

cd copy_memory/
python main.py # run copy memory task

cd mnist_pixel/
python main.py # run sequential mnist pixel task
```

Reproducible results are possible on (NVIDIA) GPUs using the [tensorflow-determinism](https://github.com/NVIDIA/tensorflow-determinism) library. It was tested with keras-tcn by @lingdoc.



#### Implementation results

```
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0949 - accuracy: 0.9706 - val_loss: 0.0763 - val_accuracy: 0.9756
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0831 - accuracy: 0.9743 - val_loss: 0.0656 - val_accuracy: 0.9807
[...]
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0486 - accuracy: 0.9840 - val_loss: 0.0572 - val_accuracy: 0.9832
1875/1875 [==============================] - 46s 25ms/step - loss: 0.0453 - accuracy: 0.9858 - val_loss: 0.0424 - val_accuracy: 0.9862
```



