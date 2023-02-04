
```bash
pip install keras-tcn
pip install keras-tcn --no-dependencies  # without the dependencies if you already have TF/Numpy.
```

For MacOS M1 users: `pip install --no-binary keras-tcn keras-tcn`. The `--no-binary` option will force pip to download the sources (tar.gz) and re-compile them locally. Also make sure that `grpcio` and `h5py` are installed correctly. There are some tutorials on how to do that online.

##  Add attention to TCN (Temporal Convolutional Network) 

- Exhibit longer memory than recurrent architectures with the same capacity.
- Performs better than original TCN on long time series 
- Parallelism (convolutional layers), flexible receptive field size (how far the model can see), stable gradients (compared to backpropagation through time, vanishing gradients)are performed better





### Input shape

3D tensor with shape `(batch_size, timesteps, input_dim)`.


### Output shape

- `return_sequences=True`: 3D tensor with shape `(batch_size, timesteps, nb_filters)`.





## Run

Reproducible results are possible on (NVIDIA) GPUs using the [tensorflow-determinism](https://github.com/NVIDIA/tensorflow-determinism) library. It was tested with keras-tcn







