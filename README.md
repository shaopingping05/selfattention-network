
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



