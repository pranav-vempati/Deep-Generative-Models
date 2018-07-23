import tensorflow as tf
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np


def sampling(args):
	z_mean, z_log_var = args
	batch = K.int_shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	epsilon = K.random_normal(shape = (batch, dim))
	reparametrized_layer_output_tensor = z_mean + K.exp(z_log_var)*epsilon # Element wise multiplication with isotropic Gaussian summed with point estimate of the latent code's mean
	return reparametrized_layer_output_tensor
 

 # Plots outputted results(sourced verbatim from: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py )
def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (original_dim,)
intermediate_representation_dim = 512
batch_size = 128

latent_bottleneck_dim = 2

epochs = 120

# Encoder network

inputs = Input(shape = input_shape, name = 'encoder_input')
x = Dense(intermediate_representation_dim, activation = 'relu')(laten_inputs)
z_mean = Dense(latent_bottleneck_dim, name = 'z_mean')(x)
z_log_var = Dense(latent_bottleneck_dim, name = 'z_log_var')(x)

z = Lambda(sampling, name = 'z')([z_mean, z_log_var])

# Instantiation of the encoder MLP

encoder = Model(inputs, [z_mean, z_log_var, z], name = 'encoder')
print(encoder.summary())
#decoder MLP

latent_inputs = Input(shape = (latent_bottleneck_dim), name = 'z_sampling')
x = Dense(intermediate_representation_dim, activation = 'relu')(latent_inputs)
outputs = Dense(original_dim, activation = 'sigmoid')(x) #Posterior distribution of input samples given observation of latent code

# Instantiation of the decoder MLP

decoder = Model(latent_inputs, outputs, name = 'decoder')
print(decoder.summary())

#Instantiation of Variational Autoencoder network

outputs = decoder(encoder(inputs)[2]) # Index across batch axis 
vae = Model(inputs, outputs, name = 'vae_mlp')
reconstruction_loss = binary_crossentropy(inputs, outputs) 
reconstruction_loss = reconstruction_loss*original_dim
kld_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) # Kullback-Leibler divergence between isotropic prior and learned latent representation (regularization term)
kld_aggregate_loss = -0.5*K.sum(kld_loss, axis = -1)
vae_loss = K.mean(reconstruction_loss + kld_aggregate_loss)
vae.compile(loss = vae_loss, optimizer = 'rmsprop', metrics = ['accuracy'])

vae.fit(x_train, epochs = 120, batch_size = 64, validation_data = (x_test,y_test))

# Implementation inspired by: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

















