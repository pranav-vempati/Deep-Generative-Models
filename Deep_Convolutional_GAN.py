from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

# Train a DCGAN to reconstruct MNIST digits 

class Deep_Convolutional_GAN():

	def __init__(self):

		self.image_rows = 28 # Idiosyncratic dimensions for MNIST training data 
		self.image_cols = 28
		self.channels = 1
		self.img_shape = (self.image_rows, self.image_cols, self.channels)
		self.latent_dim = 100

		opt = RMSprop(learning_rate =0.004, rho = 0,9, epsilon = None, decay = 0.0) # Rms prop optimizer
		self.discriminator = self.create_discriminator_network()
		self.discriminator.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

		self.generator = self.create_generator_network()

		latent_noise = Input(shape = (100,)) # Generator reconstructs input samples by first sampling from the latent distribution
		generated_image = self.generator(latent_noise) 

		validity = self.discriminator(generated_image)

		# Now we build the adverserial network by combining the Generator and Discriminator networks
		self.combined =  Model(latent_noise, validity)
		self.combined.compile(loss = 'binary_crossentropy', optimizer = opt)


    def create_generator_network(self): # Deconvolutional(or alternatively, transposed convolutional) network - maps Gaussian noise to fully rendered images intended to 
    	model = Sequential()
    	model.add(Dense(128*7*7), activation = 'relu', input_dim = self.latent_dim)
    	model.add(Reshape(7.7.128)) # Values configured for MNIST digits
    	model.add(UpSampling2D())
    	model.add(Conv2D(128, kernel_size = 3, padding = "same", activation = 'relu'))
    	model.add(BatchNormalization(momentum = 0.6))
    	model.add(UpSampling2D())
    	model.add(Conv2D(64, kernel_size = 3, padding = "same", activation = 'relu'))
    	model.add(BatchNormalization(momentum = 0.6))
    	model.add(Conv2D(self.channels,kernel_size = 3, padding = "same"))
    	model.add(Activation('tanh'))

    	latent_noise = Input(shape = (self.latent_dim,))
    	generated_image = model(latent_noise)

    	return Model(latent_noise, generated_image)


    def create_discriminator_network(self): # Convolutional neural network - maps input images to a binary prediction
    	model = Sequential()

    	model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = self.img_shape, padding = "same"))
    	model.add(LeakyReLU(alpha = 0.2))
    	model.add(Dropout(0.4))
    	model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same"))
    	model.add(LeakyReLU(alpha = 0.2))
    	model.add(BatchNormalization(momentum = 0.6))
    	model.add(Dropout(0.4))
    	model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = "same"))
    	model.add((LeakyReLU(alpha = 0.2))) 
    	model.add(BatchNormalization(momentum = 0.6))
    	model.add(Conv2D(256, kernel_size = 3, strides = 2, padding = "same"))
    	model.add(BatchNormalization(momentum = 0.6))
    	model.add(LeakyReLU(alpha = 0.2))
    	model.add(Dropout(0.4))
    	model.add(Flatten())
    	model.add(Dense(1, activation = "sigmoid"))

    	generated_image = Input(shape = self.img_shape)
    	validity = model(generated_image)
    	return Model(generated_image, validity)



    def train_networks(self, epochs, batch_size, save_model):

    	(X_train,__), (_,_) = mnist.load_data()

    	#After this rescaling, X_train's range will be limited to [-1,1]

    	X_train = X_train/127.5 - 1
    	X_train = np.expand_dims(X_train, axis = 3)

    	training_samples = np.ones((batch_size,1))
    	generator_output = np.zeros((batch_size, 1))

    	for epoch in range(epochs):
    		index = np.random.randint(0, X_train.shape[0], batch_size) # Select half of the training examples to be fed to the discriminator - the other half will be sourced from the Generator's output
    		images = X_train[index]
             # Sample from Gaussian noise and subsequently feed samples into generator
    		gaussian_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

    		generator_images = self.generator.predict(gaussian_noise)

    		# Train discriminator
    		discriminator_loss_training_examples = self.discriminator.train_on_batch(images, training_samples)
    		discriminator_loss_generator_output = self.discriminator.train_on_batch(generator_images, generator_output)
    		discriminator_aggregate_loss = 0.5*np.add(discriminator_loss_training_examples, discriminator_loss_generator_output)

    		# Train generator
    		generator_loss = self.combined.train_on_batch(gaussian_noise, training_samples)

    		print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

 # The following is sourced verbatim from https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

  def save_imgs(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


if ___name____ == 'main': # If script is explicitly executed
	deep_conv_gan = Deep_Convolutional_GAN()
	deep_conv_gan.train(epochs = 7000, batch_size = 64, save_model = 100)


# Implementation inspired by: https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py













    	

















