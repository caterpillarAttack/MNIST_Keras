from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# The Simplified Big Picture of How Most Supervised Networks Work
# The Just of It
# [1] Draw a batch of training samples x and corresponding targets y.
# [2] Run the network on x (a step called the forward pass) to obtain predictions y_pred.
# [3] Compute the loss of the network on the batch, a measure of the mismatch
# between y_pred and y.
# [4] Update all weights of the network in a way that slightly reduces the loss on this
# batch.


# [1] Draw a batch of training samples x and corresponding targets y.
# Load Training Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Format data into 60k samples by 28*28 = 784, 2d tensor - (60k, 768)
train_images = train_images.reshape((60000, 28 * 28))
# Convert to float, the 255 divides out gray scale values betwen 0 or 1 per pixel.
train_images = train_images.astype('float32') / 255

# Im assuming this is some kind of categorization bs for data type comparison.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Format data into 10k samples by 28*28 = 784, 2d tensor - (60k, 768)
test_images = test_images.reshape((10000, 28 * 28))
# Convert to float, the 255 divides out gray scale values betwen 0 or 1 per pixel.
test_images = test_images.astype('float32') / 255


# Defining/Loading the Network
# [Defining] The type of network.
network = models.Sequential()
# Takes the input information, and performs basic tensor ops, and relu activation function.
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# Takes the output of the previous operation, and shunts it into the 10 possible states, ie the numbers.
network.add(layers.Dense(10, activation='softmax'))

# [Loading] a network
# network = models.load_model('trained_mnist.h5')
# network = models.load_model('untrained_mnist.h5')

# Compiling options, though really still part of defining as youre telling the compiler what optimization scheme and loss function to use etc.
# The rms prop is the optimizer that minimizes loss, and categorical crossentropy is the loss function that is differentiated.
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# [2] Run the network on x (a step called the forward pass) to obtain predictions y_pred.
# [3] Compute the loss of the network on the batch, a measure of the mismatch
# between y_pred and y.
# [4] Update all weights of the network in a way that slightly reduces the loss on this
# batch.
# Running the network, optimizing, and updating weights.
#You have 60k samples, you take 128 samples from them, which equates to 469 runs per epoch,
# and you do this for 5 epochs, so in total you do roughly 2,345 gradient descent optimization runs.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# [Saving] a network after training is done this wayself.
# #https://www.tensorflow.org/guide/keras/save_and_serialize
# network.save('untrained_mnist.h5')

# This checks to see how well the model performs, against testing data.
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
print('test_loss:', test_loss)



#This is for visualization of model graphs.
#https://www.tensorflow.org/tensorboard/graphs
