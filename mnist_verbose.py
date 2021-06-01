print('Importing Libraries. (All of the goodies needed.)')
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
input('Press any key to continue.\n')

# The Simplified Big Picture of How Most Supervised Networks Work
# The Just of It
# [1] Draw a batch of training samples x and corresponding targets y.
# [2] Reshape and format the data, as needed.
# [2] Run the network on x (a step called the forward pass) to obtain predictions y_pred.
# [3] Compute the loss of the network on the batch, a measure of the mismatch
# between y_pred and y.
# [4] Update all weights of the network in a way that slightly reduces the loss on this
# batch.


# [1] Draw a batch of training samples x and corresponding targets y.
# Load Training Data
print('Loading Training Data. (MNIST Pictures)')
# mnist data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# ( ((60000, 28, 28), (60000)), ((10000, 28, 28), (10000)) )

# training data
print('Amount of training images:', len(train_images))                          #60000
print('Training images shape:', train_images.shape)                             #60000, 28, 28
print('Training image type:', train_images.dtype)                               #uint8
print('Amount of training labels:', len(train_labels))                          #60000
print('Training label shape:', train_labels.shape)                              #60000
print('Training label type:', train_labels.dtype)                               #uint8
print('Number:', train_labels[0])
print(train_images[0])
input('Press any key to continue.\n')

# testing data
print('Amount of testing images:', len(test_images))                            #10000
print('Testing images shape:', test_images.shape)                               #10000, 28, 28
print('Training image type:', test_images.dtype)                                #uint8
print('Amount of testing labels:', len(test_labels))                            #10000
print('Testing images shape:', test_labels.shape)                               #10000
print('Training image type:', test_labels.dtype)                                #uint8
print('Number:', test_labels[0])
print(test_images[0])
input('Press any key to continue.\n')


# [2] Reshape and format the data, as needed.
print('#Reshaping Data.')
# reshape training data
print('Reshaping Image Training Data. (Going from (60000, 28, 28) -> (60000, 784))')
print('Training images shape before:', train_images.shape)                      #60000, 28, 28
print('Number:', train_labels[0])
print(train_images[0])
input('Press any key to continue.\n')
train_images = train_images.reshape((60000, 28 * 28))
print('Training images shape after:', train_images.shape)                       #60000, 784
print('Number:', train_labels[0])
print(train_images[0])
input('Press any key to continue.\n')
# reshape testing data
print('Reshaping Image Testing Data. (Going from (10000, 28, 28) -> (10000, 784))')
print('Testing images shape before:', test_images.shape)                        #10000, 28, 28
print('Number:', test_labels[0])
print(test_images[0])
test_images = test_images.reshape((10000, 28 * 28))
print('Testing images shape after:', test_images.shape)                         #10000, 784
print('Number:', test_labels[0])
print(test_images[0])
input('Press any key to continue.\n')

print('#Formatting Data.')
# convert training data to float, the 255 divides out gray scale values betwen 0 or 1 per pixel.
print('Converting Image Training Data. uint8 -> float32')
print('Training image type before:', train_images.dtype)                        #uint8
print('Number:', train_labels[0])
print(train_images[0])
input('Press any key to continue.\n')
train_images = train_images.astype('float32') / 255
print('Training image type after:', train_images.dtype)                         #float32
print('Number:', train_labels[0])
print(train_images[0])
input('Press any key to continue.\n')

# convert testing data to float, the 255 divides out gray scale values betwen 0 or 1 per pixel.
print('Converting Image Testing Data. uint8 -> float32')
print('Testing image type before:', test_images.dtype)                          #uint8
print('Number:', test_labels[0])
print(test_images[0])
input('Press any key to continue.\n')
test_images = test_images.astype('float32') / 255
print('Testing image type after:', test_images.dtype)                           #float32
print('Number:', test_labels[0])
print(test_images[0])
input('Press any key to continue.\n')

# convert training labels from int to array, with indexes of 0's or 1's denoting the int value number.
print('#Running to categorical. (Converting int label numbers to, array vector indexes.)')
print('Training labels type before:', train_labels.dtype)                       #uint8
print('Training labels shape before:', train_labels.shape)                      #60000,
print([train_labels[i] for i in range(10)])
input('Press any key to continue.\n')
train_labels = to_categorical(train_labels)
print('Training labels type after:', train_labels.dtype)                        #float32
print('Training labels shape after:', train_labels.shape)                       #60000, 10
print([train_labels[i] for i in range(10)])
input('Press any key to continue.\n')

# convert testing labels from int to array, with indexes of 0's or 1's denoting the int value number.
print('Testing labels type before:', test_labels.dtype)                         #uint8
print('Testing labels shape before:', test_labels.shape)                        #10000,
print([test_labels[i] for i in range(10)])
input('Press any key to continue.\n')
test_labels = to_categorical(test_labels)
print('Testing labels type after:', test_labels.dtype)                          #float32
print('Testing labels shape after:', test_labels.shape)                         #10000, 10
print([test_labels[i] for i in range(10)])
input('Press any key to continue.\n')

# # Format data into 10k samples by 28*28 = 784, 2d tensor - (60k, 768)
# test_images = test_images.reshape((10000, 28 * 28))
# # Convert to float, the 255 divides out gray scale values betwen 0 or 1 per pixel.
# test_images = test_images.astype('float32') / 255

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
# network.compile(optimizer='rmsprop',
# loss='categorical_crossentropy',
# metrics=['accuracy'])


# [2] Run the network on x (a step called the forward pass) to obtain predictions y_pred.
# [3] Compute the loss of the network on the batch, a measure of the mismatch
# between y_pred and y.
# [4] Update all weights of the network in a way that slightly reduces the loss on this
# batch.
# Running the network, optimizing, and updating weights.
#You have 60k samples, you take 128 samples from them, which equates to 469 runs per epoch,
# and you do this for 5 epochs, so in total you do roughly 2,345 gradient descent optimization runs.
# network.fit(train_images, train_labels, epochs=5, batch_size=128)

# [Saving] a network after training is done this wayself.
# #https://www.tensorflow.org/guide/keras/save_and_serialize
# network.save('untrained_mnist.h5')

# This checks to see how well the model performs, against testing data.
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc: ', test_acc)
# print('test_loss:', test_loss)



#This is for visualization of model graphs.
#https://www.tensorflow.org/tensorboard/graphs
