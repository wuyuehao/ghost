
import os
import sys


os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

# Imports
import glob
import numpy as np
import os.path as path
from scipy import misc
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from keras import optimizers
import mlflow
import mlflow.keras
import mlflow.sklearn

image_path = os.environ['AUTOMLIB_DATA_PATH']

def setImagePath(path):
    print("setting path :" + path)
    global image_path
    image_path = path


def preprocess(image_format='*.png', train_test_split=0.9):

    file_paths = glob.glob(path.join(image_path, image_format))
    # Load the images
    images = [misc.imread(path) for path in file_paths]
    images = np.asarray(images)
    # Get image size
    image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
    print(image_size)
    # Scale
    images = images / 255

    # Read the labels from the filenames
    n_images = images.shape[0]
    labels = np.zeros(n_images)
    for i in range(n_images):
        filename = path.basename(file_paths[i])[0]
        labels[i] = int(filename[0])

    # Split into test and training sets
    TRAIN_TEST_SPLIT = train_test_split

    # Split at the given index
    split_index = int(TRAIN_TEST_SPLIT * n_images)
    shuffled_indices = np.random.permutation(n_images)
    train_indices = shuffled_indices[0:split_index]
    test_indices = shuffled_indices[split_index:]

    # Split the images and the labels
    x_train = images[train_indices, :, :]
    y_train = labels[train_indices]
    x_test = images[test_indices, :, :]
    y_test = labels[test_indices]
    return x_train, y_train, x_test, y_test,image_size


def visualize_data(positive_images, negative_images):
    # INPUTS
    # positive_images - Images where the label = 1 (True)
    # negative_images - Images where the label = 0 (False)

    figure = plt.figure()
    count = 0
    for i in range(positive_images.shape[0]):
        count += 1
        figure.add_subplot(2, positive_images.shape[0], count)
        plt.imshow(positive_images[i, :, :])
        plt.axis('off')
        plt.title("1")

        figure.add_subplot(1, negative_images.shape[0], count)
        plt.imshow(negative_images[i, :, :])
        plt.axis('off')
        plt.title("0")
    plt.show()


def cnn(size, n_layers, learning_rate):

    print(size)
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)
    print(nuerons)
    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, int(n_layers)):
        print(i)
        print(nuerons[i])
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape, data_format='channels_last'))
        else:
            print(nuerons[i])
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.adam(lr=learning_rate),
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model


def train(hyperparameters, image_format='*.png', train_test_split=0.9, epochs=5,batch_size=200):

    print(hyperparameters)
    print(image_path)

    x_train,y_train,x_test,y_test,image_size = preprocess(image_format, train_test_split)
    # Hyperparamater
    N_LAYERS = hyperparameters.get("num_layers")
    LEARNING_RATE = hyperparameters.get("learning_rate")

    # Instantiate the model
    model = cnn(size=image_size, n_layers=N_LAYERS, learning_rate=LEARNING_RATE)

    # Training hyperparamters
    EPOCHS = epochs
    BATCH_SIZE = batch_size


    # Early stopping callback
    PATIENCE = 10
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')


    # TensorBoard callback
    LOG_DIRECTORY_ROOT = '.'
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

    # Place the callbacks in a list
    callbacks = [early_stopping, tensorboard]


    with mlflow.start_run():

        hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

        mlflow.log_param("hidden_layers", N_LAYERS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        for val in hist.history['acc']:
            mlflow.log_metric("accuracy", val)
        for val in hist.history['loss']:
            mlflow.log_metric("loss", val)
        return val
        #mlflow.log_metric("accuracy", hist.history['acc'][-1])
        #mlflow.log_metric("loss", hist.history['loss'][-1])

        #mlflow.keras.log_model(model, "./models")
        #model.save('./models/mnist_model.h5')
        #mlflow.log_artifacts(log_dir)

        #mlflow.sklearn.log_model(model, "cnn")
        # Train the model




if __name__ == "__main__":
    train(num_layers = ys.argv[1])
