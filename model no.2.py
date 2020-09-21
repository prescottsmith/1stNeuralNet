#import required packages for project
import tensorflow as tf
import numpy as np

#import tensorflow specifics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import SGD

#Need to add this for some reason so download of dataset will work
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#download datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

#normalize data
x_train_norm = (x_train - x_train.mean())/x_train.std()
x_test_norm = (x_test - x_train.mean())/x_train.std()
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)


#Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()


input_shape = [32, 32, 3]

def create_model(my_learning_rate):
    """Create and compile a deep neural net."""

    # Create Sequential model
    model = tf.keras.models.Sequential()

    # Attempting to flatten 3D array
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=3072, activation='relu'))

    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Dense(units=1550, activation='relu'))

    model.add(tf.keras.layers.Dense(units=750, activation='relu'))

    #
    # Don't change this layer.
    model.add(tf.keras.layers.Dense(units=100, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 50
batch_size = 2000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_norm, y_train,
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)