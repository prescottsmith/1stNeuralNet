#import required packages for project
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Need to add this for some reason so download of dataset will work
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


#download datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

#normalize and one-hot encode data
x_train_norm = (x_train - x_train.mean())/x_train.std()
x_test_norm = (x_test - x_train.mean())/x_train.std()
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

#define labels for later comparison
fine_label = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]

#import required TF components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1


#Define 'Create Model'
def create_model(my_learning_rate):
    """Create and compile Neural Network model"""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu', activity_regularizer=l1(0.001), kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(Dense(100, activation='softmax'))

    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

#Define 'Train Model'
def train_model(model, train_features, train_labels, epochs,
                batch_size=None, validation_split=0.1):
    """Train Neural Network model by feeding it data"""
    history = model.fit(train_features, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

#Define plot function for training metrics
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



#Set hyperparameters and methods for training
learning_rate = 0.003
epochs = 50
batch_size = 500
validation_split = 0.2
input_shape = (32, 32, 3)
loss_function = categorical_crossentropy
optimizer = Adam(learning_rate =learning_rate)


#Establish the model's topography (Call your model function)
my_model = create_model(learning_rate)

# Train model on the normalized training set.
epochs, hist = train_model(my_model, x_train_norm, y_train_encoded,
                           epochs, batch_size, validation_split)

#Plot accuracy of training
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)
plt.show()

# Evaluate against test set
loss, accuracy = my_model.evaluate(x_test_norm, y_test_encoded)
print('test set accuracy: ', accuracy * 100)



