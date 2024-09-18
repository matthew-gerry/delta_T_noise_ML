'''
shot_noise_NN.py

Train a neural network to predict the temperature difference based on conductance, shot noise, and average temperature data from delta t shot noise experiments.

Adapted from joanna li's code on the same project.

Matthew Gerry, August 2024
'''

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import models as km
from keras import layers as kl


### CONSTANTS ###

g0 = 7.748092*10**(-5) # C^2/J/s, conductance quantum
kB = 1.380649*10**(-23) # J/K, Boltzmann constant


### FUNCTIONS ###

def build_model_fNN(num_nodes, d_rate=0.3, input_dim=2):
    ''' BUILD A MODEL OF A FEEDFORWARD NEURAL NETWORK. FIRST ARGUMENT IS A LIST SPECIFYING THE NUMBER OF NODES IN EACH LAYER--THIS FUNCTION WILL INFER THE NUMBER OF HIDDEN LAYERS TO BE THE LENGTH OF THIS LIST. '''

    model = km.Sequential() # Initialize model
    model.add(kl.InputLayer(shape=(input_dim,), activation="relu")) # Add input layer

    model.add(kl.Normalization(axis=None, mean=None, variance=None))


    for num in num_nodes: # Add a dense layer with dropout for each number of nodes specified
        model.add(kl.Dense(num, activation="tanh"))
        model.add(kl.Dropout(d_rate))

    model.add(kl.Dense(1)) # Add the output layer (no dropout)

    return model


def scheduler(epoch, lr):
    ''' ADAPTIVELY ADJUST THE LEARNING RATE DURING TRAINING (OPTIONAL) '''
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.02))


### MAIN CALLS ###

save_predictions = True

# Read delta T shot noise data
df_train = pd.read_csv('../synthetic_data_deltaT_shot_noise.csv')
df_test = pd.read_csv('../GNoiseData_complete.csv')
df_test = df_test[df_test['DeltaT']>0.5] # For now, drop the experimental data points with deltaT close to 0 - this case is not handled in the synthetic training data

# Rescale data (G already scaled by G0)
df_train['DeltaT'] = df_train['DeltaT']/df_train['T']
df_train['S'] = df_train['S']/(g0 * kB * df_train['T'])

df_test['DeltaT'] = df_test['DeltaT']/df_test['T']
df_test['S'] = df_test['S']/(g0 * kB * df_test['T'])

# Train/test split (80 % of data in training set) - commented out when we use split by synthetic/experimental instead
# df_train = df.sample(frac=0.9, random_state=2)
# df_test = df.drop(df_train.index)

# Set up numpy arrays for use with Keras network
X_train = df_train.drop(['DeltaT','T'], axis=1).to_numpy()
y_train = df_train['DeltaT'].to_numpy()
T_train = df_train['T'].to_numpy() # Save average temperature values in a separate array

X_test = df_test.drop(['DeltaT','T'], axis=1).to_numpy()
y_test= df_test['DeltaT'].to_numpy()
T_test = df_test['T'].to_numpy() # Save average temperature values in a separate array

# Build the model
model = build_model_fNN([6])
print(model.summary())

# loss function
loss_fn = tf.keras.losses.MeanAbsoluteError(reduction="sum_over_batch_size")
# loss_fn(Y[:1], prediction).numpy()

# compile model
callback = keras.callbacks.LearningRateScheduler(scheduler) # Define callback to include schedule function to modulate learning rate, if desired
model.compile(optimizer = keras.optimizers.Adam(),
              loss=loss_fn,
              metrics=['MeanSquaredError','RootMeanSquaredError'])
model.fit(X_train, y_train, epochs=50, verbose=2, callbacks=[])

# Get delta T predictions from the model, multiply by T to undo scaling
y_predicted_train = model.predict(X_train)[:,0]
deltaT_predicted_train = y_predicted_train*T_train


# On the other subplot, plot the true and predicted values based on the test data
# Similarly pass the testing data features through the model and remove the scaling, recover the true values
y_predicted_test = model.predict(X_test)[:,0]
deltaT_predicted_test = y_predicted_test*T_test


# Undo scaling of Delta T by T
df_train['DeltaT'] = df_train['DeltaT']*df_train['T']
df_test['DeltaT'] = df_test['DeltaT']*df_test['T']

# Similarly undo scaling of S
df_train['S'] = df_train['S'] * (g0 * kB * df_train['T'])
df_test['S'] = df_test['S'] * (g0 * kB * df_test['T'])

# Save predicted Delta T values to the DataFrame
df_train['DeltaT_pred'] = deltaT_predicted_train
df_test['DeltaT_pred'] = deltaT_predicted_test

# Write to csv
df_train.to_csv("../training_data_with_prediction_example.csv")
df_test.to_csv("../testing_data_with_prediction_example.csv")