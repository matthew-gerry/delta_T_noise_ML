'''
shot_noise_NN.py

TRAIN A NEURAL NETWORK TO PREDICT THE TEMPERATURE DIFFERENCE BASED ON CONDUCTANCE, SHOT NOISE, AND AVERAGE TEMPERATURE DATA FROM DELTA T SHOT NOISE EXPERIMENTS.

ADAPTED FROM JOANNA LI'S CODE ON THE SAME PROJECT.

Matthew Gerry, August 2024
'''

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


### CONSTANTS ###

g0 = 7.748092*10**(-5) # C^2/J/s, conductance quantum
kB = 1.380649*10**(-23) # J/K, Boltzmann constant


### FUNCTIONS ###

def build_model_fNN(num_nodes, d_rate=0.4, input_dim=2):
    ''' BUILD A MODEL OF A FEEDFORWARD NEURAL NETWORK. FIRST ARGUMENT IS A LIST SPECIFYING THE NUMBER OF NODES IN EACH LAYER--THIS FUNCTION WILL INFER THE NUMBER OF HIDDEN LAYERS TO BE THE LENGTH OF THIS LIST. '''

    model = km.Sequential() # Initialize model
    model.add(kl.InputLayer(shape=(input_dim,), activation="relu")) # Add input layer

    for num in num_nodes: # Add a dense layer with dropout for each number of nodes specified
        model.add(kl.Dense(num, activation="tanh"))
        model.add(kl.Dropout(d_rate))

    model.add(kl.Dense(1)) # Add the output layer (no dropout)

    return model


def scheduler(epoch, lr):
    ''' ADAPTIVELY ADJUST THE LEARNING RATE DURING TRAINING '''
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.02))


def half_max_unique_val(unique_val, true_vals, predicted_vals):
    ''' CALCULATE THE FWHM OF PREDICTED VALUES CORRESPONDING TO A GIVEN UNIQUE TRUE VALUE IN THE DATA. OUTPUT THE VALUES CORRESPONDING TO BOTH HALF MAXES AS WELL AS THE PEAK '''

    indices = np.where(true_vals==unique_val) # Indices assocaited with instances of a particular true value
    predictions_at_val = predicted_vals[indices] # All predictions made corresponding to that true value
    counts, bins = np.histogram(predictions_at_val, bins=100) # Bin the predictions to approximate peak and width
    max_ind = np.argmax(counts)
    max = counts[max_ind] # Maximum number of counts in a bin
    pred_max = (bins[max_ind] + bins[max_ind+1])/2 # Central value of the bin representing the peak

    half_max = max/2 # Number of counts associated with half max
    # Two lists of positive values with minima at the lower and upper half-max Delta T values, respectively
    lower_half = np.absolute(counts[0:max_ind]-half_max)
    upper_half = np.absolute(counts[max_ind:-1]-half_max)

    # Distance of each half max point to the peak
    neg_err = pred_max - (bins[np.argmin(lower_half)] + bins[np.argmin(lower_half) + 1])/2
    pos_err = (bins[max_ind + np.argmin(upper_half)] + bins[max_ind + np.argmin(upper_half) + 1])/2-pred_max

    return pred_max, neg_err, pos_err


### MAIN CALLS ###

# Read delta T shot noise data
df = pd.read_csv('../GNoiseData_complete.csv')

# Rescale data (G already scaled by G0)
df['DeltaT'] = df['DeltaT']/df['T']
df['S'] = df['S']/(g0 * kB * df['T'])

# Train/test split (80 % of data in training set)
df_train = df.sample(frac=0.9, random_state=2)
df_test = df.drop(df_train.index)

# Set up numpy arrays for use with Keras network
X_train = df_train.drop(['DeltaT','T'], axis=1).to_numpy()
y_train = df_train['DeltaT'].to_numpy()
T_train = df_train['T'].to_numpy() # Save average temperature values in a separate array

X_test = df_test.drop(['DeltaT','T'], axis=1).to_numpy()
y_test= df_test['DeltaT'].to_numpy()
T_test = df_test['T'].to_numpy() # Save average temperature values in a separate array

# Build the model
model = build_model_fNN([10])
# print(model.summary())

# loss function
loss_fn = tf.keras.losses.MeanAbsoluteError(reduction="sum_over_batch_size")
# loss_fn(Y[:1], prediction).numpy()

# compile model
callback = LearningRateScheduler(scheduler) # Define callback to include schedule function to modulate learning rate, if desired
model.compile(optimizer = Adam(),
              loss=loss_fn,
              metrics=['MeanSquaredError','RootMeanSquaredError'])
model.fit(X_train, y_train, epochs=30, verbose = 2, callbacks=[])

# Get delta T predictions from the model, multiply by T to undo scaling
y_predicted_train = model.predict(X_train)[:,0]
deltaT_predicted_train = y_predicted_train*T_train

# Re-calculate also the true Delta T values, unscaled
deltaT_train = y_train*T_train # All true delta T values

# Plot the predicted vs true values from the training set
plt.figure()
plt.subplot(1,2,1)
plt.scatter(deltaT_train, deltaT_predicted_train, alpha=0.15) # undo scaling
plt.xlabel('True ∆T')
plt.ylabel('Predicted ∆T')
plt.title('Training set')
plt.plot([0,30], [0,30])
plt.text(5,-0.03,f'MAE={round(np.mean(np.abs(deltaT_predicted_train - deltaT_train)),2)}')


# On the other subplot, plot the true and predicted values based on the test data
# Similarly pass the testing data features through the model and remove the scaling, recover the true values
y_predicted_test = model.predict(X_test)[:,0]
deltaT_predicted_test = y_predicted_test*T_test
deltaT_test = y_test*T_test

plt.subplot(1,2,2)
plt.scatter(deltaT_test, deltaT_predicted_test, alpha=0.25) # Undo scaling
plt.xlabel('True ∆T')
#plt.ylabel('Predicted ∆T')
plt.title('Testing set')
plt.plot([0,30], [0,30])
plt.text(5,-0.03,f'MAE={round(np.mean(np.abs( deltaT_predicted_test-deltaT_test)),2)}')

try:
    # Plot means and error bars based on binning of the predicted values at each true value and calculating the FWHM (training data)
    unique_dT_vals = np.unique(deltaT_train) # Identify unique values
    plt.subplot(1,2,1)
    for unique_dT in unique_dT_vals:
        # Identify peak value and values at both half maxes of delta T
        pred_max, neg_err, pos_err = half_max_unique_val(unique_dT,  deltaT_train, deltaT_predicted_train) 

        # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
        plt.errorbar(unique_dT, pred_max, yerr=np.array([[neg_err, pos_err]]).T, capsize=3, fmt="r--o", ecolor = "black")
except:
    True

try:
    # plot means and error bars for the test data
    unique_dT_vals = np.unique(deltaT_test)

    plt.subplot(1,2,2)
    for unique_dT in unique_dT_vals:
    # Identify peak value and values at both half maxes of delta T based on the test data
        pred_max, neg_err, pos_err = half_max_unique_val(unique_dT,  deltaT_test, deltaT_predicted_test) 

        # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
        plt.errorbar(unique_dT, pred_max, yerr=np.array([[neg_err, pos_err]]).T, capsize=3, fmt="r--o", ecolor = "black")
except:
    True

plt.show()