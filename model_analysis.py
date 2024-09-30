'''
training_results.py

Analysis of the results from training a neural netowrk to predict Delta T values based on data from Delta T shot noise experiments. Network trained in the script shot_noise_NN.py

Matthew Gerry, September 2024
'''

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras import saving


### CONSTANTS ###

G0 = 7.748092*10**(-5) # C^2/J/s, conductance quantum
kB = 1.380649*10**(-23) # J/K, Boltzmann constant


### FUNCTIONS ###

def stats_unique_val(unique_val, true_vals, predicted_vals):
    ''' CALCULATE THE FWHM OF PREDICTED VALUES CORRESPONDING TO A GIVEN UNIQUE TRUE VALUE IN THE DATA. OUTPUT THE VALUES CORRESPONDING TO BOTH HALF MAXES AS WELL AS THE PEAK '''

    indices = np.where(true_vals==unique_val) # Indices assocaited with instances of a particular true value
    predictions_at_val = predicted_vals[indices] # All predictions made corresponding to that true value
    counts, bins = np.histogram(predictions_at_val, bins=100) # Bin the predictions to approximate peak and width
    max_ind = np.argmax(counts)
    max = counts[max_ind] # Maximum number of counts in a bin
    pred_max = (bins[max_ind] + bins[max_ind+1])/2 # Central value of the bin representing the peak
    pred_mean = np.mean(predictions_at_val)
    pred_stddev = np.std(predictions_at_val)

    half_max = max/2 # Number of counts associated with half max
    # Two lists of positive values with minima at the lower and upper half-max Delta T values, respectively
    lower_half = np.absolute(counts[0:max_ind]-half_max)
    upper_half = np.absolute(counts[max_ind:-1]-half_max)

    # Distance of each half max point to the peak
    neg_err = pred_max - (bins[np.argmin(lower_half)] + bins[np.argmin(lower_half) + 1])/2
    pos_err = (bins[max_ind + np.argmin(upper_half)] + bins[max_ind + np.argmin(upper_half) + 1])/2-pred_max

    return pred_max, neg_err, pos_err, pred_mean, pred_stddev


def mean_above_threshold(norm_prob, bins, threshold):
    ''' CALCULATE THE MEAN VALUE OF A BINNED DATASET, ONLY INCLUDING BINS WITH NORMALIZED PROBABILITY ABOVE A THRESHOLD VALUE '''

    norm_prob[norm_prob<threshold] = 0 # Eliminate bins with normalized probability below the threshold value
    # Use bin centres as values to plug into mean calculation
    bin_centres = np.array([0.5*(bins[i] + bins[i+1]) for i in range(len(norm_prob))])

    return sum(np.multiply(norm_prob, bin_centres))/sum(norm_prob)

### MAIN CALLS ###

# Identify model to load in based on maximum conductance used in training
max_G = 1.0 # Maximum conductance value used in training
model_name = "../model_" + str(int(max_G)) + "G0_without_T.keras"

save_predictions = False # Toggle saving of new csv that includes the experimental data and predicted values

# # Load training and testing data from csv files that include also the predicted delta T values made when the model is applied
# df_train = pd.read_csv('../training_data_with_prediction_4G0.csv')
# df_test = pd.read_csv('../testing_data_with_prediction_4G0.csv')

# Load testing data as well as training data used for fitting the model
df_train = pd.read_csv('../synthetic_data_deltaT_shot_noise_4G0.csv')
df_test = pd.read_csv('../GNoiseData_complete.csv')
df_test = df_test[df_test['DeltaT'] > 0.5] # For now, drop the experimental data points with deltaT close to 0 - this case is not handled in the synthetic training data
df_test = df_test[df_test['G'] < max_G] # Keep only the points with G values in the range used for training

# Record number of points in each dataset
train_set_size = df_train.shape[0]
test_set_size = df_test.shape[0]

# Rescale data as done in fitting (G already scaled by G0)
df_train['DeltaT'] = df_train['DeltaT']/df_train['T']
df_train['S'] = df_train['S']/(G0 * kB * df_train['T'])

df_test['DeltaT'] = df_test['DeltaT']/df_test['T']
df_test['S'] = df_test['S']/(G0 * kB * df_test['T'])

# Isolate features and target values
X_train = df_train.drop(['DeltaT','S_full', 'T'], axis=1).to_numpy()
y_train = df_train['DeltaT'].to_numpy()
T_train = df_train['T'].to_numpy() # Save average temperature values in a separate array

X_test = df_test.drop(['DeltaT', 'T'], axis=1).to_numpy()
y_test= df_test['DeltaT'].to_numpy()
T_test = df_test['T'].to_numpy() # Save average temperature values in a separate array


# Load the model--ensure it agrees with the training data file imported above
model = saving.load_model(model_name)

# Get delta T predictions from the model on both the training and test set, multiply by T to undo scaling, write to DataFrames
y_predicted_train = model.predict(X_train)[:,0]
df_train['DeltaT_pred'] = y_predicted_train*T_train

y_predicted_test = model.predict(X_test)[:,0]
df_test['DeltaT_pred'] = y_predicted_test*T_test

# Undo scaling of Delta T by T
df_train['DeltaT'] = df_train['DeltaT']*df_train['T']
df_test['DeltaT'] = df_test['DeltaT']*df_test['T']

# Similarly undo scaling of S
df_train['S'] = df_train['S'] * (G0 * kB * df_train['T'])
df_test['S'] = df_test['S'] * (G0 * kB * df_test['T'])


if save_predictions:
    # If desired, save a new CSV that includes the predicted values
    df_train.to_csv("../training_data_with_prediction_4G0.csv")
    df_test.to_csv("../testing_data_with_prediction_4G0.csv")


# Plot the predicted vs true values from the training set
plt.figure()
plt.subplot(1,2,1)
plt.scatter(df_train['DeltaT'], df_train['DeltaT_pred'], color='red', alpha=0.1) # undo scaling
plt.xlabel('True ∆T')
plt.ylabel('Predicted ∆T')
plt.title('Training set')
plt.plot([0,30], [0,30])
plt.text(5,-0.03,'MAE = '+str(round(np.mean(np.abs(df_train['DeltaT_pred'] - df_train['DeltaT'])),2)))

plt.subplot(1,2,2)
plt.scatter(df_test['DeltaT'], df_test['DeltaT_pred'], color='red', alpha=0.1) # Undo scaling
plt.xlabel('True ∆T')
#plt.ylabel('Predicted ∆T')
plt.title('Testing set')
plt.plot([0,30], [0,30])
plt.text(5,-0.03,'MAE = '+str(round(np.mean(np.abs(df_test['DeltaT_pred'] - df_test['DeltaT'])),2)))


# Identify unique delta T values in the training and test sets
unique_dT_vals_train = np.unique(df_train['DeltaT']) 
unique_dT_vals_test = np.unique(df_test['DeltaT']) # Identify unique values

plot_errorbars = True
if plot_errorbars:
    # Plot means and error bars based on binning of the predicted values at each true value and calculating the FWHM (training data)

    # Convert the relevant values to numpy arrays for operations used to calculate error bars
    deltaT_train = df_train['DeltaT'].to_numpy()
    deltaT_predicted_train = df_train['DeltaT_pred'].to_numpy()
    deltaT_test = df_test['DeltaT'].to_numpy()
    deltaT_predicted_test = df_test['DeltaT_pred'].to_numpy()

    # Apply maxima and errorbars to plot
    plt.subplot(1,2,1)
    for unique_dT in unique_dT_vals_train:
        try:
            # Identify peak value and values at both half maxes of delta T
            pred_max, neg_err, pos_err, pred_mean, pred_stddev = stats_unique_val(unique_dT,  deltaT_train, deltaT_predicted_train) 

            # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
            plt.errorbar(unique_dT, pred_mean, yerr=np.array([[pred_stddev, pred_stddev]]).T, capsize=3, fmt="g--o", ecolor = "black", mec='black')
        except:
            True


    # plot means and error bars for the test data

    plt.subplot(1,2,2)
    for unique_dT in unique_dT_vals_test:
        try:
        # Identify peak value and values at both half maxes of delta T based on the test data
            pred_max, neg_err, pos_err, pred_mean, pred_stddev = stats_unique_val(unique_dT,  deltaT_test, deltaT_predicted_test) 

            # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
            plt.errorbar(unique_dT, pred_mean, yerr=np.array([[pred_stddev, pred_stddev]]).T, capsize=3, fmt="g--o", ecolor="black", mec='black')
        except:
            True


# Now plot histograms of the predicted values at a selection of true values, for both training and test sets
# Choose a set of indices at which to select out unique delta T values (training data)
dT_indices_train = [0, 4, 8, 12, 15, 19]

# Minimum probably to be included in secondary mean calculation
threshold = 0.02

fig2 = plt.figure(figsize=(13, 7))
axs = fig2.subplots(2,3)
for i in range(len(dT_indices_train)):
    dT = unique_dT_vals_train[dT_indices_train[i]]

    # Organize data into bins and counts, calculate normalized probabilities
    counts, bins = np.histogram(df_train.loc[df_train['DeltaT']==dT, 'DeltaT_pred'], bins=50, density=False)
    norm_prob = counts/sum(counts)
    axs[int(i>2), i%3].stairs(norm_prob, bins, fill=True)
    
    # Calculate the mean of predictions at a given true delta T, as well as the mean excluding bins that do not meet a certain threshold
    prediction_mean = df_train.loc[df_train['DeltaT']==dT, 'DeltaT_pred'].mean()
    prediction_mean_threshold = mean_above_threshold(norm_prob, bins, threshold)

    axs[int(i>2), i%3].vlines([dT, prediction_mean, prediction_mean_threshold], 0, 1.1*max(norm_prob), color=['red', 'green', 'magenta'])
    axs[int(i>2), i%3].hlines([threshold], min(bins[0], 0.98*dT), max(bins[-1], 1.02*dT), color='black')
    axs[int(i>2), i%3].set_ylim([0, 1.1*max(norm_prob)])    
    axs[int(i>2), i%3].set_xlim([min(bins[0], 0.98*dT), max(bins[-1], 1.02*dT)])    
    axs[int(i>2), i%3].set_title("$\Delta T = "+str(round(dT, 1))+r", \langle \Delta T_{pred}\rangle = $"+str(round(prediction_mean, 1)))

    if i>2:
        axs[int(i>2), i%3].set_xlabel("Predicted $\Delta T$")
    if i%3==0:
        axs[int(i>2), i%3].set_ylabel("Normalized Probability")


fig2.suptitle("Synthetic data - performance on training set ("+str(train_set_size)+" points)")


# Now for the test data
dT_indices_test = [1, 2, 4, 6, 8, 9]

fig3 = plt.figure(figsize=(13, 7))
axs = fig3.subplots(2,3)
for i in range(len(dT_indices_test)):
    dT = unique_dT_vals_test[dT_indices_test[i]]


    # Organize data into bins and counts, calculate normalized probabilities
    counts, bins = np.histogram(df_test.loc[df_test['DeltaT']==dT, 'DeltaT_pred'], bins=50, density=False)
    norm_prob = counts/sum(counts)
    axs[int(i>2), i%3].stairs(norm_prob, bins, fill=True)

    prediction_mean = df_test.loc[df_test['DeltaT']==dT, 'DeltaT_pred'].mean()
    prediction_mean_threshold = mean_above_threshold(norm_prob, bins, threshold)

    axs[int(i>2), i%3].vlines([dT, prediction_mean, prediction_mean_threshold], 0, 1.1*max(norm_prob), color=['red', 'green', 'magenta'])
    axs[int(i>2), i%3].hlines([threshold], min(bins[0], 0.98*dT), max(bins[-1], 1.02*dT), color='black')
    axs[int(i>2), i%3].set_ylim([0, 1.1*max(norm_prob)])
    axs[int(i>2), i%3].set_xlim([min(bins[0], 0.98*dT), max(bins[-1], 1.02*dT)])    
    axs[int(i>2), i%3].set_title("$\Delta T = "+str(round(dT, 2))+r", \langle \Delta T_{pred}\rangle = $"+str(round(prediction_mean, 1)))

    if i>2:
        axs[int(i>2), i%3].set_xlabel("Predicted $\Delta T$")
    if i%3==0:
        axs[int(i>2), i%3].set_ylabel("Normalized Probability")

fig3.suptitle("Experimental data - performance on testing set ("+str(test_set_size)+" points)")


plt.show()