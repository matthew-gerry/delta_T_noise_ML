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

### FUNCTIONS ###

def half_max_unique_val(unique_val, true_vals, predicted_vals):
    ''' CALCULATE THE FWHM OF PREDICTED VALUES CORRESPONDING TO A GIVEN UNIQUE TRUE VALUE IN THE DATA. OUTPUT THE VALUES CORRESPONDING TO BOTH HALF MAXES AS WELL AS THE PEAK '''

    indices = np.where(true_vals==unique_val) # Indices assocaited with instances of a particular true value
    predictions_at_val = predicted_vals[indices] # All predictions made corresponding to that true value
    counts, bins = np.histogram(predictions_at_val, bins=100) # Bin the predictions to approximate peak and width
    max_ind = np.argmax(counts)
    max = counts[max_ind] # Maximum number of counts in a bin
    pred_max = (bins[max_ind] + bins[max_ind+1])/2 # Central value of the bin representing the peak
    pred_mean = np.mean(predictions_at_val)

    half_max = max/2 # Number of counts associated with half max
    # Two lists of positive values with minima at the lower and upper half-max Delta T values, respectively
    lower_half = np.absolute(counts[0:max_ind]-half_max)
    upper_half = np.absolute(counts[max_ind:-1]-half_max)

    # Distance of each half max point to the peak
    neg_err = pred_max - (bins[np.argmin(lower_half)] + bins[np.argmin(lower_half) + 1])/2
    pos_err = (bins[max_ind + np.argmin(upper_half)] + bins[max_ind + np.argmin(upper_half) + 1])/2-pred_max

    return pred_max, neg_err, pos_err, pred_mean


### MAIN CALLS ###

# Load training and testing data from csv files that include also the predicted delta T values made when the model is applied
df_train = pd.read_csv('../training_data_with_prediction_example.csv')
df_test = pd.read_csv('../testing_data_with_prediction_example.csv')


# Plot the predicted vs true values from the training set
plt.figure()
plt.subplot(1,2,1)
plt.scatter(df_train['DeltaT'], df_train['DeltaT_pred'], alpha=0.15) # undo scaling
plt.xlabel('True ∆T')
plt.ylabel('Predicted ∆T')
plt.title('Training set')
plt.plot([0,30], [0,30])
plt.text(5,-0.03,'MAE = '+str(round(np.mean(np.abs(df_train['DeltaT_pred'] - df_train['DeltaT'])),2)))

plt.subplot(1,2,2)
plt.scatter(df_test['DeltaT'], df_test['DeltaT_pred'], alpha=0.25) # Undo scaling
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
            pred_max, neg_err, pos_err, pred_mean = half_max_unique_val(unique_dT,  deltaT_train, deltaT_predicted_train) 

            # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
            plt.errorbar(unique_dT, pred_max, yerr=np.array([[neg_err, pos_err]]).T, capsize=3, fmt="r--o", ecolor = "black")
        except:
            True


    # plot means and error bars for the test data

    plt.subplot(1,2,2)
    for unique_dT in unique_dT_vals_test:
        try:
        # Identify peak value and values at both half maxes of delta T based on the test data
            pred_max, neg_err, pos_err, pred_mean = half_max_unique_val(unique_dT,  deltaT_test, deltaT_predicted_test) 

            # Plot a point with error bars representing the FWHM amongst predicted value at each true value of delta T
            plt.errorbar(unique_dT, pred_max, yerr=np.array([[neg_err, pos_err]]).T, capsize=3, fmt="r--o", ecolor = "black")
        except:
            True

# plt.show()


# Now plot histograms of the predicted values at a selection of true values, for both training and test sets

# Choose a set of indices at which to select out unique delta T values (training data)
dT_indices_train = [0, 6, 12, 18, 24, 29]

fig2 = plt.figure(figsize=(13, 7))
axs = fig2.subplots(2,3)
for i in range(len(dT_indices_train)):
    dT = unique_dT_vals_train[dT_indices_train[i]]

    prediction_mean = df_train.loc[df_train['DeltaT']==dT, 'DeltaT_pred'].mean()

    hg = axs[int(i>2),i%3].hist(df_train.loc[df_train['DeltaT']==dT, 'DeltaT_pred'], bins=50, density=True)
    axs[int(i>2),i%3].vlines([dT, prediction_mean], 0, 1.1*max(hg[0]), color=['red', 'green'])
    axs[int(i>2),i%3].set_ylim([0, 1.1*max(hg[0])])
    axs[int(i>2),i%3].set_title("$\Delta T = "+str(round(dT, 1))+r", \langle \Delta T_{pred}\rangle = $"+str(round(prediction_mean, 1)))

fig2.suptitle("Synthetic data - performance on training set")


# Now for the test data
dT_indices_test = [1, 2, 4, 6, 8, 9]

fig3 = plt.figure(figsize=(13, 7))
axs = fig3.subplots(2,3)
for i in range(len(dT_indices_test)):
    dT = unique_dT_vals_test[dT_indices_test[i]]

    prediction_mean = df_test.loc[df_test['DeltaT']==dT, 'DeltaT_pred'].mean()

    hg = axs[int(i>2),i%3].hist(df_test.loc[df_test['DeltaT']==dT, 'DeltaT_pred'], bins=50, density=True)
    axs[int(i>2),i%3].vlines([dT, prediction_mean], 0, 1.1*max(hg[0]), color=['red', 'green'])
    axs[int(i>2),i%3].set_ylim([0, 1.1*max(hg[0])])
    axs[int(i>2),i%3].set_title("$\Delta T = "+str(round(dT, 2))+r", \langle \Delta T_{pred}\rangle = $"+str(round(prediction_mean, 1)))


fig3.suptitle("Experimental data - performance on testing set")
# plt.show()



plt.show()