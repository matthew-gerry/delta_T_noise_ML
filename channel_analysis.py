'''
channel_analysis.py

Calculate the channel transmissions based on the values of G and S when G<G_0, assuming there are only two channels with nonzero transmission (argued based on the low overall conductance).

Assess whether the manner in which the channels open sequentially varies at all with temperature.

Matthew Gerry, September 2024
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from math import pi, sqrt
from sympy import solve

### CONSTANTS ###

G0 = 7.748092*10**(-5) # C^2/Js, quantum of conductance
kB = 1.380649*10**(-23) # J/K, Boltzmann constant

### FUNCTIONS ###

def channel_transmission(G, S, T, DeltaT):
    ''' BASED ON THE RAW VALUES OF G AND S, CALCULATE THE CHANNEL TRANSMISSIONS ASSUMING TWO CHANNELS '''

    # Isolate the contributions to G and S from the non-dimensional channel transmission coefficients
    g = G # Note that G is already scaled by G0 in the data we have
    s = S/(G0 * kB * (pi**2/9 - 2/3) * DeltaT**2 / T)

    try:
        # The taus are solutions to a quadratic equation
        tau_a = 0.5*g + sqrt(0.5 * g * (1 - 0.5*g) - 0.5*s)
        # tau_b = 0.5*g - sqrt(0.5 * g * (1 - 0.5*g) - 0.5*s) # This line and the one below are equivalent
        tau_b = g - tau_a

        # Ensure both solutions are positive and s is positive, discard nonphysical solutions in which they are not
        if tau_a < 0 or tau_b < 0:
            return None, None
        else:
            # Define the larger value as tau1, smaller as tau2
            tau1 = max(tau_a, tau_b)
            tau2 = min(tau_a, tau_b)
        
            return tau1, tau2
    
    except: # sqrt will throw an error in the case that the solutions are complex
        return None, None
        

### MAIN CALLS ###

df = pd.read_csv('../GNoiseData_complete.csv')
df = df[df['G']<=1] # Keep only data for which G < G_0

# Add columns to the dataframe for the values we will derive
df.insert(4, "tau1", None)
df.insert(5, "tau2", None)
df.insert(6, "x", None)

# Calculate the channel transmissions, iterating through the df row by row
for index, row in df.iterrows():
    tau1, tau2 = channel_transmission(row['G'], row['S'], row['T'], row['DeltaT'])
    
    if tau2 != None:
        x = tau2/row['G']
        df.at[index,'tau1'] = tau1
        df.at[index,'tau2'] = tau2
        df.at[index,'x'] = x

# Drop rows where tau1 and tau2 are not defined
df = df.dropna()

unique_T_vals = np.unique(df['T'].to_numpy())
# In our remaining data there are eight unique T values

# Plot histograms of the different x values at each different T value
fig, axs = plt.subplots(2, int(np.ceil(0.5*len(unique_T_vals))))
for i in range(len(unique_T_vals)):
    T = unique_T_vals[i]
    df_temp = df[df['T']==T] # Select out rows with the corresponding temperature

    x_av = df_temp['x'].mean() # Average x at each T value
    x_av = round(x_av, 3)
    count = df_temp.shape[0] # Number of valid data points at this T value

    # Determine where in the subplot grid each plot will go (this is due to my weird indexing)
    subplot_row = int(i >= 0.5*len(unique_T_vals))
    subplot_col = i % int(np.floor(0.5*len(unique_T_vals)))

    # Plot a histogram of all x values at the given T value
    axs[subplot_row, subplot_col].hist(df_temp['x'], bins=50)
    # Include temperature, average x, and count in the title of each subplot
    axs[subplot_row, subplot_col].set_title("$T = "+str(T)+r", \bar{x} = "+str(x_av)+", cnt = "+str(count)+"$")    
    if subplot_row==1:
        axs[subplot_row, subplot_col].set_xlabel("$x$")

plt.show()