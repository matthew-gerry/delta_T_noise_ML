'''
channel_analysis.py

Calculate the channel transmissions based on the values of G and S when G<G_0, assuming there are only two channels with nonzero transmission (argued based on the low overall conductance).

Also fit the G<G_0 data to a quadratic as an additional way of quantifying the sequential opening.

Assess whether the manner in which the channels open sequentially varies at all with temperature.

Matthew Gerry, September 2024
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from math import pi, sqrt
from scipy.optimize import curve_fit

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
    
def s_model(g, y):
    ''' THE RELATIONSHIP BETWEEN NON-DIMENSIONALIZED s AND g '''
    return g*(1 - y*g)

def s_linear(g, m):
    ''' LEARN THE SLOPE OF THE OVERALL ACCUMULATION OF s WITH INCREASING g '''
    return m*(g-1)

### MAIN CALLS ###

df = pd.read_csv('../GNoiseData_complete.csv')
df = df[df['G']<=1] 

df2 = df[df['G']<=1] # Dataframe for dual-channel analysis, keep only data for which G < G_0

# Add columns to the dataframe for the values we will derive
df2.insert(4, "tau1", None)
df2.insert(5, "tau2", None)
df2.insert(6, "x", None)

# Calculate the channel transmissions, iterating through the df row by row
for index, row in df2.iterrows():
    tau1, tau2 = channel_transmission(row['G'], row['S'], row['T'], row['DeltaT'])
    
    if tau2 != None:
        x = tau2/row['G']
        df2.at[index,'tau1'] = tau1
        df2.at[index,'tau2'] = tau2
        df2.at[index,'x'] = x

# Drop rows where tau1 and tau2 are not defined (inconsistent with the assumption of two channels)
df2 = df2.dropna()

unique_T_vals = np.unique(df2['T'].to_numpy())
# In our remaining data there are eight unique T values

# Plot histograms of the different x values at each different T value
fig, axs = plt.subplots(2, int(np.ceil(0.5*len(unique_T_vals))))
for i in range(len(unique_T_vals)):
    T = unique_T_vals[i]
    df2_temp = df2[df2['T']==T] # Select rows with the corresponding temperature

    x_av = df2_temp['x'].mean() # Average x at each T value
    x_av = round(x_av, 3)
    count = df2_temp.shape[0] # Number of valid data points at this T value

    # Determine where in the subplot grid each plot will go (this is due to my weird indexing)
    subplot_row = int(i >= 0.5*len(unique_T_vals))
    subplot_col = i % int(np.floor(0.5*len(unique_T_vals)))

    # Plot a histogram of all x values at the given T value
    axs[subplot_row, subplot_col].hist(df2_temp['x'], bins=50, density=True)
    # Include temperature, average x, and count in the title of each subplot
    axs[subplot_row, subplot_col].set_title("$T = "+str(T)+r", \bar{x} = "+str(x_av)+", count = "+str(count)+"$")    
    if subplot_row==1:
        axs[subplot_row, subplot_col].set_xlabel("$x$")

plt.show()


# Fit S - G data to a parabola - just one temperature value for now
df3 = df[df['G']<=1]
df3 = df3[df3['T']==21.5] # Carry data points with specific T

df3['s'] = df3['S']/(G0 * kB * (pi**2/9 - 2/3) * df3['DeltaT']**2 / df3['T'])
df3 = df3.sort_values('G')

popt, pcov = curve_fit(s_model, df3['G'], df3['s'])
print(popt[0])

# Show data with the curve resulting from the fit
plt.scatter(df3['G'], df3['s'])
plt.plot(df3['G'], s_model(df3['G'], popt[0]), color='orange')
plt.xlabel("$G/G_0$")
plt.ylabel("$S/G_0k_BT$")
plt.show()


# Fit S - G data to a line for G>G_0 to try and infer the limit to which channels after the first open and then stop
dflin = df[df['G'] >= 1]
dflin['s'] = dflin['S']/(G0 * kB * (pi**2/9 - 2/3) * dflin['DeltaT']**2 / dflin['T'])

# Helps to remove outliers for the fit
dflin = dflin[dflin['s'] > 0]
dflin = dflin[dflin['s'] < 5]

dflin = dflin.sort_values('G')

popt_lin, pcov_lin = curve_fit(s_linear, dflin['G'], dflin['s'])
print(popt_lin[0])


plt.plot(dflin['G'], dflin['s'], marker='.')
plt.plot(dflin['G'], s_linear(dflin['G'], popt_lin[0]), color='orange')
plt.show()
