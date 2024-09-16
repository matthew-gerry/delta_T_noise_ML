'''
synthetic_data.py

Generate synthetic data with hopes of using it to train a neural network to learn the relationship between average temperature (T), temperature difference (delta T), conductance (G) and delta-T shot noise (S) in break junction experiments.

This procedure for generating data will sample a distribution for x, a parameter governing the sequential opening of channels (as in Phys. Rev. Lett. 82(7), 1526(4), 1999). Then, at a given choice of T and delta T, it will generate a range of G and S values corresponding to different x values. THe goal is to cover a range of different T and delta T values with a range of different channel transmissions.

Matthew Gerry, September 2024
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define constants

G0 = 7.748092*10**(-5) # C^2/Js, conductance quantum
kB = 1.380649*10**(-23) # J/K, Boltzmann constant

### FUNCTIONS ###

def get_tau(G, x, tau_max):
    ''' TRANSMISSION OF EACH CHANNEL AS A FUNCTION OF THE CONDUCTANCE BASED ON A CHOSEN x CHARACTERIZING SEQUENTIAL OPENING AND tau_max RERPESENTING THE EXTENT TO WHICH CHANNELS OPEN (EXCEPT CHANNEL 1 WHICH OPENS ALL THE WAY). '''

    if x < 0 or x > 0.5:
        raise ValueError("x must be between 0 and 0.5.")
    if tau_max < 0.5 or tau_max > 1:
        raise ValueError("tau_max must be between 0 and 1.")
    if G > 1 + 28 * tau_max:
        raise ValueError("This function assumes there are no more than 20 transmission channels. Accordingly, choose a smaller value of g")

    # Assume there are no more than 20 transmission channels
    tau = np.zeros(30) # Initialize array for output
    # Channels that are not open will retain the value of zero for the transmission
    
    if G <= 1:
        ''' IN THE LOW G CASE, ASSUME TWO CHANNELS ARE OPEN, AND THAT tau_max DOES NOT BOUND THE TRANSMISSION OF THE FIRST CHANNEL '''

        # Linear functions for each channel transmission, they should add up to G
        tau[0] = G * (1 - x)
        tau[1] = G * x
    
    if G > 1 and G <= 1 + tau_max:
        ''' IN THIS INTERMEDIATE REGIME, THERE IS A SPECIAL PROCEDURE FOR OPENING CHANNELS TO BRIDGE TO CHANNEL 1 BEING FULLY OPEN '''

        tau[0] = (1 - x) + (G - 1) * x/tau_max
        tau[1] = x + (G - 1) * (1 - x - x/tau_max)
        tau[2] = (G - 1)*x
    
    if G > 1 + tau_max:
        ''' FROM HERE ON, THERE ARE ALWAYS THREE PARTIALLY OPEN CHANNELS WHICH OPEN IN A CONSISTENT MANNER UP TO tau_max '''
        n = int(np.floor((G - 1)/tau_max)) # Number of maximally open channels

        tau[0] = 1 # The first channel is fully open
        for i in range(1, n):
            tau[i] = tau_max # Channels above the first that are maximally open, if any

        tau[n] = (1 - x) * tau_max + (G - 1 - n*tau_max) * x
        tau[n + 1] = tau_max * x + (1 - 2*x)*(G - 1 - n*tau_max)
        tau[n + 2] = (G - 1 - n*tau_max) * x

    return tau


def generate_data(num, Gmax, x_av, tau_max_lower, tau_max_upper):
    ''' GENERATE A SET OF DIMENSIONLESS s VALUES BASED ON RANDOMLY SAMPLED G, x, AND tau_max '''

    s_data = np.zeros(num) # Initialize array to hold the generated data
    G_data = np.zeros(num)

    i = 0
    while i < num:
        G = np.random.uniform(0, Gmax) # Sample a uniform distribution to get G
        G_data[i] = G # Record G value

        x = 1
        while x > 0.5: # Sample an exponential distribution to get x, throw out if greater than 0.5
            x = -x_av*np.log(np.random.uniform())

        # tau_max = 0.5 # For now, just set tau_max constant
        tau_max = np.random.uniform(tau_max_lower, tau_max_upper)

        tau = get_tau(G, x, tau_max)
        s_data[i] = sum(np.multiply(tau, 1-tau))

        i += 1
    
    return G_data, s_data

def full_S(s, T, deltaT):
    ''' FROM THE NON-DIMENSIONALIZED s VALUE, RETURN THE VALUE OF THE SHOT NOISE WITH UNITS OF CHARGE^2/TIME '''    
    return s * (np.pi**2/9 - 2/3) * G0 * kB * (deltaT**2) / T
    

### MAIN CALLS ###

# Create a synthetic dataset and plot it
np.random.seed(1) # Set random seed for reproducibility

# Set paramaters for generating data
Gmax = 8.0 # Maximum conductance (scaled by G_0)
x_av = 0.1 # Average value of quantity x characterizing channel opening
num_points_at_temp = 500 # Number of data points to generate at each T, delta T pair

# Lower and upper bounds on the randomly sampled value of tau_max
tau_max_lower = 0.65
tau_max_upper = 0.9

# Set parameters for the generation of T-deltaT pairs
Tmin = 10 # Minimum temperature value
Tmax = 25 # Maximum temperature value
num_temps = 30 # Number of different temperature values to use
deltaT_over_T_range = [0.25, 1.75] # Range to which deltaT/T is restricted (bounds of a subinterval of the interval from 0 to 2)

# Based on the parameter values above, generate several random T-DeltaT pairs
T_vals = Tmin + (Tmax - Tmin)*np.random.rand(num_temps) # Specified amount of random temperature values between Tmin and Tmax
deltaT_vals = np.zeros(len(T_vals))
for i in range(len(T_vals)):
    T = T_vals[i]
    deltaT_vals[i] = np.random.uniform(deltaT_over_T_range[0]*T, deltaT_over_T_range[1]*T)


# Generate the synthetic G and S data
df = pd.DataFrame() # Initialize a dataframe to store the snythetic data

for i in range(len(T_vals)):
    # Draw the temperature and delta T from the list of values generated above
    T = T_vals[i]
    deltaT = deltaT_vals[i]

    # Generate a bunch of non-dimensionalized G and s data
    G, s = generate_data(num_points_at_temp, Gmax, x_av, tau_max_lower, tau_max_upper)

    # Map the shot noise to the corresponding dimensionful quantity
    S = full_S(s, T_vals[i], deltaT_vals[i])

    # Create a dataframe with all the synthetic data that also stores the T and delta T values
    df_temp = pd.DataFrame()
    df_temp['G'] = G
    df_temp['S'] = S
    df_temp['T'] = T_vals[i]
    df_temp['DeltaT'] = deltaT_vals[i]

    # Add to the dataframe holding the full synthetic dataset
    df = pd.concat([df, df_temp])


# Save the synthetic data to a csv file
df.to_csv("../synthetic_data_deltaT_shot_noise.csv", index=False)

# Dimensionless quantity representing the shot noise but still with temperature dependence (in contrast to the output of the function generate_data)
df['S_scaled'] = df['S']/(G0 * kB * df['T'])

# Visualize the synthetic data in a scatter plot
# Start with T vs Delta T

# For visualization purposes, draw lines that bound the region of T-deltaT space we are probing
minline = deltaT_over_T_range[0]*np.linspace(0, 1.5*Tmax)
maxline = deltaT_over_T_range[1]*np.linspace(0, 1.5*Tmax)

fig, axs = plt.subplots(1,2)
axs[0].scatter(T_vals, deltaT_vals, marker='.', color='k')
axs[0].plot(np.linspace(0, 1.5*Tmax), minline, color='blue')
axs[0].plot(np.linspace(0, 1.5*Tmax), maxline, color='blue')
axs[0].vlines([Tmin, Tmax], 0, 2*Tmax, colors='red')
axs[0].set_xlabel('$T$')
axs[0].set_ylabel('$\Delta T$')
axs[0].set_xlim([0, 1.1*Tmax])
axs[0].set_ylim([0, 2*Tmax])

sctr = axs[1].scatter(df['G'], df['S_scaled'], s=0.4, c=df['DeltaT']/df['T'])
axs[1].set_xlim([0, Gmax])
axs[1].set_ylim([0, 2.5])
axs[1].set_xlabel('$G/G_0$')
axs[1].set_ylabel('$S/G_0k_BT$')
plt.colorbar(sctr, label='$\Delta T/T$')
plt.show()



# Visualize the channel-opening protocol
G_list = np.arange(0, 4, 0.1) # List of G values for horizontal axis
tau_result = np.zeros([30, len(G_list)]) # Initialize arrays to hold results
sumtau = np.zeros(len(G_list))

for i in range(len(G_list)): # Calculate the list of taus at every value of G
    tau_result[:, i] = get_tau(G_list[i], 0.1, 0.75)
    sumtau = sum(tau_result[:, i])

# Plot each channel's transmission as a separate curve
fig, ax = plt.subplots()
for i in range(len(tau_result[:, 0])):
    ax.plot(G_list, tau_result[i, :])
ax.set_xlabel("$G$")
ax.set_ylabel(r"$\tau_n$")
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 3.5])
plt.show()