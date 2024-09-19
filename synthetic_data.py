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


def get_tau_2(G, x, tau_max):
    '''
    ANALOGOUS TO get_tau BUT EACH CHANNEL OPENS TO A DIFFERENT MAXIMUM VALUE.
    THE ARGUMENT tau_max IS NOW A DESCENDING LIST OF MAXIMUM TRANSMISSION VALUES. THE LAST VALUE IN THE LIST WILL BE THE MAXIMUM TRANSMISSION FOR ALL SUBSEQUENT CHANNELS.
    '''

    # Assume there can be up to four additional channels that can open up to the smallest tau_max value
    tau_max = tau_max + 4*[tau_max[-1]]
    
    # Initialize array for output - add the possibility for two more partially open channels
    tau = np.zeros(len(tau_max) + 2)

    if x < 0 or x > 0.5:
        raise ValueError("x must be between 0 and 0.5.")
    if G > sum(tau_max):
        raise ValueError("The function to generate tau values assumes there are no more than four relevant channels with the minimum max transmission. Choose a smaller conductance value.")
    if G < 0:
        raise ValueError("Negative values of G are unphysical.")
    if sum([int(t < 0) for t in tau_max]) > 0 or sum([int(t > 1) for t in tau_max]) > 0:
        raise ValueError("All values in tau_max must be numbers between 0 and 1.")
    if sum([int(tau_max[i] < tau_max[i+1]) for i in range(len(tau_max) - 1)]) > 0:
        raise ValueError("tau_max must be in descending order.")

    # Figure out which channels are fully and partially open
    cumtaumax = np.cumsum(tau_max)
    N = len(cumtaumax[cumtaumax < G]) # Number of channels whose max transmissions sum up to a value less than the current G
    # As we will see below, N determines the number of open channels
    if N >= 1:
        lowsum = max(cumtaumax[cumtaumax < G]) # Largest cumulative sum of tau_max values less than the current G

    if N == 0: # Below the first tau_max, two channels are partially open
        tau[0] = (1 - x) * G
        tau[1] = x * G
    else: # Otherwise, three channels are partially open and all channels below (if any) are fully open
        for i in range(N - 1):
            tau[i] = tau_max[i]
        tau[N - 1] = tau_max[N - 1] * (1 - x) + (G - lowsum) * x * (tau_max[N - 1]/tau_max[N])
        tau[N] = tau_max[N - 1] * x + (G - lowsum) * (1 - x - x * tau_max[N - 1]/tau_max[N])
        tau[N + 1] = (G - lowsum) * x

    return tau


def generate_data(num, Gmax, x_av, tau_max_list):
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

        # tau_max = np.random.uniform(tau_max_lower, tau_max_upper)

        tau = get_tau_2(G, x, tau_max_list)
        s_data[i] = sum(np.multiply(tau, 1-tau))

        i += 1
    
    return G_data, s_data


def approx_S(s, T, deltaT):
    ''' FROM THE NON-DIMENSIONALIZED s VALUE, RETURN THE VALUE OF THE SHOT NOISE WITH UNITS OF CHARGE^2/TIME, BASED ON THE LEADING ORDER APPROXIMATION IN T '''    
    return s * (np.pi**2/9 - 2/3) * G0 * kB * (deltaT**2) / T


def fermi(E, mu, T):
    ''' FERMI-DIRAC DISTRIBUTION, ENERGY GIVEN IN UNITS OF TEMPERATURE (ENERGY DIVIDED BY kB) '''
    return 1/(1 + np.exp((E - mu)/T))


def full_S(s, T, deltaT):
    ''' FULL VALUE OF S FROM NON-DIMENSIONALIZED s BASED ON THE FULL INTEGRAL EXPRESSION, ASSUMING NO CHEMICAL POTENTIAL DIFFERENCE BETWEEN THE LEADS '''
    
    # Parameters for the integration - cover the whole region where the difference between fermi functions is considerably nonzero
    # Energy parameters here are actually E/kB (therefore in units of temperature)
    dE = 0.01
    bounds = [-350, 350]
    E = np.arange(bounds[0], bounds[1], dE) # Energy axis over which to integrate

    # Get the temperatures of the hot and cold leads
    Thot = T + 0.5*deltaT; Tcold = T - 0.5*deltaT

    # Calculate the fermi functions
    fhot = fermi(E, 0, Thot); fcold = fermi(E, 0, Tcold)

    # Carry out the integration, multiply factor of kB due to scaling of energy values used in calculating the fermi functions
    integrand = fhot * (1 - fcold) + fcold * (1 - fhot)
    S = s * 2 * G0 * kB * dE * sum(integrand) # Multiply by the factor associated with the channel transmissions as well
    return S


### MAIN CALLS ###

# Create a synthetic dataset and plot it
np.random.seed(1) # Set random seed for reproducibility

# Set paramaters for generating data
Gmax = 4.0 # Maximum conductance (scaled by G_0)
x_av = 0.1 # Average value of quantity x characterizing channel opening
num_points_at_temp = 1500 # Number of data points to generate at each T, delta T pair

# Lower and upper bounds on the randomly sampled value of tau_max
# tau_max_lower = 0.65
# tau_max_upper = 0.9
tau_max_list = [1, 0.8, 0.6, 0.4]

# Set parameters for the generation of T-deltaT pairs
Tmin = 10 # Minimum temperature value
Tmax = 25 # Maximum temperature value
num_temps = 20 # Number of different temperature values to use
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
    G, s = generate_data(num_points_at_temp, Gmax, x_av, tau_max_list)

    # Map the shot noise to the corresponding dimensionful quantity
    S = approx_S(s, T_vals[i], deltaT_vals[i])
    # S = full_S(s, T_vals[i], deltaT_vals[i])

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
axs[1].set_ylim([0, 3.5])
axs[1].set_xlabel('$G/G_0$')
axs[1].set_ylabel('$S/G_0k_BT$')
plt.colorbar(sctr, label='$\Delta T/T$')
plt.show()


# Visualize the channel-opening protocol
G_list = np.arange(0, 4, 0.1) # List of G values for horizontal axis
tau_result = np.zeros([10, len(G_list)]) # Initialize arrays to hold results
sumtau = np.zeros(len(G_list)) # To check that all the tau's sum up to G

for i in range(len(G_list)): # Calculate the list of taus at every value of G
    tau_result[:, i] = get_tau_2(G_list[i], 0.1, tau_max_list)
    sumtau[i] = sum(tau_result[:, i])

# Plot each channel's transmission as a separate curve
fig, ax = plt.subplots()
for i in range(len(tau_result[:, 0])):
    ax.plot(G_list, tau_result[i, :])
ax.set_xlabel("$G$")
ax.set_ylabel(r"$\tau_n$")
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 3.9])
plt.show()