'''
synthetic_data.py

Generate synthetic data with hopes of using it to train a neural network to learn the relationship between average temperature (T), temperature difference (delta T), conductance (G) and delta-T shot noise (S) in break junction experiments.

This procedure for generating data will sample a distribution for x, a parameter governing the sequential opening of channels (as in Phys. Rev. Lett. 82(7), 1526(4), 1999). Then, at a given choice of T and delta T, it will generate a range of G and S values corresponding to different x values. THe goal is to cover a range of different T and delta T values with a range of different channel transmissions.

Matthew Gerry, September 2024
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Set pyplot global params to get nice fonts
plt.rcParams["mathtext.fontset"] = "cm" # Match Latex font as closely as possible
plt.rcParams['font.size'] = '14' # Globally set font size of plots
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams['axes.unicode_minus'] = False # NEEDED FOR MINUS SIGNS TO RENDER PROPERLY but can't give me back that hour of my life in 2021


### CONSTANTS ###

kB = 1.380649e-23 # J/K, Boltzmann constant
q = 1.6e-19 # C, electron charge
h = 6.626e-34 # Js, Planck's constant

G0 = 2*q**2/h # C^2/Js, conductance quantum


### FUNCTIONS ###

def get_tau(G, x, tau_max, tau_noise):
    '''
    ANALOGOUS TO get_tau BUT EACH CHANNEL OPENS TO A DIFFERENT MAXIMUM VALUE.
    THE ARGUMENT tau_max IS NOW A DESCENDING LIST OF MAXIMUM TRANSMISSION VALUES. THE LAST VALUE IN THE LIST WILL BE THE MAXIMUM TRANSMISSION FOR ALL SUBSEQUENT CHANNELS.
    tau_noise SETS THE WIDTH OF A RANGE OF VALUES WHOSE UPPER BOUND IS THE RELEVANT tau_max FROM WHICH TAU WILL BE CHOSEN.
    '''

    if G < 0:
        raise ValueError("Negative values of G are unphysical.")
    if x < 0 or x > 0.5:
        raise ValueError("x must be between 0 and 0.5.")
    if sum([int(tau_max[i] < tau_max[i+1]) for i in range(len(tau_max) - 1)]) > 0:
        raise ValueError("tau_max must be in descending order.")
    if sum([int(t < 0) for t in tau_max]) > 0 or sum([int(t > 1) for t in tau_max]) > 0:
        raise ValueError("All values in tau_max must be numbers between 0 and 1.")

    # Assume there can be up to four additional channels that can open up to the smallest tau_max value
    tau_max = tau_max + 5*[tau_max[-1]]
    tau_max = tau_max - tau_noise*np.random.rand(len(tau_max)) # Add some noise to the maximum channel transmissions
    # tau_max = np.sort(tau_max)[::-1] # Re-sort in descending order with noise added - noise should be smaller than the gap between tau_max values fed in

    if G > sum(tau_max):
        raise ValueError("The function to generate tau values assumes there are no more than five relevant channels with the minimum max transmission. Choose a smaller conductance value.")
    
    # Initialize array for output - add the possibility for two more partially open channels
    tau = np.zeros(len(tau_max) + 2)

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


def generate_data(num, Gmax, x_av, tau_max_list, tau_noise):
    ''' GENERATE A SET OF DIMENSIONLESS s VALUES BASED ON RANDOMLY SAMPLED G, x, AND tau_max '''

    # Initialize arrays to hold the generated data
    G_data = np.zeros(num)
    s_data = np.zeros(num)
    tausquared_data = np.zeros(num) # We'll need to save the sum of tau values squared for calculating the full integral for S

    i = 0
    while i < num:
        G_input = np.random.uniform(0, Gmax) # Sample a uniform distribution to get G

        x = 1
        while x > 0.4: # Sample an exponential distribution to get x, throw out if greater than 0.5
            x = -x_av*np.log(np.random.uniform())

        # tau_max = np.random.uniform(tau_max_lower, tau_max_upper)

        tau = get_tau(G_input, x, tau_max_list, tau_noise)

        G_data[i] = sum(tau) # Record true G value, which can differ from the input G value due to the noise that is added when we calculate taus
        s_data[i] = sum(np.multiply(tau, 1 - tau))
        tausquared_data[i] = sum(np.power(tau, 2))

        i += 1
    
    return G_data, s_data, tausquared_data


def approx_S(s, T, deltaT):
    ''' FROM THE NON-DIMENSIONALIZED s VALUE, RETURN THE VALUE OF THE SHOT NOISE WITH UNITS OF CHARGE^2/TIME, BASED ON THE LEADING ORDER APPROXIMATION IN T '''    
    return s * ((np.pi**2)/9 - 2/3) * G0 * kB * (deltaT**2) / T


def fermi(E, mu, T):
    ''' FERMI-DIRAC DISTRIBUTION, CONVERT TEMPERATURE TO APPROPRIATE ENERGY UNITS WHEN IMPLEMENTED '''
    return (1 + np.exp((E - mu)/T))**(-1)


def full_S(s, tausquared, G, T, deltaT):
    ''' FULL VALUE OF S FROM NON-DIMENSIONALIZED s BASED ON THE FULL INTEGRAL EXPRESSION, ASSUMING NO CHEMICAL POTENTIAL DIFFERENCE BETWEEN THE LEADS '''
    
    # Parameters for the integration - cover the whole region where the difference between fermi functions is considerably nonzero
    # Energy parameters here are in eV
    dE = 0.01
    bounds = [-350, 350]
    E_axis = np.arange(bounds[0], bounds[1], dE) # Energy axis over which to integrate (in eV)

    # Get the temperatures of the hot and cold leads
    Thot = T + 0.5*deltaT; Tcold = T - 0.5*deltaT

    # Calculate the fermi functions
    fhot = fermi(E_axis, 0, Thot); fcold = fermi(E_axis, 0, Tcold)

    # Carry out the integration, multiply factor of kB due to scaling of energy values used in calculating the fermi functions
    # Integrands are functions of the Fermi-Dirac distributions
    integrand1 = np.multiply(fhot, 1 - fhot) + np.multiply(fcold, 1 - fcold)
    integrand2 = np.multiply(fhot, 1 - fcold) + np.multiply(fcold, 1 - fhot)

    # Integrate to get the shot noise, subtract off the thermal noise which is simply proportional to T
    S1 = 2 * G0 * kB * dE * sum(integrand1) * tausquared
    S2 = 2 * G0 * kB * dE * sum(integrand2) * s
    Sth = 4 * G0 * kB * T * G

    return  S1 + S2 - Sth


### MAIN CALLS ###

# Create a synthetic dataset and plot it
np.random.seed(1) # Set random seed for reproducibility

# Set paramaters for generating data
Gmax = 4.0 # Maximum conductance (scaled by G_0)
x_av = 0.1 # Average value of quantity x characterizing channel opening
num_points_at_temp = 500 # Number of data points to generate at each T, delta T pair

# Parameterize the maximum channel transmissions
tau_max_list = [1, 0.8, 0.6, 0.4]
tau_noise = 0.1

# Set parameters for the generation of T-deltaT pairs
Tmin = 10 # Minimum temperature value
Tmax = 25 # Maximum temperature value
num_temps = 20 # Number of different temperature values to use
deltaT_over_T_range = [0.2, 2.0] # Range to which deltaT/T is restricted (bounds of a subinterval of the interval from 0 to 2)

# Generate a uniform grid of delta T-T pairs throughout the region specified above
T_vals = []
deltaT_vals = []
T_grid_num = 5
T_list0 = np.linspace(10,25,num=T_grid_num)

for i in range(T_grid_num):
  
  T = np.repeat(T_list0[i],i+2)
  T_vals = np.concatenate((T_vals,T))
  
  dT = np.linspace(5,2*max(T),num=i+2)
  deltaT_vals = np.concatenate((deltaT_vals,dT))

# # Based on the parameter values above, generate several random T-DeltaT pairs
# T_vals = Tmin + (Tmax - Tmin)*np.random.rand(num_temps) # Specified amount of random temperature values between Tmin and Tmax
# deltaT_vals = np.zeros(len(T_vals))
# for i in range(len(T_vals)):
#     T = T_vals[i]
#     deltaT_vals[i] = np.random.uniform(deltaT_over_T_range[0]*T, deltaT_over_T_range[1]*T)


# Generate the synthetic G and S data
df = pd.DataFrame() # Initialize a dataframe to store the snythetic data

for i in range(len(T_vals)):
    # Draw the temperature and delta T from the list of values generated above
    T = T_vals[i]
    deltaT = deltaT_vals[i]

    # Generate a bunch of non-dimensionalized G and s data
    G, s, tausquared = generate_data(num_points_at_temp, Gmax, x_av, tau_max_list, tau_noise)

    # Map the shot noise to the corresponding dimensionful quantity
    S = approx_S(s, T_vals[i], deltaT_vals[i])
    S_full = full_S(s, tausquared, G, T_vals[i], deltaT_vals[i])

    # Create a dataframe with all the synthetic data that also stores the T and delta T values
    df_temp = pd.DataFrame()
    df_temp['G'] = G
    df_temp['S'] = S
    df_temp['S_full'] = S_full
    df_temp['T'] = T_vals[i]
    df_temp['DeltaT'] = deltaT_vals[i]

    # Add to the dataframe holding the full synthetic dataset
    df = pd.concat([df, df_temp])


# # Save the synthetic data to a csv file
# filename = 'Models/SynIP_' + str(int(Gmax)) + 'G0_uT.csv'
# df.to_csv(filename, index=False)

# Dimensionless quantity representing the shot noise but still with temperature dependence (in contrast to the output of the function generate_data)
df['S_scaled'] = df['S']/(G0 * kB * df['T'])
df['S_full_scaled'] = df['S_full']/(G0 * kB * df['T'])

# df = df[df['T']<8.0]

# Visualize the synthetic data in a scatter plot
# Start with T vs Delta T - include also the T/Delta T values from the experiment

# Experimental temperatures
df_exp = pd.read_csv('../GNoiseData_complete.csv')
T_exp = df_exp['T'].unique()
T_exp = np.insert(T_exp, [8, 8], [12.0, 15.0])
dT_exp = df_exp['DeltaT'].unique()

# For visualization purposes, draw lines that bound the region of T-deltaT space we are probing
minline = deltaT_over_T_range[0]*np.linspace(0, 1.5*Tmax)
maxline = deltaT_over_T_range[1]*np.linspace(0, 1.5*Tmax)

# Plots
fig1, ax1 = plt.subplots()
ax1.scatter(T_vals, deltaT_vals, marker='o', color='k')
ax1.scatter(T_exp, dT_exp, marker='o', color='c')
ax1.plot(np.linspace(0, 1.5*Tmax), minline, linestyle='--', color='blue')
ax1.plot(np.linspace(0, 1.5*Tmax), np.linspace(0, 1.5*Tmax), linestyle='--', color='c')
ax1.plot(np.linspace(0, 1.5*Tmax), maxline, linestyle='--', color='blue')
ax1.vlines([Tmin, Tmax], 0, 2*Tmax, linestyle='--', color='red')
ax1.set_xlabel('$T$')
ax1.set_ylabel('$\Delta T$')
ax1.legend(['Synthetic', 'Experiment'])
ax1.text(25.3,7,'$\Delta T = $' + str(deltaT_over_T_range[0]) + '$T$', color='b')
ax1.text(25.3,28,'$\Delta T = T$', color='c')
ax1.text(21.5,46,'$\Delta T = $' + str(deltaT_over_T_range[1]) + '$T$', color='b')
ax1.set_xlim([0, 1.1*Tmax])
ax1.set_ylim([0, 2*Tmax])

fig2, axs = plt.subplots(1,2)
sctr1 = axs[0].scatter(df['G'], df['S_scaled'], s=0.4, c=df['DeltaT']/df['T'])
axs[0].set_xlim([0, Gmax])
axs[0].set_ylim([0, 2.5])
axs[0].set_xlabel('$G/G_0$')
axs[0].set_ylabel('$S/G_0k_BT$ (approx)')
plt.colorbar(sctr1, label='$\Delta T/T$')

sctr2 = axs[1].scatter(df['G'], df['S_full_scaled'], s=0.4, c=df['DeltaT']/df['T'])
axs[1].set_xlim([0, Gmax])
axs[1].set_ylim([0, 2.5])
axs[1].set_xlabel('$G/G_0$')
axs[1].set_ylabel('$S/G_0k_BT$ (full)')
plt.colorbar(sctr2, label='$\Delta T/T$')
plt.show()


# Visualize the channel-opening protocol
G_list = np.arange(0, 4, 0.03) # List of G values for horizontal axis
tau_result = np.zeros([len(tau_max_list) + 7, len(G_list)]) # Initialize arrays to hold results
sumtau = np.zeros(len(G_list)) # To check that all the tau's sum up to G

for i in range(len(G_list)): # Calculate the list of taus at every value of G
    # Calculate the transmissions here without noise--we will show the noise by shading a whole region on the plot
    tau_result[:, i] = get_tau(G_list[i], 0.1, tau_max_list, 0)
    sumtau[i] = sum(tau_result[:, i])

# Plot each channel's transmission as a separate curve
fig, ax = plt.subplots()
for i in range(len(tau_result[:, 0])):
    ax.plot(G_list, tau_result[i, :])

    tau_lower = max(tau_result[i, :]) - tau_noise # Lower bound of the max transmission for each channel

    ax.fill_between(G_list, tau_lower, tau_result[i, :], alpha=0.2, where=tau_result[i,:] > tau_lower)
ax.set_xlabel("$G$ (input)")
ax.set_ylabel(r"$\tau_n$")
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 3.9])
plt.show()


# One polished figure with the sythetic data and channel opening protocol
fig4, ax = plt.subplots()
sctr = ax.scatter(df['G'], df['S_scaled'], s=0.4, c=df['DeltaT']/df['T'])
ax.set_xlim([0, Gmax])
ax.set_ylim([0, 2.5])
ax.set_xlabel('$G/G_0$')
ax.set_ylabel(r'$S/G_0k_B\bar{T}$')
plt.colorbar(sctr, label=r'$\Delta T/\bar{T}$')

# Channel opening protocal in inset
axins = fig4.add_axes([0.217, 0.52, 0.32, 0.33])
for i in range(len(tau_result[:, 0])):
    axins.plot(G_list, tau_result[i, :])

    tau_lower = max(tau_result[i, :]) - tau_noise # Lower bound of the max transmission for each channel

    axins.fill_between(G_list, tau_lower, tau_result[i, :], alpha=0.2, where=tau_result[i,:] > tau_lower)
axins.set_xlabel("$G/G_0$ (input)", fontsize=12)
axins.set_ylabel(r"$\tau_n$", fontsize=12)
axins.tick_params(axis='x', labelsize=10)
axins.tick_params(axis='y', labelsize=10)
axins.set_ylim([0, 1.05])
axins.set_xlim([0, 3.9])
plt.show()