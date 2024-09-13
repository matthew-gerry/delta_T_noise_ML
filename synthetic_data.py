'''
synthetic_data.py

Generate synthetic data with hopes of using it to train a neural network to learn the relationship between average temperature (T), temperature difference (Delta T), conductance (G) and delta-T shot noise (S) in break junction experiments.

This procedure for generating data will sample a distribution for x, a parameter governing the sequential opening of channels (as in Phys. Rev. Lett. 82(7), 1526(4), 1999). Then, at a given choice of T and Delta T, it will generate a range of G and S values corresponding to different x values. THe goal is to cover a range of different T and Delta T values with a range of different channel transmissions.

Matthew Gerry, September 2024
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### FUNCTIONS ###

def get_tau(g, x, tau_max):
    ''' TRANSMISSION OF EACH CHANNEL AS A FUNCTION OF THE CONDUCTANCE BASED ON A CHOSEN x CHARACTERIZING SEQUENTIAL OPENING AND tau_max RERPESENTING THE EXTENT TO WHICH CHANNELS OPEN (EXCEPT CHANNEL 1 WHICH OPENS ALL THE WAY). '''

    if x < 0 or x > 0.5:
        raise ValueError("x must be between 0 and 0.5.")
    if tau_max < 0.5 or tau_max > 1:
        raise ValueError("tau_max must be between 0 and 1.")
    if g > 1 + 28 * tau_max:
        raise ValueError("This function assumes there are no more than 20 transmission channels. Accordingly, choose a smaller value of g")

    # Assume there are no more than 20 transmission channels
    tau = np.zeros(30) # Initialize array for output
    # Channels that are not open will retain the value of zero for the transmission
    
    if g <= 1:
        ''' IN THE LOW G CASE, ASSUME TWO CHANNELS ARE OPEN, AND THAT tau_max DOES NOT BOUND THE TRANSMISSION OF THE FIRST CHANNEL '''

        # Linear functions for each channel transmission, they should add up to g
        tau[0] = g * (1 - x)
        tau[1] = g * x
    
    if g > 1 and g <= 1 + tau_max:
        ''' IN THIS INTERMEDIATE REGIME, THERE IS A SPECIAL PROCEDURE FOR OPENING CHANNELS TO BRIDGE TO CHANNEL 1 BEING FULLY OPEN '''

        tau[0] = (1 - x) + (g - 1) * x/tau_max
        tau[1] = x + (g - 1) * (1 - x - x/tau_max)
        tau[2] = (g - 1)*x
    
    if g > 1 + tau_max:
        ''' FROM HERE ON, THERE ARE ALWAYS THREE PARTIALLY OPEN CHANNELS WHICH OPEN IN A CONSISTENT MANNER UP TO tau_max '''
        n = int(np.floor((g - 1)/tau_max)) # Number of maximally open channels

        tau[0] = 1 # The first channel is fully open
        for i in range(1, n):
            tau[i] = tau_max # Channels above the first that are maximally open, if any

        tau[n] = (1 - x) * tau_max + (g - 1 - n*tau_max) * x
        tau[n + 1] = tau_max * x + (1 - 2*x)*(g - 1 - n*tau_max)
        tau[n + 2] = (g - 1 - n*tau_max) * x

    return tau


def generate_s(num, gmax, x_av, seed):
    ''' GENERATE A SET OF DIMENSIONLESS s VALUES BASED ON RANDOMLY SAMPLED G, x, AND tau_max '''

    np.random.seed(seed) # Set the random seed

    s = np.zeros(num) # Initialize array to hold the generated data
    g_list = np.zeros(num)

    i = 0
    while i < num:
        g = np.random.uniform(0, gmax) # Sample a uniform distribution to get g
        g_list[i] = g # Record g value

        x = 1
        while x > 0.5: # Sample an exponential distribution to get x, throw out if greater than 0.5
            x = -x_av*np.log(np.random.uniform())

        tau_max = 0.75 # For now, just set tau_max constant

        tau = get_tau(g, x, tau_max)
        s[i] = sum(np.multiply(tau, 1-tau))

        i += 1
    
    return g_list, s

    

### MAIN CALLS ###

# g_list = np.arange(0, 4, 0.1)
# tau0, tau1, tau2, tau3, tau4, tau5, tau6, sumtau = np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list)), np.zeros(len(g_list))

# for i in range(len(g_list)):
#     tau_array = get_tau(g_list[i], 0.1, 0.5)

#     tau0[i] = tau_array[0]
#     tau1[i] = tau_array[1]
#     tau2[i] = tau_array[2]
#     tau3[i] = tau_array[3]
#     tau4[i] = tau_array[4]
#     tau5[i] = tau_array[5]
#     tau6[i] = tau_array[6]

#     sumtau[i] = sum(tau_array)


# plt.plot(g_list, tau0)
# plt.plot(g_list, tau1)
# plt.plot(g_list, tau2)
# plt.plot(g_list, tau3)
# plt.plot(g_list, tau4)
# plt.plot(g_list, tau5)
# plt.plot(g_list, tau6)
# plt.show()


g_list, s = generate_s(100, 10, 0.1, 1)

plt.scatter(g_list, s)