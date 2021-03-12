import numpy as np

# Frequency = 400MHz
f = 4e8
wavelength = 3e8/f

# k := Wavenumber
k = 2*np.pi/wavelength 

# d := Dimension of imaging domain D in meters
d = 2 

# R := Radius of measurement domain
R = 4

# M := Number of Receivers per illumination
M = 32

# V := Number of illuminations
V = 32
print('parameters initialised')

import sys
sys.path.insert(0, './utility/')
import setup_functions
print('setting up functions')
# The imaging domain is discretized into L X L cells
L = 32 
# Dimension of one cell
n = d/L 
# Number of pixels
N = L^2

# Positions of center of each grid. Shape = [N, 2]
# pos_D[i,:] = [x coordinate, y coordinate] of cell #i 
pos_D = setup_functions.gen_pos_D(d,L,n)

# Positions of Receivers and Transceivers. Shape = [M,2] and [V,2] respectively
# pos_S[i,:] = [x coordinate, y coordinate] of receiver #i 
pos_S = setup_functions.gen_pos_S(R, M, d) 
# pos_Tx[i,:] = [x coordinate, y coordinate] of transceiver #i 
pos_Tx = setup_functions.gen_pos_Tx(R*1.5, V, d)

# Incident Field, Shape = [N,1] (Complex vector)
e = setup_functions.gen_e(k, pos_D, pos_Tx)

print('starting forward solver')
# For forward solver
L_forward = 100
n_forward = d/L_forward

pos_D_forward = setup_functions.gen_pos_D(d,L_forward,n_forward)
e_forward = setup_functions.gen_e(k, pos_D_forward, pos_Tx)

print('importing util functions')
import util_cgfft
import util_functions
# Forward Solver parameters for L = 100 
# FFT representation of G_D matrix
g_D_forward, g_D_fft_forward, g_D_fft_conj_forward = util_cgfft.construct_g_D(pos_D_forward, k, n_forward)

# G_S matrix for forward solver
G_S_forward = util_functions.construct_G_S(pos_D_forward, pos_S, k, n_forward)

# Forward Solver parameters for L = 32 
# FFT representation of G_D matrix
g_D, g_D_fft, g_D_fft_conj = util_cgfft.construct_g_D(pos_D, k, n)

# G_S matrix for forward solver
G_S = util_functions.construct_G_S(pos_D, pos_S, k, n)

## Ignore the warning which comes after running the code

print('generating shapes')
import gen_shapes
import matplotlib.pyplot as plt

num_images = 3 

#fig, ax = plt.subplots(2,5)

fin_x = np.zeros(shape=(num_images, L_forward, L_forward))
fin_y = np.zeros(shape=(num_images, M, V)).astype('complex128')

for i in range(num_images):
    x_au = gen_shapes.NMM_profile(L_forward)
    #print(i) 
    x = np.reshape(x_au,[L_forward*L_forward,1])
    y, _ = util_cgfft.cg_fft_forward_problem(x, G_S_forward, g_D_fft_forward, e_forward, 1e-6, e_forward, 1000)

    fin_x[i,:,:] = x_au
    fin_y[i,:,:] = y

    #im=ax[0, i].imshow(x_au)
    #fig.colorbar(im, ax=ax[0,i])
    #im=ax[1, i].imshow(np.absolute(y))
    #fig.colorbar(im, ax=ax[1,i])

with open('final_x.npy', 'wb') as f:
    np.save(f, fin_x)

with open('final_y.npy', 'wb') as f:
    np.save(f, fin_y)

print(fin_x)
print(fin_y)
#plt.show()


##max_contrast = 10
#x_au = gen_shapes.NMM_profile(L_forward)
#
## Display Austria Profile
#
##with open('x.npy', 'wb') as f:
##    np.save(f, x_au)
#fig, axs = plt.subplots(2)
#im = axs[0].imshow(np.real(x_au))
##plt.xticks([L_forward*0.25, L_forward*0.5, L_forward*0.75], [-0.5, 0, 0.5],fontsize = '16')
##plt.yticks([L_forward*0.25, L_forward*0.5, L_forward*0.75], [-0.5, 0, 0.5],fontsize = '16')
##plt.xlabel('x (in m)', fontsize='16')
##plt.ylabel('y (in m)', fontsize='16')
##plt.title('Austria Profile', fontsize='18')
#fig.colorbar(im, ax=axs[0])
#
#print('generating scattered field')
## Generating scattered field from profile
## Reshape profile into [N,1] vector
#x = np.reshape(x_au,[L_forward*L_forward,1])
## Run the forward solver
#y, _ = util_cgfft.cg_fft_forward_problem(x, G_S_forward, g_D_fft_forward, e_forward, 1e-6, e_forward, 1000)
## Add 25dB Gaussian Noise
##y = util_functions.add_noise(y, 25)
##with open('y.npy', 'wb') as f:
##    np.save(f, y)
##print(y)
##print('saved y')
#im=axs[1].imshow(np.absolute(y))
#fig.colorbar(im, ax=axs[1])
#plt.show()
