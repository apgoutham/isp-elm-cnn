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
import time

num_images = 5

#fig, ax = plt.subplots(2,num_images)

fin_x = np.zeros(shape=(num_images, L_forward, L_forward))
fin_y = np.zeros(shape=(num_images, M, V)).astype('complex128')
fin_ctr = np.zeros(shape=(num_images, 5))
r_time=[]

for i in range(num_images):
    t1 = time.time()
    x_au, contrast = gen_shapes.NMM_profile(L_forward)
    #print(i) 
    x = np.reshape(x_au,[L_forward*L_forward,1])
    y, _ = util_cgfft.cg_fft_forward_problem(x, G_S_forward, g_D_fft_forward, e_forward, 1e-6, e_forward, 1000)

    #fin_x[i,:,:] = x_au
    #fin_y[i,:,:] = y
    #fin_ctr[i,:] = contrast
    
    t2 = time.time()
    r_time.append(t2-t1)

    if i == 0:
        f1=open('fin_x.npy','wb')
        np.save(f1,x_au.reshape(1,x_au.shape[0],x_au.shape[1]))
        f1.close()

        f2=open('fin_y.npy','wb')
        np.save(f2,y.reshape(1,y.shape[0],y.shape[1]))
        f2.close()

        f3=open('fin_ctr.npy','wb')
        np.save(f3,np.reshape(contrast, (1,len(contrast))))
        f3.close()

    else:
        z=np.load('fin_x.npy')
        z=np.append(z,x_au.reshape(1,x_au.shape[0],x_au.shape[1]),axis=0)
        f1=open('fin_x.npy','wb')
        np.save(f1,z)
        f1.close()

        z=np.load('fin_y.npy')
        z=np.append(z,y.reshape(1,y.shape[0],y.shape[1]),axis=0)
        f2=open('fin_y.npy','wb')
        np.save(f2,z)
        f2.close()

        z=np.load('fin_ctr.npy')
        z=np.append(z,np.reshape(contrast,(1,len(contrast))), axis=0)
        f3=open('fin_ctr.npy','wb')
        np.save(f3,z)
        f3.close()


    #im=ax[0, i].imshow(x_au)
    #fig.colorbar(im, ax=ax[0,i])
    #im=ax[1, i].imshow(np.absolute(y))
    #fig.colorbar(im, ax=ax[1,i])



#with open('final_x.npy', 'ab') as f:
#    np.save(f, fin_x)
#    f.close()

#with open('final_y.npy', 'ab') as f:
#    np.save(f, fin_y)
#    f.close()

#with open('final_contrast.npy', 'ab') as f:
#    np.save(f, fin_ctr)
#    f.close()

print("minimum run time: "+str(np.min(r_time)))
print("maximum run time: "+str(np.max(r_time)))
print("mean run time: "+str(np.mean(r_time)))

#print(fin_x)
#print(fin_y)
#plt.show()

