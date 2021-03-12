import matplotlib.pyplot as plt
import numpy as np

y = np.load('y.npy')
x=np.load('x.npy')

print(x)
print(y.shape)
#fig, ax = plt.subplots(2)
#ax[0].imshow(x)
#ax[1].imshow(np.absolute(y))
plt.imshow(np.absolute(y))
plt.show()
