import numpy as np
from skimage import draw as draw
import matplotlib.pyplot as plt


def draw_ring(centre, r1, r2, contrast, N): #r1>r2, r_coord: (x,y)
    ring=np.zeros((N,N), dtype=np.int8)
    x1,y1 = draw.disk(centre,r1,shape=ring.shape)
    x2,y2 = draw.disk(centre,r2,shape=ring.shape)
    ring[x1,y1]=contrast
    ring[x2,y2]=0
    return ring

def draw_rectangle(centre, w, h , contrast, N):
    rect=np.zeros((N,N),dtype=np.int8)
    start = tuple(np.subtract(centre,(w/2,h/2)).astype(int))
    extent = tuple(np.add(centre,(w/2,h/2)).astype(int))
    x,y = draw.rectangle(start, extent, shape=rect.shape)
    rect[x,y] = contrast
    return rect

def draw_disk(centre, r, contrast, N):
    disk=np.zeros((N,N),dtype=np.int)
    x,y = draw.disk(centre, r, shape=disk.shape)
    disk[x,y]=contrast
    return disk

def draw_square(centre, l, contrast, N):
    square=np.zeros((N,N), dtype=np.int8)
    start = tuple(np.subtract(centre,(l/2,l/2)).astype(int))
    extent = tuple(np.add(centre,(l/2,l/2)).astype(int))
    x,y = draw.rectangle(start, extent, shape=square.shape)
    square[x,y]=contrast
    return square

def NMM_profile(N):
    
    image=np.zeros((N,N), dtype=np.int8)
    
    ring_r1 = np.random.randint(10,30)
    ring_r2 = np.random.randint(5,ring_r1)
    ring_centre = tuple(np.random.randint(ring_r1, N-ring_r1, size=2))
    ring_contrast = np.random.randint(1,10)
    ring1 = draw_ring(ring_centre, ring_r1, ring_r2, ring_contrast, N)
    image = np.maximum(image,ring1)
    
    rect_w = np.random.randint(5,30)
    rect_h = np.random.randint(5,30)
    rect_centre = tuple(np.random.randint(max(rect_w,rect_h), N-max(rect_w,rect_h), size=2))
    rect_contrast = np.random.randint(1,10)
    rect1 = draw_rectangle(rect_centre, rect_w, rect_h, rect_contrast, N)
    image = np.maximum(image,rect1)
    
    disk1_r = np.random.randint(5,10)
    disk1_centre = tuple(np.random.randint(disk1_r, N-disk1_r, size=2))
    disk1_contrast = np.random.randint(1,10)
    disk1 = draw_disk(disk1_centre, disk1_r, disk1_contrast, N)
    image = np.maximum(image,disk1)
    
    disk2_r = np.random.randint(5,10)
    disk2_centre = tuple(np.random.randint(disk2_r, N-disk2_r, size=2))
    disk2_contrast = np.random.randint(1,10)
    disk2 = draw_disk(disk2_centre, disk2_r, disk2_contrast, N)
    image = np.maximum(image,disk2)
    
    
    square_l = np.random.randint(5,30)
    square_centre = tuple(np.random.randint(square_l, N-square_l, size=2))
    square_contrast = np.random.randint(1,10)
    square1 = draw_square(square_centre, square_l, square_contrast, N)
    image = np.maximum(image,square1)

    return image
    
    #with open('img.npy','wb') as f:
    #    np.save(f, image)
    
    #plt.imshow(image)
    #plt.colorbar()
    #plt.show()

