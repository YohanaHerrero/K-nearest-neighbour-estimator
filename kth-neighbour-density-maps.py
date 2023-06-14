import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import math
from matplotlib.ticker import FormatStrFormatter  
import scipy.ndimage as ndimage
from sklearn.neighbors import NearestNeighbors


N=5 #5 neighbours, to define by the user
#your 3D data: redshift, right ascension and declination
RAf_mxdf = np.array([...])
Decf_mxdf = np.array([...])
Zf_mxdf = np.array([...])

#define sky grid for your 3D coordinates
z_grid = np.arange(3.,6.1,0.1) 
sky_grid_ra = np.arange(np.min(RAf_mxdf),np.max(RAf_mxdf)+0.003,0.003)
sky_grid_dec = np.arange(np.min(Decf_mxdf),np.max(Decf_mxdf)+0.003,0.003) 

#compute K-neighbour estimator
for k, zk in enumerate(z_grid):
    for_mean=np.array([])
    if k<len(z_grid)-1:
        #select galaxies within your redshift slice
        select = (Zf_mxdf>=zk) & (Zf_mxdf<z_grid[k+1]) 
        ra = RAf_mxdf[select]
        dec = Decf_mxdf[select]
        z = Zf_mxdf[select]   
        #calculate density of each grid point on sky
        densityN = np.array([])
        if len(z)>=1:
            for i, item in enumerate(sky_grid_ra):
                for j, itemj in enumerate(sky_grid_dec):
                    distance = np.sqrt((item-ra)**2+(itemj-dec)**2)#+(zk-z)**2)
                    distance_sorted = np.sort(distance)                  
                    
                    densityN = np.append(densityN, N/(math.pi * distance_sorted[N]**2)) #projected density
                    for_mean = np.append(for_mean, N/(math.pi * distance_sorted[N]**2)) #projected density to compute the mean density
        
        densityN_matrix = densityN.reshape((len(sky_grid_ra), len(sky_grid_dec))).T
        #select the group of N neighbours
        xy=np.vstack((ra, dec)).T
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='brute').fit(xy)
        distances, indices = nbrs.kneighbors(xy) #projected distances between galaxy_i and N neighbours         
        distances_mean=np.array([])
        for j, itemj in enumerate(distances):
            distances_mean=np.append(distances_mean,np.mean(itemj))
        distances_mean_IDmin=np.argmin(distances_mean)
        min_indices=indices[distances_mean_IDmin]
        #Coordinates of the N galaxies that belong to the overdensity
        ra_group=ra[min_indices]
        dec_group=dec[min_indices]
        z_group=z[min_indices]
        
        #neighbour map of densities above the mean (density contrast)
        overdensity_matrix=densityN_matrix/np.mean(for_mean)
        #smooth the density matrix with gaussian filter
        smoothed_overdensity_matrix=ndimage.gaussian_filter(overdensity_matrix, sigma=.8, order=0)
        #plot K-neighbour map results
        fig = plt.figure().add_subplot(111)
        cm = plt.cm.get_cmap('jet')
        centers = [np.min(sky_grid_ra),np.max(sky_grid_ra),np.min(sky_grid_dec),np.max(sky_grid_dec)]
        dx, = np.diff(centers[:2])/(smoothed_overdensity_matrix.shape[1]-1)
        dy, = -np.diff(centers[2:])/(smoothed_overdensity_matrix.shape[0]-1)
        extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
        plt.imshow(smoothed_overdensity_matrix,cmap=plt.cm.jet, origin='lower',extent=extent, aspect="auto")
        plt.xlabel('RA', fontsize = 14)
        plt.ylabel('Dec', fontsize = 14)
        plt.title(str(round(zk,2))+'<z<'+str(round(z_grid[k+1],2)))
        cbar=plt.colorbar()
        cbar.set_label(r'Density contrast', fontsize = 12)
        cbar.ax.tick_params( direction='in')
        fig.xaxis.set_ticks_position('both')
        fig.yaxis.set_ticks_position('both')
        fig.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.xaxis.set_tick_params(direction='in', which='both')
        fig.yaxis.set_tick_params(direction='in', which='both')
        plt.grid(False)
        fig.tick_params(labelsize = 'large')
        plt.scatter(ra, dec, s=10, c='k', marker='.')
        plt.scatter(ra_group, dec_group,  c='k', marker='X')
        plt.gca().invert_xaxis()
        plt.tight_layout()
        #plt.savefig("Smoothed density contrast map from N-neighbour, z slice "+str(round(zk,2))+".png",dpi=200)
        plt.show()   
    else:
        print('There are no galaxies with redshift between',round(zk,1),round(z_grid[k],1))
