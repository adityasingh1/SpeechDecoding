import numpy as np
from scipy.spatial import Delaunay
import dill
import pathos.multiprocessing as mp
from mne import EpochsArray

# Preprocess functions for custom filtering or smoothing data

def get_neighbors(xy,tri=None):
	# Makes a lookup list of a node's neighbors (from which filtering is based) 

	# Make an index list, indy
	indy = range(0,xy.shape[0])

	if tri is None:
		# Returns neigh, a list of neighbors from a channel xy list
		tri = Delaunay(xy,qhull_options="QJ")


	# Find "ind" in rows then get the unique connected neighbours (flattened)
	GetN = lambda ind: \
			np.unique(tri.simplices[np.any(tri.simplices==ind,axis=1),:].flatten())

	# Find the neighbors
	neigh = list(map(GetN,indy))

	# Then remove the self reference (ind)
	neigh = list(map(\
		lambda ind: np.delete(neigh[ind],np.where(neigh[ind]==ind)),\
		indy))

	# And place back into first position [0]
	return list(map( lambda ind: np.insert(neigh[ind],0,ind), indy))


def filter_outliers(z,neigh,method='median',factor=10):
	# Apply a filter (median or mean) to one frame of data
	# factor: mad/std away from mean (outlier)
	z_tmp = z

	# Get ths method function, M, and abs -> A for convenience
	M = getattr(np,method.lower())
	A = getattr(np,'abs')

	# Find the deviation of the index:
	# If a[0] is True (self reference of get_neighbors), returns True
	method_test = lambda a : ( A(a-M(a)) > factor*M(A(a-M(a))) )[0]
	# Apply this filter test to the neighbors
	map_fun = lambda ind: method_test(z[neigh[ind]])

	# Apply filter test to the neighbors
	to_filter = list(map(map_fun,range(0,len(z))))
	
	# retrieve z of 1: neighbors (all but self) and apply method
	apply_filter = lambda ind: M( z[ neigh[ind][1:] ] )

	# Apply to z_tmp to only required values
	z_tmp[to_filter] = list(map(apply_filter,np.where(to_filter)[0]))

	# Update z_tmp >> z
	return z_tmp


def filter_dataset(Z,xy,method='median',factor=10):
	
	# Init parpool using dill and pathos.mulitprocessing for lambda compatibility
	pool = mp.Pool()

	# Get neighbors once
	neigh = get_neighbors(xy)

	# Apply this function per frame (row) for whole dataset
	per_frame_z = lambda z: filter_outliers(z,neigh,method,factor)

	# Return filtered dataset, Z
	return np.array( list(pool.map(per_frame_z,Z)) )


def epochs_filter(epochs,xy,method_='median',factor_=10):
	# Filter an MNE epoch object spatially for outliers
	# epochs: MNE epochs object
	# xy: ChannelCoord Nx2
	# method_: string of median or mean
	# factor_: 10 (# std / mad away from mean = outlier)

	print("Mesh filtering epochs...")

	# Get data: trials, channels, time
	Z = epochs.get_data()
	# Get shape for later
	shp = Z.shape

	# Permute so channels is last dim, then flatten epochs and time
	Z = np.einsum('ijk->ikj',Z)
	Z = Z.reshape(-1,shp[1]) # channel shape

	Z = filter_dataset(Z,xy,method=method_,factor=factor_)

	# Reshape the matrix back together
	Z = Z.reshape(shp[0],shp[2],-1) # channel # arbritary
	Z = np.einsum('ikj->ijk',Z)

	# Add back to epochs
	return EpochsArray(Z, epochs.info, epochs.events, epochs.tmin, epochs.event_id)
