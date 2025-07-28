### Visualize 2D Maps on a given date ###
#
# Original code: Michailis Mytilinaios
# Rev: 2.0 (Emilio Lapenna)
#

import os
import sys
import numpy as np
import argparse
from datetime import datetime
import netCDF4 as nc
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from matplotlib.colors import Normalize
#from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


# Set vars (file limits)
date_min = '2011-12-31'
date_max = '2017-01-01'

#data_path = '/mnt/data_vre/'
data_path = './data/'


def aemet_mm_scale_filter(x):
	#print(x)
	if x is None:
			x = -1
	elif x <= 0:
			x = -1
	elif 0.0 < x <= 0.1:
			x = 0 # np.where(B == N[i])
	elif 0.1 < x <= 0.2:
		x = 1
	elif 0.2 < x <= 0.4:
		x = 2
	elif 0.4 < x <= 0.8:
		x = 3
	elif 0.8 < x <= 1.2:
		x = 4
	elif 1.2 < x <= 1.6:
		x = 5
	elif 1.6 < x <= 3.2:
		x = 6
	elif 3.2 < x <= 6.4:
		x = 7
	else:
		x = 8
	return x


def normalize_colorbar(N):
	normN = np.nan * np.ones_like(N)
	#print(len(normN))
	for i in range(len(normN)):
		#print(N[i])
		tmp = N[i].tolist()
		normN[i] = list(map(aemet_mm_scale_filter, tmp))
	return normN


"""
def normalize_colorbar(B, N):
	normN = np.nan * np.ones(len(N))
	print(len(normN))
	for i in range(len(normN)):
		if np.any(N[i] == B):
			normN[i] = np.where(B == N[i])
		elif np.any(B > N[i]):
			normN[i] = np.where(B > N[i]) - 0.5
		else:
			normN[i] = 9.5
	return normN
"""

def NormalizeData(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))


#
def check_date(tardate):

	global date_min
	global date_max

	# Set date interval
	startdate = datetime.strptime(date_min, '%Y-%m-%d')
	enddate = datetime.strptime(date_max, '%Y-%m-%d')

	tardate = datetime.strptime(tardate, '%Y-%m-%d')

	#date_strt, date_end = datetime(2019, 3, 14), datetime(2020, 1, 4)
	if tardate <= startdate or tardate >= enddate:
		return False

	return True


#
def main(obs_date, map_region):

	global date_min
	global date_max
	global data_path

	if not check_date(obs_date):
		print('Date out of available data range : '+date_min+' / '+date_max+'!')
		sys.exit()

	# Convert date to string
	tmp = datetime.strptime(obs_date, '%Y-%m-%d')
	tardate = str( tmp.strftime('%Y%m%d') )


	# Set data file (path, name)
	wmo_file = data_path + tardate+'_3H_MEDIAN.nc'
	midas_file = data_path + 'MODIS-AQUA-C061_AOD-and-DOD-V1-GRID_RESOLUTION_0.1-'+tardate+'.nc'
	monarch_file = data_path + 'MO_od550du_'+tardate+'03_av_an.nc'
	#print(wmo_file)
	#print(midas_file)
	#print(monarch_file)


	# WMO file check
	if not os.path.exists(wmo_file):
		print('WMO file does not exist!')
		sys.exit()

	# MIDAS file check
	if not os.path.exists(midas_file):
		print('MIDAS file does not exist!')
		sys.exit()

	# MONARCH file check
	if not os.path.exists(monarch_file):
		print('MONARCH file does not exist!')
		sys.exit()



	# ******* FILE LOADING SECTION ******* #

	### WMO
	# longitude [degrees east]
	dataset = nc.Dataset(wmo_file)
	LON_W2 = dataset.variables['longitude'][:]
	LON_W1 = np.double(LON_W2)
	# latitude [degrees north]
	LAT_W2 = dataset.variables['latitude'][:]
	LAT_W1 = np.double(LAT_W2)
	# aerosol optical depth
	f_od550du = dataset.variables['OD550_DUST'][:]
	DODwmo = f_od550du[1,:,:]

	#print(DODwmo)

	### MIDAS
	# longitude [degrees east]
	dataset = nc.Dataset(str(midas_file))
	LON_M2 = dataset.variables['Longitude'][:]
	LON_M1 = np.unique(np.round(LON_M2, 1))
	# latitude [degrees north]
	LAT_M2 = dataset.variables['Latitude'][:]
	LAT_M1 = np.unique(np.round(LAT_M2, 1))
	# aerosol optical depth
	DODmidas = dataset.variables['Modis-total-dust-optical-depth-at-550nm'][:]

	#print(DODmidas)

	### MONARCH Reanalysis
	# longitude [degrees east]
	dataset = nc.Dataset(monarch_file)
	LONrea = dataset.variables['lon'][:]
	# latitude [degrees north]
	LATrea = dataset.variables['lat'][:]
	# aerosol optical depth
	f_od550du = dataset.variables['od550du'][:]
	f_od550du[f_od550du < 0] = np.nan
	DODrea = f_od550du[4,:,:]

	#print(DODrea)

	# ************************************ #


	# AEMET COLOURMAP RGB CODES
	rgbDUST = [
		(1.0000, 1.0000, 1.0000),
		(0.6314, 0.9294, 0.8902),
		(0.3608, 0.8902, 0.7294),
		(0.9882, 0.8431, 0.4588),
		(0.8549, 0.4471, 0.1882),
		(0.6196, 0.3843, 0.1490),
		(0.4431, 0.2863, 0.1294),
		(0.2235, 0.1451, 0.0667),
		(0.1137, 0.0745, 0.0353)
		]

	# Set colourbar non-linear scale
	B = [0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 3.2, 6.4]

	"""
	# Defining custom colormap
	for n_bin in B:
		#print(n_bin)
		cmap = LinearSegmentedColormap.from_list('meteoList', rgbDUST, N=n_bin)
	"""

	# Defining custom colormap
	cmap = LinearSegmentedColormap.from_list('meteoList', rgbDUST, 9)

	T = np.arange(0.1, (len(B) + 2) / 10)
	#limits = [T[0], T[-1]]
	#limits = [B[0], B[-1]]

	#print(T)
	#print(limits)

	if map_region == 'EuAfAm':
		# Set LAT / LON borders
		blatmin = -20 # 10
		blatmax =  60 # 90
		blonmin =   0 #  0
		blonmax =  70 # 60
	else:
		blatmin = -10 #
		blatmax =  10 #
		blonmin = -10 #
		blonmax =  10 #

	# Set plot
	fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()}) # 1, 3

	"""
	nrows = 0
	ncols = 0
	Z = np.arange(nrows * ncols).reshape(nrows, ncols)
	x = np.arange(ncols + 1)
	y = np.arange(nrows + 1)

	#XLON = [ -25.0, -13.0, -7.0, 5.0, 17.0, 29.0, 41.0 ]

	print('X: ',len(x))
	print('Y: ',len(y))

	print('LAT: ',len(LAT))
	print('LON: ',len(LON))
	print(Z.shape)
	"""


	# ******* WMO SDS-WAS MULTIMODEL ******* #
	#NZ = Normalize(vmin=0, vmax=9, clip=True)((DODwmo - B[0]) / (B[1] - B[0])) # <-- PROBLEM

	#NZ = normalize_colorbar(B, DODwmo)
	#NZ[np.isnan(DODwmo)] = np.nan

	#print('LAT: ',len(LAT_W1))
	#print('LON: ',len(LON_W1))

	# Reshape and normalization
	NZ = DODwmo.reshape(len(LAT_W1), len(LON_W1))
	NZ_norm = normalize_colorbar(NZ)
	#NZ_norm = np.interp(NZ, (NZ.min(), NZ.max()), (0, 9)) # Normalization between to numbers as max / min
	#print('WMO data shape: ', NZ_norm.shape)
	#sys.exit()

	c = axs[0].pcolormesh(LON_W1, LAT_W1, NZ_norm[:-1, :-1], cmap=cmap, vmin=0, vmax=9) #, cmap=plt.get_cmap('jet', 10))
	axs[0].coastlines()

	# Major ticks every 20, minor ticks every 5
	#major_ticks = np.arange(-20, 60, 20)
	#minor_ticks = np.arange(0, 70, 5)
	axs[0].set_xticks(np.arange(blatmin, blatmax, 20))
	#axs[0].set_xticks(minor_ticks, minor=True)
	axs[0].set_yticks(np.arange(blonmin, blonmax, 10))
	#axs[0].set_yticks(minor_ticks, minor=True)

	axs[0].set_title('WMO SDS-WAS MultiModel\n'+str(tmp)+' 12:00 UTC')
	axs[0].set_xlim([blatmin,blatmax])
	axs[0].set_ylim([blonmin,blonmax])
	# Add units to xytics
	axs[0].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))
	axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))

	# And a corresponding grid
	axs[0].grid(which='both')
	# Or if you want different settings for the grids:
	#axs[0].grid(which='minor', alpha=0.2)
	axs[0].grid(which='major', alpha=0.5)


	bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	fig.colorbar(
		c,
		ax = axs[0],
		norm = norm,
		ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		format = ticker.FixedFormatter(['0.0', '0.1', '0.2', '0.4', '0.8', '1.2', '1.6', '3.2', '6.4']),
		extend = 'max',
		label = ''
	) # spacing = 'uniform'



	# ******* MIDAS ******* #

	#print('LAT_M1: ',len(LAT_M1))
	#print('LON_M1: ',len(LON_M1))

	# Reshape and normalization
	NZ = DODmidas.reshape(len(LAT_M1), len(LON_M1))
	NZ_norm = normalize_colorbar(NZ)
	#print('MIDAS data shape: ', NZ_norm.shape)

	c = axs[1].pcolormesh(LON_M1, LAT_M1, NZ_norm[:-1, :-1], cmap=cmap, vmin=0, vmax=9) # plt.get_cmap('Reds', 10)
	axs[1].coastlines()

	# Major ticks every 20, minor ticks every 10
	axs[1].set_xticks(np.arange(blatmin, blatmax, 20))
	axs[1].set_yticks(np.arange(blonmin, blonmax, 10))

	# Set title and limits
	axs[1].set_title('MIDAS\n'+str(tmp)+' 12:00 UTC')
	axs[1].set_xlim([blatmin,blatmax])
	axs[1].set_ylim([blonmin,blonmax])
	# Add units to xytics
	axs[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))
	axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))

	# And a corresponding grid
	axs[1].grid(which='both')
	# Or if you want different settings
	axs[1].grid(which='major', alpha=0.5)

	bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	fig.colorbar(
		c,
		ax = axs[1],
		norm = norm,
		ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		format = ticker.FixedFormatter(['0.0', '0.1', '0.2', '0.4', '0.8', '1.2', '1.6', '3.2', '6.4']),
		extend = 'max',
		label = ''
	) # spacing = 'uniform'



	# ******* MONARCH Reanalysis ******* #

	#print('LATrea: ',len(LATrea))
	#print('LONrea: ',len(LONrea))

	# Reshape and normalization
	NZ = DODrea.reshape(len(LATrea), len(LONrea))
	NZ_norm = normalize_colorbar(NZ)
	#print('MONARCH data shape: ', NZ_norm.shape)

	c = axs[2].pcolormesh(LONrea, LATrea, NZ_norm[:-1, :-1], cmap=cmap, vmin=0, vmax=9) # plt.get_cmap('Reds', 10)
	axs[2].coastlines()

	# Major ticks every 20, minor ticks every 10
	axs[2].set_xticks(np.arange(blatmin, blatmax, 20))
	axs[2].set_yticks(np.arange(blonmin, blonmax, 10))

	# Set tile and limits
	axs[2].set_title('MONARCH Reanalysis\n'+str(tmp)+' 12:00 UTC')
	axs[2].set_xlim([blatmin,blatmax])
	axs[2].set_ylim([blonmin,blonmax])
	# Add units to xytics
	axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))
	axs[2].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))

	# And a corresponding grid
	axs[2].grid(which='both')
	# Or if you want different settings
	axs[2].grid(which='major', alpha=0.5)

	bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	fig.colorbar(
		c,
		ax = axs[2],
		norm = norm,
		ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		format = ticker.FixedFormatter(['0.0', '0.1', '0.2', '0.4', '0.8', '1.2', '1.6', '3.2', '6.4']),
		extend = 'max',
		label = ''
	) # spacing = 'uniform'


	plt.tight_layout()

	#plt.show()

	plt.savefig(map_region+'_'+tardate+'.png')


#
def parse_args():

	"""
	Parses command-line arguments for...

	Returns:
		argparse.Namespace: Parsed command-line arguments
	"""
	parser = argparse.ArgumentParser(description="""
	Download data from AERONET data service and produce DOD.
	Arguments:\n
		obs_site (str): Observation site to analyse.\n
		obs_date (str): Start date in format YYYY-MM-DD.\n

	Example usage:\n
		python3 2dm.py --obs_date 2014-05-30 --map_region EuAfAm
	""")

	parser.add_argument('--obs_date', type=str, help='Observation date', required=True)
	parser.add_argument('--map_region', type=str, help='Map region', required=True)

	#parser.add_argument('--obs_date_min', type=str, help='Start date', required=True)
	#parser.add_argument('--obs_date_max', type=str, help='End date', required=True)

	parser.add_argument('--basepath', type=str, help='Base output path', default='.') # implicit
	#parser.add_argument('--variab_conf', type=str, help='Path to variable config file', default=variab_conf)

	if len(sys.argv) == 1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	return parser.parse_args()

#
if __name__ == '__main__':

	#
	args = parse_args()

	# Observational site
	#obs_site = 'IMAA_Potenza' # IMAA_Potenza
	# Formato di input: Y,M,D,H,MI,S (MI & S sempre 0)

	print('')
	print('************************')
	print('*** Dust map routine ***')
	print('************************')

	print('Observation date: '+args.obs_date)
	print('Map region: '+args.map_region)

	#print('Target dates: '+args.obs_date_min+' -> '+args.obs_date_max)

	"""
	# Check implicit basepath
	if args.basepath:
		data_path = os.path.join(args.basepath, data_path)
		#graph_path = os.path.join(args.basepath, graph_path)

	# Check working directory, if any
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	if not os.path.exists(graph_path):
		os.makedirs(graph_path)
	"""

	"""
	# Check if site present
	if not check_site(args.obs_site):
		print('ERROR: Observation site does not exist or is not allowed!')
		sys.exit(1)
	"""

	# Call main
	main(args.obs_date, args.map_region)


#
