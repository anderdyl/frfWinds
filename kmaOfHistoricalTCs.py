import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import pandas as pd
import os
import datetime
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mda
from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
from dateutil.relativedelta import relativedelta
import random
import xarray as xr
import geopy.distance
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDate(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.date(d[0], d[1], d[2]) for d in d_vec]


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out




SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_smaller.mat')
# SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
SLP = SLPs['slp_mem']
SLPtime = SLPs['time']



import pickle

with open(r"historicalTCs.pickle", "rb") as input_file:
   historicalTCs = pickle.load(input_file)

c1times = historicalTCs['c1times']
c2times = historicalTCs['c2times']
c3times = historicalTCs['c3times']
c4times = historicalTCs['c4times']
c5times = historicalTCs['c5times']
c6times = historicalTCs['c6times']

cluster6SLPs = historicalTCs['cluster6SLPs']
cluster5SLPs = historicalTCs['cluster5SLPs']
cluster4SLPs = historicalTCs['cluster4SLPs']
cluster3SLPs = historicalTCs['cluster3SLPs']
cluster2SLPs = historicalTCs['cluster2SLPs']
cluster1SLPs = historicalTCs['cluster1SLPs']

allTCtimes1 = np.vstack((c1times,c2times))
allTCtimes2 = np.vstack((allTCtimes1,c3times))
allTCtimes3 = np.vstack((allTCtimes2,c4times))
allTCtimes4 = np.vstack((allTCtimes3,c5times))
allTCtimes5 = np.vstack((allTCtimes4,c6times))

allTCtimes = np.asarray(dateDay2datetime(allTCtimes5))

df = pd.DataFrame(allTCtimes,columns=['date'])
dropDups = df.drop_duplicates('date')

tcDates = dropDups['date'].dt.date.tolist()
slpDates = dateDay2datetimeDate(SLPtime)
overlap = [x for x in slpDates if x in tcDates]

ind_dict = dict((k,i) for i,k in enumerate(slpDates))
inter = set(slpDates).intersection(tcDates)
indices = [ ind_dict[x] for x in inter ]

# ugh this was stupid, this rogue TC should be handled differently
# these are 4 days AFTER TCs have been removed that need to be added back.
# and they are indices in a time series with no January 1979.
# SLPless = np.delete(SLPless,(11440,11441,11442,11443),axis=0)

# which means they need to be ADDED to the "indices" vector here
# but need to have 31 added for the length of January?

indices.append(int(11440+31))
indices.append(int(11441+31))
indices.append(int(11442+31))
indices.append(int(11443+31))

# mask = np.ones(len(SLP), np.bool)
# mask[indices] = 0
# SLPless = SLP[mask,:]

indices.sort()

SLPtcs = SLP[indices,:]
TIMEtcs = SLPtime[indices,:]



SlpGrd = SLPtcs#np.hstack((SLPless,GRDless))
SlpGrdMean = np.mean(SlpGrd,axis=0)
SlpGrdStd = np.std(SlpGrd,axis=0)
SlpGrdNorm = (SlpGrd[:,:] - SlpGrdMean) / SlpGrdStd
SlpGrdNorm[np.isnan(SlpGrdNorm)] = 0

# principal components analysis
ipca = PCA(n_components=min(SlpGrdNorm.shape[0], SlpGrdNorm.shape[1]))
PCs = ipca.fit_transform(SlpGrdNorm)
EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= 0.95 * 100)[0][-1]


num_clusters = 21
PCsub = PCs[:, :nterm+1]
EOFsub = EOFs[:nterm+1, :]
kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)
# groupsize
_, group_size = np.unique(kma.labels_, return_counts=True)
# groups
d_groups = {}
for k in range(num_clusters):
    d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
# centroids
centroids = np.dot(kma.cluster_centers_, EOFsub)
# km, x and var_centers
km = np.multiply(
    centroids,
    np.tile(SlpGrdStd, (num_clusters, 1))
) + np.tile(SlpGrdMean, (num_clusters, 1))



#order = [10,5,0,4,2,9,6,7,1,11,8,3]
#kma_order = [0,1,2,3,4,5,6,7,8,9,10,11]
#kma_order = [7,11,10,6,8,4,3,2,5,9,0,1]
#kma_order = [9,8,2,1,7,4,10,11,3,6,0,5]
# # sort kmeans
#kma_order = np.argsort(np.mean(-km, axis=1))


# kma_order = [16,15,8,12,1,0,10,
#              9,17,14,7,13,19,5,
#              20,4,6,3,18,11,2]
kma_order = [15,14,7,11,0,20,9,
             8,16,13,6,12,18,4,
             19,3,5,2,17,10,1]


# kma_order = np.arange(0,21)
# # sort kmeans
#kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
for i in range(num_clusters):
    posc = np.where(kma.labels_ == kma_order[i])
    bmus_corrected[posc] = i

# reorder centroids
sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
sorted_centroids = centroids[kma_order, :]




repmatDesviacion = np.tile(SlpGrdStd, (num_clusters,1))
repmatMedia = np.tile(SlpGrdMean, (num_clusters,1))
Km_slp = np.multiply(centroids,repmatDesviacion) + repmatMedia
Xs = np.arange(np.min(X_in),np.max(X_in),2)
Ys = np.arange(np.min(Y_in),np.max(Y_in),2)
lenXB = len(X_in)
[XR,YR] = np.meshgrid(Xs,Ys)
sea_nodes = []
for qq in range(lenXB-1):
    sea_nodes.append(np.where((XR == X_in[qq]) & (YR == Y_in[qq])))



# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,5))
gs1 = gridspec.GridSpec(3, 7)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(num_clusters):
    ax = plt.subplot(gs1[hh])
    clevels = np.arange(-27, 27, 1)
    num = kma_order[hh]
    spatialField = Km_slp[(num), :] / 100 - np.nanmean(SLPtcs, axis=0) / 100
    rectField = spatialField.reshape(63, 32)

   # cluster2SLPs = np.nanmean(SLP[cluster2SLPIndex, :], axis=0).reshape(73, 43) / 100 - np.nanmean(SLP, axis=0).reshape(73, 43) / 100

    # rectField = np.ones((np.shape(X_in))) * np.nan
    # # temp = np.nanmean(SLP[cluster6SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
    # #temp = spatialField.flatten()
    # for tt in range(len(sea_nodes)):
    #     rectField[sea_nodes[tt]] = spatialField[tt]
    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10, resolution='c')
    #m.fillcontinents(color=dwtcolors[36])
    cx, cy = m(X_in, Y_in)
    m.drawcoastlines()
    CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-12, vmax=12, cmap=cm.RdBu_r, shading='gouraud')
    tx, ty = m(320, -0)
    ax.text(tx, ty, '{}'.format((group_size[num])))









import pickle

dwtPickle = 'dwtsOfExtraTropicalDays21Clusters.pickle'
outputDWTs = {}
outputDWTs['SLPtcs'] = SLPtcs
outputDWTs['TIMEtcs'] = TIMEtcs
outputDWTs['APEV'] = APEV
outputDWTs['EOFs'] = EOFs
outputDWTs['EOFsub'] = EOFsub
#outputDWTs['GRD'] = GRD
#outputDWTs['GRDless'] = GRDless
#outputDWTs['Km_'] = Km_
#outputDWTs['Km_grd'] = Km_grd
outputDWTs['Km_slp'] = Km_slp
outputDWTs['PCA'] = PCA
outputDWTs['PCs'] = PCs
outputDWTs['PCsub'] = PCsub
outputDWTs['SLP'] = SLP
#outputDWTs['SLP_C'] = SLP_C
#outputDWTs['SLPless'] = SLPless
##outputDWTs['SLPs'] = SLPs
outputDWTs['SLPtime'] = SLPtime
outputDWTs['SlpGrd'] = SlpGrd
outputDWTs['SlpGrdMean'] = SlpGrdMean
outputDWTs['SlpGrdStd'] = SlpGrdStd
#outputDWTs['Timeless'] = Timeless
outputDWTs['XR'] = XR
#outputDWTs['X_B'] = X_B
outputDWTs['X_in'] = X_in
outputDWTs['XS'] = Xs
outputDWTs['YR'] = YR
#outputDWTs['Y_B'] = Y_B
outputDWTs['Y_in'] = Y_in
outputDWTs['YS'] = Ys
outputDWTs['allTCtimes'] = allTCtimes
outputDWTs['bmus_corrected'] = bmus_corrected
outputDWTs['centroids'] = centroids
outputDWTs['d_groups'] = d_groups
outputDWTs['group_size'] = group_size
outputDWTs['ipca'] = ipca
outputDWTs['km'] = km
outputDWTs['kma'] = kma
outputDWTs['kma_order'] = kma_order
#outputDWTs['mask'] = mask
outputDWTs['nPercent'] = nPercent
outputDWTs['nterm'] = nterm
outputDWTs['num_clusters'] = num_clusters
outputDWTs['sea_nodes'] = sea_nodes
outputDWTs['slpDates'] = slpDates
outputDWTs['sorted_cenEOFs'] = sorted_cenEOFs
outputDWTs['sorted_centroids'] = sorted_centroids
outputDWTs['tcDates'] = tcDates
outputDWTs['variance'] = variance
outputDWTs['tcIndices'] = indices

with open(dwtPickle,'wb') as f:
    pickle.dump(outputDWTs, f)










