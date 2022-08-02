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



def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR.

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False

    return sc.flatten('F')


SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
SLP = SLPs['slp_mem']
GRD = SLPs['grd_mem']
SLPtime = SLPs['time']



SlpGrd = np.hstack((SLP,GRD))
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

# plotting the EOF patterns
plt.figure()
c1 = 0
c2 = 0
for hh in range(9):
    p1 = plt.subplot2grid((3,3),(c1,c2))
    spatialField = np.multiply(EOFs[hh,0:len(X_in)],np.sqrt(variance[hh]))
    Xs = np.arange(np.min(X_in),np.max(X_in),2)
    Ys = np.arange(np.min(Y_in),np.max(Y_in),2)
    lenXB = len(X_in)
    [XR,YR] = np.meshgrid(Xs,Ys)
    sea_nodes = []
    for qq in range(lenXB-1):
        sea_nodes.append(np.where((XR == X_in[qq]) & (YR == Y_in[qq])))

    rectField = np.ones((np.shape(XR))) * np.nan
    for tt in range(len(sea_nodes)):
        rectField[sea_nodes[tt]] = spatialField[tt]

    clevels = np.arange(-2,2,.05)
    m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
    #m.fillcontinents(color=dwtcolors[qq])
    cx,cy =m(XR,YR)
    m.drawcoastlines()
    CS = m.contourf(cx,cy,rectField,clevels,vmin=-1.2,vmax=1.2,cmap=cm.RdBu_r,shading='gouraud')
    p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
    c2 += 1
    if c2 == 3:
        c1 += 1
        c2 = 0


num_clusters = 49
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

# # sort kmeans
# kma_order = np.argsort(np.mean(-km, axis=1))

# sort kmeans
kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
for i in range(num_clusters):
    posc = np.where(kma.labels_ == kma_order[i])
    bmus_corrected[posc] = i

# reorder centroids
sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
sorted_centroids = centroids[kma_order, :]

# # reorder clusters: bmus, km, cenEOFs, centroids, group_size
# sorted_bmus = np.zeros((len(kma.labels_),), ) * np.nan
# for i in range(num_clusters):
#     posc = np.where(kma.labels_ == kma_order[i])
#     sorted_bmus[posc] = i
# sorted_km = km[kma_order]
# sorted_cenEOFs = kma.cluster_centers_[kma_order]
# sorted_centroids = centroids[kma_order]
# sorted_group_size = group_size[kma_order]

plt.figure()
#p1 = plt.subplot2grid((3, 3), (c1, c2))
spatialField = SLP[11443,:]/100-np.mean(SLP,axis=0)/100
Xs = np.arange(np.min(X_in), np.max(X_in), 2)
Ys = np.arange(np.min(Y_in), np.max(Y_in), 2)
lenXB = len(X_in)
[XR, YR] = np.meshgrid(Xs, Ys)
sea_nodes = []
for qq in range(lenXB - 1):
    sea_nodes.append(np.where((XR == X_in[qq]) & (YR == Y_in[qq])))

rectField = np.ones((np.shape(XR))) * np.nan
for tt in range(len(sea_nodes)):
    rectField[sea_nodes[tt]] = spatialField[tt]

clevels = np.arange(-17, 17, 0.5)
m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
# m.fillcontinents(color=dwtcolors[qq])
cx, cy = m(XR, YR)
m.drawcoastlines()
CS = m.contourf(cx, cy, rectField, clevels, vmin=-15, vmax=15, cmap=cm.RdBu_r, shading='gouraud')



repmatDesviacion = np.tile(SlpGrdStd, (49,1))
repmatMedia = np.tile(SlpGrdMean, (49,1))
Km_ = np.multiply(sorted_centroids,repmatDesviacion) + repmatMedia



[mK, nK] = np.shape(Km_)

Km_slp = Km_[:,0:int(nK/2)]
Km_grd = Km_[:,int(nK/2):]
X_B = X_in
Y_B = X_in
SLP_C = SLP
Km_slp = Km_slp[:,0:len(X_B)]
Km_grd = Km_grd[:,0:len(X_B)]

import matplotlib.cm as cm

etcolors = cm.rainbow(np.linspace(0, 1,49))
tccolors = np.flipud(cm.gray(np.linspace(0,1,7)))

dwtcolors = np.vstack((etcolors,tccolors[1:,:]))

SLPs2 = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021.mat')
# SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')
X_in2 = SLPs2['X_in']
Y_in2 = SLPs2['Y_in']

# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(7, 7)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(49):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh])

    if hh <= 49:
        num = kma_order[hh]

        # spatialField = Km_slp[(num), :] / 100 - np.nanmean(SLP_C, axis=0) / 100
        spatialField = Km_slp[(hh), :] / 100 - np.nanmean(SLP_C, axis=0) / 100

        #spatialField = np.multiply(EOFs[hh,0:len(X_in)],np.sqrt(variance[hh]))
        Xs = np.arange(np.min(X_in),np.max(X_in),2)
        Ys = np.arange(np.min(Y_in),np.max(Y_in),2)
        lenXB = len(X_in)
        [XR,YR] = np.meshgrid(Xs,Ys)
        sea_nodes = []
        for qq in range(lenXB-1):
            sea_nodes.append(np.where((XR == X_in[qq]) & (YR == Y_in[qq])))

        rectField = np.ones((np.shape(XR))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-27,27,1)
        m = Basemap(projection='merc',llcrnrlat=-32,urcrnrlat=48,llcrnrlon=275,urcrnrlon=370,lat_ts=10,resolution='c')

        # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
        m.fillcontinents(color=dwtcolors[hh])
        cx,cy =m(XR,YR)
        m.drawcoastlines()
        CS = m.contourf(cx,cy,rectField,clevels,vmin=-12,vmax=12,cmap=cm.RdBu_r,shading='gouraud')
        #p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
        tx,ty = m(320,-30)
        ax.text(tx,ty,'{}'.format(group_size[num]))

    else:
        num = hh


    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0

    if plotIndx < 8:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 8:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
