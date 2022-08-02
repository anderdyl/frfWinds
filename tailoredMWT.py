import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta

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


def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDatetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2]) for d in d_vec]


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)



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


def ClusterProbabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us,cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
       cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs



def axplot_WT_Probs(ax, wt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Blues', caxis='black'):
    'axes plot WT cluster probabilities'

    # clsuter transition plot
    pc = ax.pcolor(
        np.flipud(wt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # axis color
    plt.setp(ax.spines.values(), color=caxis)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )

    # axis linewidth
    if caxis != 'black':
        plt.setp(ax.spines.values(), linewidth=3)

    return pc



# loading in a North Atlantic continuous SLP record without any of the memory built into it
# SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_bigger.mat')
SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_ESTELA_area_smaller.mat')
# SLPs = ReadMatfile('/media/dylananderson/Elements1/NC_climate/NorthAtlanticSLPs_June2021_bigger_area_rect.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
SLP = SLPs['slp_mem']
SLPtime = SLPs['time']
sea = SLPs['sea_sq']
# M,N = np.shape(X_in)
# sea = np.arange(0,N*M)
XRsq = SLPs['XRsq']
YRsq = SLPs['YRsq']
# Applying PC to the those such that every day has a value in the first couple PCs..
SlpGrd = SLP
SlpMean = np.mean(SLP,axis=0)
SlpStd = np.std(SLP,axis=0)
SlpNorm = (SLP[:,:] - SlpMean) / SlpStd
SlpNorm[np.isnan(SlpNorm)] = 0

# principal components analysis
ipca = PCA(n_components=min(SlpNorm.shape[0], SlpNorm.shape[1]))
PCs = ipca.fit_transform(SlpNorm)
EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

# The first 4 explain 51.5% of the variance... good enough?

# Lets load in the DWT assignments and separate the PCs by their DWT number

with open(r"dwts49Clusters.pickle", "rb") as input_file:
# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
dwtBmus = historicalDWTs['bmus_corrected']

# with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
with open(r"dwtsOfExtraTropicalDays21Clusters.pickle", "rb") as input_file:
   historicalTWTs = pickle.load(input_file)
#timeTCs = historicalTWTs['tcDates']
twtBmus = historicalTWTs['bmus_corrected']
twtOrder = historicalTWTs['kma_order']
tcIndices = historicalTWTs['tcIndices']
TIMEtcs = historicalTWTs['TIMEtcs']


timeSLPs = dateDay2datetimeDate(timeDWTs)
timeTCs = dateDay2datetimeDate(TIMEtcs)

overlap = [x for x in timeSLPs if x in timeTCs]
ind_dict = dict((k,i) for i,k in enumerate(timeSLPs))
inter = set(timeSLPs).intersection(timeTCs)
indices = [ ind_dict[x] for x in inter ]
indices.sort()

mask = np.ones(len(timeDWTs), np.bool)
mask[indices] = 0
timeEWTs = timeDWTs[mask,:]

mask2 = np.zeros(len(timeDWTs), np.bool)
mask2[indices] = 1
timeTWTs = timeDWTs[mask2,:]

bmus = np.nan * np.ones((len(timeDWTs)))
bmus[mask] = dwtBmus+0
bmus[mask2] = twtBmus+49

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))


# Let's trim these things so they are the same length?
# Just need to remove January from SLPs...
# Or maybe take January-May away, and Feb-May away from the bmus?
# bmus_dates = dateDay2datetimeDatetime(timeDWTs)
# bmus_dates = bmus_dates[120:]


# bmus_dates = timeDWTs[120:,:]
# bmus = bmus[120:]
# SLPtime = SLPtime[151:]
# PCs = PCs[151:,:]



# bmus_dates = timeDWTs[28:,:]
# bmus = bmus[28:]
# SLPtime = SLPtime[59:]
# PCs = PCs[59:,:]
# bmus_dates_months = bmus_dates_months[28:]
# bmus_dates_days = bmus_dates_days[28:]
bmus_dates = timeDWTs
bmus = bmus
SLPtime = SLPtime[31:]
PCs = PCs[31:,:]
# bmus_dates_months = bmus_dates_months
# bmus_dates_days = bmus_dates_days


# SLPtime = SLPtime[120:]
# PCs = PCs[120:,:]

DailyPCs = PCs
DailyDatesMatrix = bmus_dates
DailySortedBmus = bmus
# DailyDatesMatrix = np.array([dt2cal(hh) for hh in bmus_dates])



#June/July/August
#Sept/Oct/November
#Dec/Jan/Feb
#Mar/Apr/May

# Monthly Time Series
dt = date(1979, 2, 1)
end = date(2021, 7, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
seasonalTime = []
while dt < end:
    seasonalTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step



s1season = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)

for i in range(len(seasonalTime)-1):
    sSeason = np.where((DailyDatesMatrix[:,0] == seasonalTime[i].year) & (DailyDatesMatrix[:,1] == seasonalTime[i].month) & (DailyDatesMatrix[:,2] == 1))
    if i == 507:
        ssSeason = np.where(
            (DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (
                        DailyDatesMatrix[:, 1] == (seasonalTime[i].month)) & (
                    DailyDatesMatrix[:, 2] == 30))
    else:
        if seasonalTime[i].month < 12:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month+1)) & (DailyDatesMatrix[:, 2] == 1))
        else:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == (seasonalTime[i].year+1)) & (DailyDatesMatrix[:, 1] == 2) & (DailyDatesMatrix[:, 2] == 1))

    for j in range(len(np.unique(DailySortedBmus))):
        s1season[j, i] = len(np.where(DailySortedBmus[sSeason[0][0]:ssSeason[0][-1]] == j)[0])



#
# s1 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
# # June/June #!!!!
# for i in range(len(np.unique(DailyDatesMatrix[:,0])) - 1):
#     s = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i]) & (DailyDatesMatrix[:,1] == 6))
#     ss = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i] + 1) & (DailyDatesMatrix[:,1] == 5))
#     for j in range(len(np.unique(DailySortedBmus))):
#         s1[j, i] = len(np.where(DailySortedBmus[s[0][0]:ss[0][-1]] == j)[0])



PC1 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)
PC2 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)
PC3 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)
PC4 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)
PC5 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)
PC6 = np.full([len(np.unique(DailySortedBmus)), len(seasonalTime)-1], np.nan)


# June/June #!!!!
for i in range(len(seasonalTime)-1):
    # sSeason = np.where((DailyDatesMatrix[:,0] == seasonalTime[i].year) & (DailyDatesMatrix[:,1] == seasonalTime[i].month) & (DailyDatesMatrix[:,2] == 1))
    # if i == 168:
    #     ssSeason = np.where(
    #         (DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (
    #                     DailyDatesMatrix[:, 1] == (seasonalTime[i].month + 2)) & (
    #                 DailyDatesMatrix[:, 2] == 30))
    # else:
    #     if seasonalTime[i].month < 10:
    #         ssSeason = np.where(
    #             (DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month+3)) & (
    #                         DailyDatesMatrix[:, 2] == 1))
    #     else:
    #         ssSeason = np.where(
    #             (DailyDatesMatrix[:, 0] == (seasonalTime[i].year+1)) & (DailyDatesMatrix[:, 1] == 3) & (
    #                         DailyDatesMatrix[:, 2] == 1))
    sSeason = np.where((DailyDatesMatrix[:,0] == seasonalTime[i].year) & (DailyDatesMatrix[:,1] == seasonalTime[i].month) & (DailyDatesMatrix[:,2] == 1))
    if i == 507:
        ssSeason = np.where(
            (DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (
                        DailyDatesMatrix[:, 1] == (seasonalTime[i].month)) & (
                    DailyDatesMatrix[:, 2] == 30))
    else:
        if seasonalTime[i].month < 12:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month+1)) & (DailyDatesMatrix[:, 2] == 1))
        else:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == (seasonalTime[i].year+1)) & (DailyDatesMatrix[:, 1] == 2) & (DailyDatesMatrix[:, 2] == 1))

    for j in range(len(np.unique(DailySortedBmus))):
        yearlyBmus = DailySortedBmus[sSeason[0][0]:ssSeason[0][-1]]     # lets get the bmus for this year
        yearlyPCs = DailyPCs[sSeason[0][0]:ssSeason[0][-1],:]           # lets get the PCs for this year
        indBmus = np.where(yearlyBmus == j)[0]         # which of those DWTs do we care about in this iteration
        # PC1[indDWT,i] = np.nanmean(DailyPCs[0,indBmus])
        # PC2[indDWT,i] = np.nanmean(DailyPCs[1,indBmus])
        # PC3[indDWT,i] = np.nanmean(DailyPCs[2,indBmus])
        PC1[j,i] = np.nansum(yearlyPCs[indBmus,0])
        PC2[j,i] = np.nansum(yearlyPCs[indBmus,1])
        PC3[j,i] = np.nansum(yearlyPCs[indBmus,2])
        PC4[j,i] = np.nansum(yearlyPCs[indBmus,3])
        PC5[j,i] = np.nansum(yearlyPCs[indBmus,4])
        PC6[j,i] = np.nansum(yearlyPCs[indBmus,5])


npercent = nPercent
tempPC1 = np.nansum(PC1,axis=0)
tempPC2 = np.nansum(PC2,axis=0)
tempPC3 = np.nansum(PC3,axis=0)
tempPC4 = np.nansum(PC4,axis=0)
tempPC5 = np.nansum(PC5,axis=0)
tempPC6 = np.nansum(PC6,axis=0)

normPC1 = np.divide(tempPC1,np.nanmax(tempPC1))*npercent[0]
normPC2 = np.divide(tempPC2,np.nanmax(tempPC2))*npercent[1]
normPC3 = np.divide(tempPC3,np.nanmax(tempPC3))*npercent[2]
normPC4 = np.divide(tempPC4,np.nanmax(tempPC4))*npercent[3]
normPC5 = np.divide(tempPC5,np.nanmax(tempPC5))*npercent[4]
normPC6 = np.divide(tempPC6,np.nanmax(tempPC6))*npercent[5]



n_components = 3 # !!!!

pcAggregates = np.full((len(normPC1),n_components),np.nan)
pcAggregates[:,0] = normPC1
pcAggregates[:,1] = normPC2
pcAggregates[:,2] = normPC3
# pcAggregates[:,3] = normPC4
# pcAggregates[:,4] = normPC5
# pcAggregates[:,5] = normPC6


n_clusters = 12

kmeans = KMeans(n_clusters, init='k-means++', random_state=100)  # 80

data = pcAggregates#[:, 0:n_components]

#    data1=data/np.std(data,axis=0)

awt_bmus = kmeans.fit_predict(data)

groups = [len(np.where(kk==awt_bmus)[0]) for kk in np.arange(0,n_clusters)]
print(groups)
seasonalMonth = np.array([i.month for i in seasonalTime])

monthNumber = [seasonalMonth[np.where(kk==awt_bmus)[0]] for kk in np.arange(0,n_clusters)]



fig = plt.figure(figsize=[14, 9])
gs2 = gridspec.GridSpec(n_components + 1, 1)
for nn in range(n_components):
    ax2 = fig.add_subplot(gs2[nn])
    # ax2.plot(np.unique(DailyDatesMatrix[:,0])[:-1], pcAggregates[:, nn], 'k.-', linewidth=1.8, markersize=8)
    ax2.plot(seasonalTime[0:-1], pcAggregates[:, nn], 'k.-', linewidth=1.8, markersize=8)

    ax2.set_ylabel('PC-' + str(nn + 1), fontsize=13)
    ax2.grid('minor')
    ax2.set_xticklabels([])

ax2 = fig.add_subplot(gs2[nn + 1])
# ax2.plot(np.unique(DailyDatesMatrix[:,0])[:-1], awt_bmus + 1, 'k.:', linewidth=1.8, markersize=10, color='grey')
ax2.plot(seasonalTime[0:-1], awt_bmus + 1, 'k.:', linewidth=1.8, markersize=10, color='grey')








dailyAWT = np.ones((len(DailySortedBmus),))
dailyPC1 = np.ones((len(DailySortedBmus),))
dailyPC2 = np.ones((len(DailySortedBmus),))
dailyPC3 = np.ones((len(DailySortedBmus),))
# dailyPC4 = np.ones((len(DailySortedBmus),))

for i in range(len(awt_bmus)):
    # sSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == seasonalTime[i].month) & (DailyDatesMatrix[:, 2] == 1))
    # if i == 168:
    #     ssSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month + 2)) & (DailyDatesMatrix[:, 2] == 31))
    # else:
    #     if seasonalTime[i].month < 10:
    #         ssSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month + 3)) & (DailyDatesMatrix[:, 2] == 1))
    #     else:
    #         ssSeason = np.where((DailyDatesMatrix[:, 0] == (seasonalTime[i].year + 1)) & (DailyDatesMatrix[:, 1] == 3) & (DailyDatesMatrix[:, 2] == 1))
    sSeason = np.where((DailyDatesMatrix[:,0] == seasonalTime[i].year) & (DailyDatesMatrix[:,1] == seasonalTime[i].month) & (DailyDatesMatrix[:,2] == 1))
    if i == 507:
        ssSeason = np.where(
            (DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (
                        DailyDatesMatrix[:, 1] == (seasonalTime[i].month)) & (
                    DailyDatesMatrix[:, 2] == 30))
    else:
        if seasonalTime[i].month < 12:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == seasonalTime[i].year) & (DailyDatesMatrix[:, 1] == (seasonalTime[i].month+1)) & (DailyDatesMatrix[:, 2] == 1))
        else:
            ssSeason = np.where((DailyDatesMatrix[:, 0] == (seasonalTime[i].year+1)) & (DailyDatesMatrix[:, 1] == 2) & (DailyDatesMatrix[:, 2] == 1))

    dailyAWT[sSeason[0][0]:ssSeason[0][0]+1] = awt_bmus[i]*dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]
    dailyPC1[sSeason[0][0]:ssSeason[0][0]+1] = normPC1[i]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    dailyPC2[sSeason[0][0]:ssSeason[0][0]+1] = normPC2[i]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    dailyPC3[sSeason[0][0]:ssSeason[0][0]+1] = normPC3[i]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    # dailyPC4[sSeason[0][0]:ssSeason[0][0]+1] = normPC4[i]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)



awtSLPs = SLP[0:,:]/100 - np.mean(SLP[0:,:],axis=0)/100
fig = plt.figure(figsize=(10, 6))

gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
for i in range(len(np.unique(awt_bmus))):

    m, n = np.shape(XRsq)
    ind = np.where((dailyAWT == i))
    avgSLP = np.mean(awtSLPs[ind[0],:],axis=0)
    wt = np.nan * np.ones((m*n,))
    for j in range(len(avgSLP)):
        wt[sea[j]-1] = avgSLP[j]
    spatialField = wt.reshape(n,m)

    ax = plt.subplot(gs1[i])
    # ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
    clevels = np.arange(-9, 9, 1)

    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10, resolution='c')
    m.fillcontinents(color=[0.5,0.5,0.5])
    cx, cy = m(XRsq, YRsq)
    m.drawcoastlines()
    # m.bluemarble()
    CS = m.contourf(cx, cy, spatialField.T, clevels, vmin=-9, vmax=9, cmap=cm.RdBu_r, shading='gouraud')
    tx, ty = m(320, -0)
    parallels = np.arange(0,360,10)
    # m.drawparallels(parallels,labels=[True,True,True,False],textcolor='white')
    m.drawparallels(parallels,labels=[False,False,False,False],textcolor='white')

    # ax.text(tx, ty, '{}'.format((group_size[num])))
    meridians = np.arange(0,360,20)
    # m.drawmeridians(meridians,labels=[True,True,True,True],textcolor='white')
    m.drawmeridians(meridians,labels=[False,False,False,False],textcolor='white')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.set_label('SLP (mbar)')
    #    cb = plt.colorbar(CS)
    cbar.set_clim(-9.0, 9.0)


asdfg

fig10 = plt.figure()
gs = gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.15)

for i in range(len(np.unique(awt_bmus))):
    ax = plt.subplot(gs[i])
    # select DWT bmus at current AWT indexes

    index_1 = np.where((dailyAWT == i))[0][:]
    sel_2 = DailySortedBmus[index_1]
    set_2 = np.arange(70)

    cps = ClusterProbabilities(sel_2, set_2)
    C_T = np.reshape(cps, (10, 7))

    # # axis colors
    # if wt_colors:
    #     caxis = cs_wt[ic]
    # else:
    caxis = 'black'

    # plot axes
    axplot_WT_Probs(
        ax, C_T,
        ttl='WT {0}'.format(i + 1),
        cmap='Reds', caxis=caxis,
    )
    ax.set_aspect('equal')




totalWTs = np.nan * np.ones((len(np.unique(awt_bmus)),))
for i in range(len(np.unique(awt_bmus))):
    # select DWT bmus at current AWT indexes
    index_1 = np.where((awt_bmus == i))[0][:]
    totalWTs[i] = len(index_1)



def ChangeProbabilities(series, set_values):
    'return series transition count and probabilities'

    # count matrix
    count = np.zeros((len(set_values), len(set_values)))
    for ix, c1 in enumerate(set_values):
        for iy, c2 in enumerate(set_values):

            # count cluster-next_cluster ocurrences
            us, cs = np.unique((series[:-1]==c1) & (series[1:]==c2), return_counts=True)
            d_count = dict(zip(us,cs))
            count[ix, iy] = d_count[True] if True in d_count.keys() else 0

    # probabilities
    probs = np.zeros((len(set_values), len(set_values)))
    for ix, _ in enumerate(set_values):

        # calculate each row probability
        probs[ix,:] = count[ix,:] / np.sum(count[ix, :])

    return count, probs


count, probs = ChangeProbabilities(awt_bmus,np.arange(0,8))

fig1 = plt.figure()
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
pc = ax.pcolor(np.flipud(probs),vmin=0,vmax=0.5, cmap=cm.Reds)
ax.set_xticks(np.arange(8) + 0.5)
ax.set_yticks(np.arange(8) + 0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
# add colorbar
cbar = plt.colorbar(pc, ax=ax)
cbar.ax.tick_params(labelsize=8)
# if vmin != 0 or vmax != 1:
#     cbar.set_ticks(np.linspace(vmin, vmax, 6))
cbar.ax.set_ylabel('transition probability', rotation=270, labelpad=20)


with open('/home/dylananderson/projects/duckGeomorph/NAO2021.txt', 'r') as fd:
    c = 0
    dataNAO = list()
    for line in fd:
        splitLine = line.split(',')
        secondSplit = splitLine[1].split('/')
        dataNAO.append(float(secondSplit[0]))
nao = np.asarray(dataNAO)

dt = date(1950, 1, 1)
end = date(2021, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
naoTime = []
while dt < end:
    naoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

naoSub = nao[350:]
naoTimeSub = naoTime[350:]


naoMWT = np.nan * np.ones((len(naoSub),))

for i in range(len(awt_bmus)):
    indMWT = np.where((awt_bmus == i))
    mwtTimes = np.asarray(seasonalTime)[indMWT]
    for j in mwtTimes:
        naoInd = np.where((j == np.asarray(naoTimeSub)))
        naoMWT[naoInd[0][0]:naoInd[0][0]+3] = np.ones((3,))*i

from scipy.optimize import curve_fit
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

plt.figure()
gs = gridspec.GridSpec(9, 1, wspace=0.1, hspace=0.15)
for i in awt_bmus:
    ax = plt.subplot(gs[i])
    index = np.where((naoMWT == i))[0][:]
    ax.hist(naoSub[index],15,color='black')
    #
    # bin_heights, bin_borders = np.histogram(naoSub[index], range=[-3.5,3.5], bins=20)
    # bin_widths = np.diff(bin_borders)
    # bin_centers = bin_borders[:-1] + bin_widths / 2
    # popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    #
    # x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 100)
    #
    # #plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
    # ax.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit', c='black')
    # # plt.legend()
    ax.set_xlim([-3.5,3.5])
    # ax.set_ylim([0,12])



wls = ReadMatfile('/home/dylananderson/projects/atlanticClimate/frfTideGaugeSerafin2.mat')

tide = wls['dailyData']['tide']
wl = wls['dailyData']['wl']
seasonal = wls['dailyData']['seasonal']
msl = wls['dailyData']['msl']
mmsla = wls['dailyData']['mmsla']
dsla = wls['dailyData']['dsla']
ss = wls['dailyData']['ss']
timeHourly = wls['dailyData']['hourlyDateVec']
timeMonthly = wls['dailyData']['monthDateVec']
mmslaMonth = wls['dailyData']['mmsla_month']
#
# MMSL = wls['MMSL']
# MMSLA = wls['MMSLA']
# MMSLA_hourly = wls['MMSLA_hourly']
# MSL = wls['MSL']
# DSLA = wls['DSLA']
# climatology = wls['climatology']
# climatologyDaily = wls['climatologyDaily']
# hourlyDateVec = wls['hourlyDateVec']
# monthDateVec = wls['monthDateVec']
# wl = wls['dat']
#
# mmslaChopped = MMSLA[2:]
#

def hourlyVec2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2], d[3]) for d in d_vec]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


timeMMSLA = hourlyVec2datetime(timeMonthly)


# timeMMSLAChopped = timeMMSLA[2:]
#
timeWL = hourlyVec2datetime(timeHourly)
#
# plt.figure()
# plt.plot(timeWL,climatologyDaily)
#
index = np.where((np.isnan(mmslaMonth)))
MMSLAcopy = mmslaMonth
MMSLAcopy[index] = 0 * np.ones(len(index[0]))
smooth = running_mean(MMSLAcopy,9)

#
plt.figure()
plt.plot(timeMMSLA,mmslaMonth)
plt.plot(timeMMSLA[4:-4],smooth)

#
# tFRF = hourlyVec2datetime(hourlyDateVec)
# resFRF = wl - MSL - MMSLA_hourly - climatologyDaily
#
# plt.figure()
# plt.plot(tFRF,resFRF)





# smslDate = wls['smsDate']
# smsl = wls['smsl']
# mmsl = wls['mmsl']
# timeSMSL = [datenum_to_datetime(int(x)) for x in wls['smsDate']]
# wlsDatevec = wls['dateVec']
# wlsDatetime = dateDay2datetimeDatetime(wlsDatevec)
# wlsLessSLR = wls['lessSLR']
# wlsResidual = wls['residual']
# wlsSLR = wls['slr']
# wlsPred = wls['predictedWL']
# wlsVerified = wls['verified']
# #python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
#
# plt.figure()
# ax = plt.subplot2grid((3,1),(0,0),rowspan=1,colspan=1)
# ax.plot(wlsDatetime,wlsVerified,label='water levvels')
# ax.plot(wlsDatetime,wlsSLR,label='SLR')
# ax.legend()
# ax2 = plt.subplot2grid((3,1),(1,0),rowspan=1,colspan=1)
# ax2.plot(wlsDatetime,wlsResidual)


plt.figure()
gs = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.15)
for i in awt_bmus:
    ax = plt.subplot(gs[i])
    index = np.where((naoMWT == i))[0][:]
    # bins_list = [-0.24,-0.21,-0.18,-0.15,-0.12,-0.09,-0.06,-0.03,0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24]
    bins_list = [-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18]

    ax.hist(smooth[index],bins=bins_list)
    #
    # bin_heights, bin_borders = np.histogram(mmslaChopped[index], range=[-0.25,0.25], bins=16)
    # bin_widths = np.diff(bin_borders)
    # bin_centers = bin_borders[:-1] + bin_widths / 2
    # popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    #
    # x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 16)
    #
    # #plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
    # ax.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='fit', c='black')
    # # # plt.legend()
    ax.set_xlim([-0.25,0.25])
    # ax.set_ylim([0,12])




asdfg

import pickle

mwtPickle = 'monthlywtPCs.pickle'
outputMWTs = {}
outputMWTs['PC1'] = normPC1
outputMWTs['PC2'] = normPC2
outputMWTs['PC3'] = normPC3
outputMWTs['PC4'] = normPC4
outputMWTs['mwt_bmus'] = awt_bmus
outputMWTs['seasonalTime'] = seasonalTime
outputMWTs['dailyMWT'] = dailyAWT
outputMWTs['dailyDates'] = bmus_dates
outputMWTs['dailyPC1'] = dailyPC1
outputMWTs['dailyPC2'] = dailyPC2
outputMWTs['dailyPC3'] = dailyPC3
# outputMWTs['dailyPC4'] = dailyPC4
outputMWTs['nPercent'] = nPercent


with open(mwtPickle,'wb') as f:
    pickle.dump(outputMWTs, f)

dailyMWT = dailyAWT
plt.figure()
plt.plot(seasonalTime,normPC1)
plt.plot([datetime(r[0],r[1],r[2]) for r in bmus_dates],dailyPC1)


bmus_months = np.array([r.month for r in seasonalTime])
summerInd = np.where(bmus_months == 6)
summerWTs = awt_bmus[summerInd]
fallInd = np.where(bmus_months == 9)
fallWTs = awt_bmus[fallInd]
winterInd = np.where(bmus_months == 12)
winterWTs = awt_bmus[winterInd]
springInd = np.where(bmus_months == 3)
springWTs = awt_bmus[springInd]








def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(days=365)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


def GenOneSeasonDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(3*365/12)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]



import matplotlib.pyplot as plt


import matplotlib.dates as mdates

from matplotlib import gridspec

# # Lets get complicated...
# # a grid, 8 x 4 for the 8 SWTs and the 4 seasons?
# # generate perpetual seasonal list
# fig = plt.figure()
# gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.15)
#
# monthsIni = [3,6,9,12]
# c = 0
# for m in monthsIni:
#
#     list_pSeason = GenOneSeasonDaily(month_ini=m)
#     m_plot = np.zeros((70, len(list_pSeason))) * np.nan
#     num_clusters=70
#     num_sim=1
#     # sort data
#     for i, dpy in enumerate(list_pSeason):
#         _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
#         #b = bmus[s,:]
#         b = bmus[s]
#         #b = b.flatten()
#
#         for j in range(num_clusters):
#             _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#
#             m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
#     ax = plt.subplot(gs[c])
#     # plot stacked bars
#     bottom_val = np.zeros(m_plot[1, :].shape)
#     for r in range(num_clusters):
#         row_val = m_plot[r, :]
#         ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#
#         # store bottom
#         bottom_val += row_val
#     # customize  axis
#     months = mdates.MonthLocator()
#     monthsFmt = mdates.DateFormatter('%b')
#     ax.set_xlim(list_pSeason[0], list_pSeason[-1])
#     ax.xaxis.set_major_locator(months)
#     ax.xaxis.set_major_formatter(monthsFmt)
#     ax.set_ylim(0, 100)
#     ax.set_ylabel('')
#     c = c + 1


dailyMWT = dailyMWT[0:-2]

# evbmus_sim = evbmus_sim - 1
# bmus = bmus + 1
fig = plt.figure()
gs = gridspec.GridSpec(9, 4, wspace=0.1, hspace=0.15)
c = 0
for awt in np.unique(awt_bmus):

    ind = np.where((dailyMWT == awt))[0][:]
    timeSubDays = bmus_dates_days[ind]
    timeSubMonths = bmus_dates_months[ind]
    #a = bmus[ind,:]
    a = bmus[ind]
    monthsIni = [3,6,9,12]
    for m in monthsIni:

        list_pSeason = GenOneSeasonDaily(month_ini=m)
        m_plot = np.zeros((70, len(list_pSeason))) * np.nan
        num_clusters=70
        num_sim=1
        # sort data
        for i, dpy in enumerate(list_pSeason):
            _, s = np.where([(timeSubMonths == dpy.month) & (timeSubDays == dpy.day)])
            #b = a[s,:]
            b = a[s]
            # b = bmus[s]
            #b = b.flatten()
            if len(b) > 0:
                for j in range(num_clusters):
                    _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
                    # _, bb = np.where([(j == b)])  # j starts at 0 bmus value!

                    m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)


        # indNan = np.where(np.isnan(m_plot))[0][:]
        # if len(indNan) > 0:
        #     m_plot[indNan] = np.ones((len(indNan),))
        #m_plot = m_plot[1:,:]
        ax = plt.subplot(gs[c])
        # plot stacked bars
        bottom_val = np.zeros(m_plot[1, :].shape)
        for r in range(num_clusters):
            row_val = m_plot[r, :]
            indNan = np.where(np.isnan(row_val))[0][:]
            if len(indNan) > 0:
                row_val[indNan] = 0
            ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))

            # store bottom
            bottom_val += row_val
        # customize  axis
        months = mdates.MonthLocator()
        monthsFmt = mdates.DateFormatter('%b')
        ax.set_xlim(list_pSeason[0], list_pSeason[-1])
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        #ax.set_ylim(0, 100)
        ax.set_ylabel('')
        c = c + 1






