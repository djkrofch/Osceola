# Load required libraries and establish data directories

# ------- Notebook config
import matplotlib.colors
import matplotlib.pyplot as plt

# ------- Load dependencies
import pandas as pd
import numpy as np
import random
import seaborn as sns
import gdal, os, osr, warnings

# Curve fitting and linear models
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy # R-esque version of formula declaration

# ------- Working directory
dataDir = 'Y:/DanK/DinkeyCreek/ProjectedClimate_MS/'

# ---- Map dependencies
import fiona
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from descartes import PolygonPatch
from itertools import chain
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection

# ---- Function defs
def centOutputGen(dataDir, simname):
    wkDir_reps =  dataDir + simname + '/'
    centFileName = 'NECN-succession-log.csv'
    centFileNameM = 'NECN-succession-monthly-log.csv'

    repnames = next(os.walk(wkDir_reps))[1]
    num_replicates = len(repnames)

    idx = 0
    for rep in repnames:
        repnum = rep.split('replicate')[1]
        wkDir_data = wkDir_reps + 'replicate' + repnum + '/'
        cent = pd.read_csv(wkDir_data + centFileName)
        cent['rep'] = int(repnum)
        centM = pd.read_csv(wkDir_data + centFileNameM)
        centM['rep'] = int(repnum)

        if idx == 0: 
            centTot = cent
            centTotM = centM

        else:
            centTot = pd.concat((centTot, cent))
            centTotM = pd.concat((centTotM, centM))

        idx = idx + 1
    return centTot, centTotM

def genCentWeightedMeansDF(aggregatedAnnualCentFile, simName):
    simCentDF = aggregatedAnnualCentFile
    # ------- Create new vars in Cent file and handle unit conversion ------- #
    simCentDF['TEC'] = (simCentDF['SOMTC'] + simCentDF['AGB']* 0.5 ) * 0.01 # Add total ecosystem carbon, Mg/ha
    simCentDF['AGBc'] = simCentDF['AGB']* 0.5 * 0.01 # Convert AGB from g /m^2 biomass to Mg/ha C
    simCentDF['NEEC'] = simCentDF['NEEC'] * -1 * 0.01 # Convert NEE sign convention 
    simCentDF['NECB'] = simCentDF['NEEC'] - (simCentDF['FireCEfflux'] * 0.01)
    activeArea = simCentDF.NumSites.unique().sum() # Calculate total sites in the simulation

    # ------- Generate mean and std of weighted ecoregion data ------- #
    simCentDF['TECw'] = simCentDF.TEC * (simCentDF.NumSites / activeArea)
    simCentDF['AGBw'] = simCentDF.AGBc * (simCentDF.NumSites / activeArea)
    simCentDF['NEECw'] = simCentDF.NEEC * (simCentDF.NumSites / activeArea)
    
    # Convert the fire efflux from g/m2 to Mg/ha (already in units of C), and 
    # produce the ecoregion weighted per hectare value
    simCentDF['FireCEffluxw'] = simCentDF.FireCEfflux * 0.01 * (simCentDF.NumSites / activeArea)
    simCentDF['NECBw'] = simCentDF['NEECw'] - simCentDF['FireCEffluxw']
    
    # Take mean and std across reps
    naiveMean = simCentDF.groupby(['EcoregionName','Time']).mean()
    naiveStd = simCentDF.groupby(['EcoregionName','Time']).std()

    # Generate a dataframe for that we can leverage when we want to ask questions
    # about the cumulative nature of some century variable, by replicate.
    naiveSum= simCentDF.groupby(['rep']).sum()

    naiveMean['NEECw_std'] = naiveStd.NEEC * (naiveMean.NumSites / activeArea) # 
    naiveMean['AGBw_std'] = naiveStd.AGBc * (naiveMean.NumSites / activeArea)
    naiveMean['TECw_std'] = naiveStd.TEC * (naiveMean.NumSites / activeArea)
    naiveMean['FireCEffluxw_std'] = naiveStd.FireCEfflux * (naiveMean.NumSites / activeArea)
    naiveMean['NECBw_std'] = naiveStd.NECBw * (naiveMean.NumSites / activeArea)
    resetMean = naiveMean.reset_index()
    resetSum = naiveSum.reset_index()
    repsTot = simCentDF.reset_index()
    
    # ------- Generate region-wide weighted mean of all ecoregions ------ #
    weightedMean = resetMean.groupby('Time').sum()
    index = pd.date_range('2000-1-1', periods=len(weightedMean), freq='1A')
    weightedMean.index = index
    weightedMean[weightedMean.NEECw == 0] = np.nan
    weightedMean['Sim'] = simName
    resetSum['Sim'] = simName
    simCentDF['Sim'] = simName

    return weightedMean, simCentDF, resetSum

def plotSimulationCarbon(weightedMeansDF):
    weightedMean = weightedMeansDF
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize =(16,10))
    plt.subplots_adjust(wspace = 0.3)
    simidx = 0
    simColors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c', 'black']
    for sim in np.unique(weightedMean.Sim):
        print sim
        simMean = weightedMean[weightedMean.Sim == sim]
        simMean.AGBw.plot(ax = ax1, color = simColors[simidx])
        simMean.TECw.plot(ax = ax2, color = simColors[simidx])
        simMean.NEECw.plot(ax = ax3, color = simColors[simidx])
        simMean.FireCEffluxw.cumsum().plot(ax = ax4, color = simColors[simidx])
        simMean.NECBwHA = simMean.NECBw * 0.01 # Mg/ha for cumulative curves
        simMean.NECBwHA_stdHA = simMean.NECBw_std * 0.01 # Mg/ha for cumulative curves
        simMean.NECBwHA.plot(ax = ax6, color = simColors[simidx])
        

        ax1.fill_between(simMean.index, 
                         simMean.AGBw+simMean.AGBw_std, 
                         simMean.AGBw-simMean.AGBw_std, 
                         alpha = 0.2, color = simColors[simidx])
 
        ax2.fill_between(simMean.index, 
                         simMean.TECw+simMean.TECw_std, 
                         simMean.TECw-simMean.TECw_std, 
                         alpha = 0.2, color = simColors[simidx])
        
        ax3.fill_between(simMean.index, 
                         simMean.NEECw+simMean.NEECw_std, 
                         simMean.NEECw-simMean.NEECw_std, 
                         alpha = 0.2, color = simColors[simidx])
        
        #Replaced the cumulatiev curves with boxplots of cumulative C emissions from fire
        ax4.fill_between(simMean.index, 
                         simMean.FireCEffluxw.cumsum()+simMean.FireCEffluxw_std, 0,
                         alpha = 0.2, color = simColors[simidx])
        
        ax6.fill_between(simMean.index, 
                         simMean.NECBwHA+simMean.NECBwHA_stdHA, 
                         simMean.NECBwHA-simMean.NECBwHA_stdHA, 
                         alpha = 0.2, color = simColors[simidx])
        
        simidx = simidx + 1

    # ------ Aesthetics ------ #
    ax1.set_ylabel('AGB (MgC ha $^{-1}$)')
    ax2.set_ylabel('TEC (Mg ha $^{-1}$)')
    ax3.set_ylabel('NEE (gC m$^{-2}$)')
    ax4.set_ylabel('Cumulative Fire Efflux gC m$^{-2}$)')
    ax5.set_ylabel('AGB (MgC ha $^{-1}$)')
    ax6.set_ylabel('Cumulative NECB MgC ha$^{-1}$)')
    
    ax2.set_xlabel('Simulation Year')
    ax5.set_xlabel('Model Simulation')
    
    lastTen = weightedMean[weightedMean.index.year >= 2090]
    #g = sns.boxplot(x="Sim", y=weightedMean.FireCEffluxw,
    #              data=weightedMean, ax = ax4, palette=simColors)
    ax4.set_xticklabels(lastTen.Sim.unique(),rotation = 45)
    #ax4.set_ylim([0,1000])

    g = sns.boxplot(x="Sim", y="AGBw",
                  data=lastTen, ax = ax5, palette=simColors)
    ax5.set_xticklabels(lastTen.Sim.unique(),rotation = 45)
    
    sns.despine()
    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.5)

    ax1.legend(np.unique(weightedMean.Sim), loc = 'lower right')
    
def severityStack(rasterLoc, runLength):
    rasterlist = []
    timesteps = np.linspace(1,runLength,runLength)
    
    repnames = next(os.walk(rasterLoc))[1]
    num_replicates = len(repnames)

    idx = 0
    for rep in repnames:
        counter = 0
        repnum = rep.split('replicate')[1]
        for time in timesteps:  
            wkDir_data = rasterLoc + '/' 'replicate' + repnum + '/' + 'fire/'
            sevmap = 'severity-' + str(int(time)) + '.img'
            src_ds = gdal.Open( wkDir_data + sevmap ) 
            sevarray = src_ds.ReadAsArray()
            sevarray = sevarray.astype('float')
            rasterlist.append(sevarray)
            
    return rasterlist

def thinStack(rasterLoc, runLength):
    rasterlist = []
    thinlist = []
    rxfirelist = []
    timesteps = np.linspace(1,runLength,runLength)
    
    repnames = next(os.walk(rasterLoc))[1]
    num_replicates = len(repnames)

    idx = 0
    for rep in repnames:
        counter = 0
        repnum = rep.split('replicate')[1]
        for time in timesteps:  
            wkDir_data = rasterLoc + '/' 'replicate' + repnum + '/ThinMAP/'
            thinmap = 'biomass-removed-' + str(int(time)) + '.img'
            src_ds = gdal.Open( wkDir_data + thinmap ) 
            thinarray = src_ds.ReadAsArray()
            thinarray = thinarray.astype('float')
            rasterlist.append(thinarray)
            if time < 10:
                thinlist.append(thinarray)
            else:
                rxfirelist.append(thinarray)
            
    return rasterlist, thinlist, rxfirelist


def severityGen(rasterList):
    rasterStack = np.dstack(rasterList)  
    rasterAdj = rasterStack - 2
    rasterAdj[rasterAdj < 0] = np.nan
    meanSev = np.nanmean(rasterAdj, axis = 2)
    varSev = np.nanvar(rasterAdj, axis = 2)
    return meanSev, varSev

def thinGen(rasterList):
    rasterStack = np.dstack(rasterList)
    countTHIN = np.count_nonzero(rasterStack, axis = 2)
    rasterStack[rasterStack == 0] = np.nan
    cumTHIN = np.nansum(rasterStack, axis = 2)
    varTHIN = np.nanvar(rasterStack, axis = 2)
    return countTHIN, cumTHIN, varTHIN

def plotTS(simsDF, simname, var, var_sd = None, ax = None, *args, **kwargs):
    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()
                   
    simMean = simsDF[simsDF.Sim == simname]
    simMean[var].plot(ax = ax, *args, **kwargs)
    
    if var_sd != None:
        ax.fill_between(simMean.index, 
                    simMean[var]+3*simMean[var_sd]/np.sqrt(50), 
                    simMean[var]-3*simMean[var_sd]/np.sqrt(50), 
                    alpha = 0.3)
        
def plotTS_cumulative(simsDF, simname, var, var_sd = None, ax = None, *args, **kwargs):
    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()
            
    simMean = simsDF[simsDF.Sim == simname]            
    simMean[var].cumsum().plot(ax = ax, *args, **kwargs)
    
    if var_sd != None:                 
        ax.fill_between(simMean.index, 
                 simMean[var].cumsum()+3*simMean[var_sd].cumsum()/np.sqrt(50), 
                 simMean[var].cumsum()-3*simMean[var_sd].cumsum()/np.sqrt(50), 
                 alpha = 0.3)

def importRaster(rasterPath, **kwargs):
    # Open and read in the raster as an array
    raster_ds = gdal.Open(rasterPath)
    rastermap = raster_ds.ReadAsArray()
    
    # Set the default data type to 'float'
    if 'dtype' not in kwargs:
        dtype = 'float'
    rastermap = rastermap.astype(dtype)
    
    # If specified, set the no data value to NaN
    if 'noData' in kwargs:
        rastermap[rastermap == noData] = np.nan
    return rastermap
        
def plotRaster(image, ax=None, *args, **kwargs):

    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()
                   
    # Normalize color scheme
    if 'norm' not in kwargs:
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if vmin is None:
            vmin = np.min(image) # or what ever
        if vmax is None:
            vmax = np.max(image)
        norm = matplotlib.colors.Normalize(vmin, vmax)
        kwargs['norm'] = norm

    #ax.figure.canvas.draw() # if you want to force a re-draw
    ax.imshow(image, *args, **kwargs)
    # Setup axes
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    
# Stacked bar chart code grabbed then modified from the web
def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=1.2,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, edgecolor = 'k', color = 'white', hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5], title = 'Fire Severity')
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1], title = 'Fire Weather') 
    axe.add_artist(l1)
    return axe

def appendTreatments(totalDFs):
    warnings.filterwarnings("ignore")
    totalDFs['Trtmnt'] = 'No Management'
    totalDFs.Trtmnt[totalDFs.Sim == 'HSFPCCSM'] = 'HSFP'
    totalDFs.Trtmnt[totalDFs.Sim == 'HSFPCNRM'] = 'HSFP'
    totalDFs.Trtmnt[totalDFs.Sim == 'HSFPFGOALS'] = 'HSFP'
    totalDFs.Trtmnt[totalDFs.Sim == 'HSFPMIROC5'] = 'HSFP'

    totalDFs.Trtmnt[totalDFs.Sim == 'OpConCCSM'] = 'OpCon'
    totalDFs.Trtmnt[totalDFs.Sim == 'OpConCNRM'] = 'OpCon'
    totalDFs.Trtmnt[totalDFs.Sim == 'OpConFGOALS'] = 'OpCon'
    totalDFs.Trtmnt[totalDFs.Sim == 'OpConMIROC5'] = 'OpCon'
    warnings.filterwarnings("default")

    return totalDFs

def plotMinMaxMedReps(simCentDF, var, ax = None, *args, **kwargs):
    # Generate ecoregion weighted sum    
    summed = simCentDF.groupby(['Time','rep','Sim']).sum()

    # Reset the index for grouping
    reset_summed = summed.reset_index()

    # Create the min, max, and median DFs
    maxdf = reset_summed.groupby('Time').max()
    mindf = reset_summed.groupby('Time').min()
    meddf = reset_summed.groupby('Time').median()

    # Add the three plots to a single set of axes
    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()

    maxdf[var].plot(ax = ax, *args, **kwargs)
    mindf[var].plot(ax = ax, *args, **kwargs)
    meddf[var].plot(ax = ax, *args, **kwargs)

    
def plotAllSimsMinMaxMed(simCentDF_Stack, var, treatment, ax = None, *args, **kwargs):
    # Generate ecoregion weighted sum    
    summed = simCentDF_Stack.groupby(['Time','rep','Sim','Trtmnt']).sum()

    # Reset the index for grouping
    reset_summed = summed.reset_index()
    treatmentDF = reset_summed[reset_summed.Trtmnt == treatment]
    
    # Create the min, max, and median DFs
    maxdf = treatmentDF.groupby('Time').max()
    mindf = treatmentDF.groupby('Time').min()
    meddf = treatmentDF.groupby('Time').median()
    quant25df = treatmentDF.groupby('Time').quantile(0.05)

    
    index = pd.date_range('2000-1-1', periods=len(maxdf), freq='1A')
    for df in [maxdf, mindf, meddf, quant25df]:
        df.index = index

    # Add the three plots to a single set of axes
    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()
    
    # Plot lines for min, max, and median
    maxdf[var].plot(ax = ax, lw = 1.2, color = 'black', *args, **kwargs)
    mindf[var].plot(ax = ax, lw = 1.2, color = 'black', *args, **kwargs)
    meddf[var].plot(ax = ax, lw = 1.2, ls = '--', color = 'gray', *args, **kwargs)
    quant25df[var].plot(ax = ax, lw = 1.2, ls = '--', color = 'red', *args, **kwargs)
    
    # Fill between min and max
    ax.fill_between(maxdf.index, maxdf[var], mindf[var], color = 'white', alpha = 0.3)