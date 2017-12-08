import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os, datetime, math
from scipy import stats

# Define ISI (initial spread index)
def calcISI(fWSV, fFFMC):
    ISI = [0.208 * a*b for a,b in zip(fWSV, fFFMC)]  
    return ISI

# define f(W)
def windEffect(WSV):
    fW = []
    for Wind in WSV:
        if Wind <= 40:
            fWE = np.exp(0.05039 * Wind)
        else:
            fWE = 12 * (1 - np.exp(-0.0818 * (Wind - 28)))
        fW.append(fWE)
    return fW

# define f(F)
def fuelMoistureEffect(FFMC):
    fM = (147.2 * (101 - FFMC)) / (59.5 + FFMC)
    fF = 91.9 * np.exp(-0.1386 * fM) * (1 + ((fM ** 5.31) / (4.93 * (10**7))))
    return fF

# define RSI
# Rate of Spread index (RSI) is a function of weather and fuels, so we need to
# associate this function with our fuels dataframe.
# fuelsDF  -- dataframe containing the dynamic fire fuels parameterization
# fuelCode -- the fuel index assocaited with the fuel type by the dynamic fuels extension
# ISI      -- an initial spread index (function of WSV and FFMC)
def rateOfSpreadIndex(fuelsDF, fuelCode, ISI):
    FC = fuelsDF[fuelsDF.LANDIS_Code == fuelCode]
    a = FC.a
    b = FC.b
    c = FC.c
    RSI = a * (1 - np.exp(-b * ISI)) ** c
    return RSI

# define BE
# Buildup Effect (BE), just like RSI, is calculated using both weather 
# data AND fuels constants.
# fuelsDF  -- dataframe containing the dynamic fire fuels parameterization
# fuelCode -- the fuel index assocaited with the fuel type by the dynamic fuels extension
# ISI      -- a buildup index (calculated from RAWS data, based on time since precip, TA, and rH)
def buildUpEffect(fuelsDF, fuelCode, BUI):
    FC = fuelsDF[fuelsDF.LANDIS_Code == fuelCode]
    q = FC.q
    BUI_0 = FC.BUI
    BE = np.exp(50. * np.log(q)*((1/BUI)-(1/BUI_0)))
    return BE

# fuelsROS(fuelsDF, fuelCODE, ISIarray, BUIarray)
# fuelsDF  -- dataframe containing the dynamic fire fuels parameterization
# fuelCode -- the fuel index assocaited with the fuel type by the dynamic fuels extension
# ISIarray -- An array of ISI values 
# BUIarray -- An array of BUI values
# returns RSI_df, a pandas dataframe, with column and row names set to the BUI and ISI 
# array values respectively
def fuelsROS(fuelsDF, fuelCode, ISIarray, BUIarray):
    RSI_A = np.zeros((len(ISIarray), len(BUIarray)))
    row = 0
    col = 0
    for isi in ISIarray:
        for bui in BUIarray:
            RSI_A[row][col] = np.floor(rateOfSpreadIndex(fuelsDF,fuelCode,isi) \
            * buildUpEffect(fuelsDF,fuelCode,bui))        
            col = col + 1
        col = 0
        row = row + 1
    RSI_df = pd.DataFrame(RSI_A)
    RSI_df.index = ISI_L.astype(int)
    RSI_df.columns = BUI_L.astype(int)

    return RSI_df
	
# Lets start with FMC -- We have to create a lookup-table to make 
# our FMC calculation flexible across fuel types
def calcSFC(FBP_Code, BUI, FFMC):
    if FBP_Code == 'C1':
        SFC = 1.5 * (1 - np.exp(-0.2230 * (FFMC - 81.)))
        if SFC < 0:
            SFC = 0
    if FBP_Code in ['C2', 'M3','M4']:
        SFC = 5.0 * (1 - np.exp(-0.0115 * BUI))
    if FBP_Code in ['C3', 'C4']:
        SFC = 5.0 * (1 - np.exp(-0.0164 * BUI))^2.24
    if FBP_Code in ['C5', 'C6']:
        SFC = 5.0 * (1 - np.exp(-0.0149 * BUI))^2.48
    if FBP_Code == 'C7':
        FFC = 2 * (1 - np.exp(-0.104 * (FFMC - 70)))
        if FFC < 0:
            FFC = 0
        WFC = 1.5 * (1 - np.exp(-0.0201 * BUI))
        SFC = FFC + WFC
    if FBP_Code == 'D1':
        SFC = 1.5 * (1 - np.exp(-0.0183 * BUI))
    return SFC

# Function DEF
def calcCSI(fuelsDF, fuelCode, FMC):  
    FC = fuelsDF[fuelsDF.LANDIS_Code == fuelCode]
    CBH = FC.CBH
    CSI = 0.001 * CBH^1.5 * (460. + 25.9 * FMC)^1.5
    return CSI

# Function DEF
def calcRSO(CSI, SFC):
    RSO = CSI / (300. * SFC)
    return RSO

# Function DEF
def calcCFB(ROS, RSO):
    CFB = 1 - np.exp(-0.23 * (ROS - RSO))
    return CFB