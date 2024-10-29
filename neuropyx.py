#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neuropyx.py
"""
import sys
sys.path.append('/Users/tortugar/My Drive/Penn/Programming/PySleep')
import sleepy
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import csv
import pingouin as pg
import seaborn as sns
import scipy.io as so
import re
import scipy.stats as stats
from scipy import linalg
import matplotlib as mpl
from sklearn.decomposition import PCA
import math
import h5py
# debugger
import pdb



def brstate_class(np_path, sleep_path, sleep_rec, mouse, tend=-1, tstart=0, pzscore=True, 
                class_mode='', pearson=True, pnorm_spec=True, single_mice=True, 
                ma_thr=10, ma_rem_exception=False, box_filt=[],
                pplot=True, config_file='mouse_config.txt'):
    """
    calculate average firing rate during each brain state and then 
    perform statistics for units to classify them into REM-max, Wake-max, or NREM-max.
    For each ROI anova is performed, followed by Tukey-test
            
    :param np_path: folder where firing rates are located
    :param sleep_path: folder where EEG data and sleep annotation is located
    :param sleep_rec: name of sleep recording
    
            Note: easiest way to get np_path, sleep_path, and sleep_rec:    
            paths = neuropyx.load_config(config_file)[mouse]
            ppath, name = os.path.split(paths['SL_PATH'])
            np_path = paths['NP_PATH']

    :param mouse: mouse name 
    :param pzscore: if True, z-score DF/F traces
    
    :param class_mode: class_mode == 'basic': classify ROIs into 
           REM-max, Wake-max and NREM-max ROIs
                                              
           class_mode == 'rem': further separate REM-max ROIs 
           into REM > Wake > NREM (R>W>N) and REM > NREM > Wake (R>N>W) ROIs
                       
           REM-max neurons where Wake and NREM is not significantly different
           are classified as 'R>N=W'
                       
           Neurons that are signficantly modulated by the brain state
           but not part of any of these different classes are labeled Z
                       
           Neurons that are not significantly modulated by the brain state
           are labeled X
           
           NOTE: A given unit can only be part of one subclass.
           R-Off comes before W-max.
           
                       
    :param single_mice: boolean, if True use separate colors for single mice in 
                        summary plots
                        
    :return df_class: pd.DataFrame 
          with columns 
          ['ID', 'R', 'W', 'N', 'F-anova', 'P-anova', 'P-tukey', 'Type', 
           'Depth', 'Quality', 'mouse', 'brain_region']
          
    :rem df_pearson: pd.DataFrame
          with columns
          ['unit', 'r', 'p', 'sig', 'state','depth','Type','Quality']
          
                        
    """
    units, cell_info, M, kcut = load_mouse(mouse, config_file)
    
    # import file that has information on depth
    with open(os.path.join(np_path,'cluster_info.TSV')) as inf:
        reader = csv.reader(inf, delimiter="\t")
        cell_info=list(reader)
    cell_list=[]      
    
    for x in cell_info[1:]: 
        cell_list.append([x[0],x[3],x[6]])
        
    # import sleep annotation file and also do MA functions if chosen   
    sr = sleepy.get_snr(sleep_path,sleep_rec)
    nbin = int(np.round(sr)*2.5)
    sdt = nbin * (1.0/sr)
    
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*sdt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################

    # cut out kcuts: ###############
    #print('Applying kcut')
    tidx = kcut_idx(M, units, kcut)    
    if tidx[-1] >= units.shape[0]:
        tidx = tidx[0:-1]
        
    M = M[tidx]
    units = units.iloc[tidx,:]
    ################################
    
                   
    istart = int(np.round(tstart/sdt))
    if tend == -1:
            iend = M.shape[0]
    else:
        iend = int(np.round(tend/sdt))                   
    if iend >= len(units):
            M = M[istart:len(units)]
    else: 
        M = M[istart:iend]

    # calculate sigma power
    if pearson:
        band=[10,15]
        state_map = {1:'REM', 2:'Wake', 3:'NREM'}
        ddir = os.path.join(sleep_path, sleep_rec)
        P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % sleep_rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where((freq >= band[0]) & (freq <= band[1]))[0]
        df = freq[1] - freq[0]
        
        if len(box_filt) > 0:
            filt = np.ones(box_filt)
            filt = np.divide(filt, filt.sum())
            SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
        
        if pnorm_spec:
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            pow_band = SP[ifreq, :].mean(axis=0)
        else:
            pow_band = SP[ifreq,:].sum(axis=0)*df
        
        pow_band = pow_band[istart:iend]


    # make a nested dict that has each of the units and inside the fr values for all the different sleep bins
    units_stateval = {}
    for unit in units:
        units_stateval[unit] = {1:[], 2:[], 3:[],'depth':[]}
   
    for unit in units:
        depth_idx=units.columns.get_loc(unit)
        if pzscore:
            values = np.array((units[unit] - units[unit].mean())/units[unit].std(ddof=0))
        else: 
            values=np.array(units[unit])
        for state in [1,2,3]:
            seq = np.where(M==state)[0]
            units_stateval[unit][state]=values[seq].tolist()
            units_stateval[unit]['depth']=cell_list[depth_idx][2]        

    columns = ['ID', 'R', 'W', 'N', 'F-anova', 'P-anova', 'P-tukey', 'Type','Depth','Quality']
    data = []
    data_p = []
    for unit in units_stateval:
        stateval = units_stateval[unit]
        val = np.concatenate([stateval[1], stateval[2], stateval[3]],axis=0)
        state = ['R']*len(stateval[1]) + ['W']*len(stateval[2]) + ['N']*len(stateval[3])
        depth=float(units_stateval[unit]['depth'])
        if 'good' in unit: 
            unit_quality='good'
        else:
            unit_quality='mua'
        d = {'state':state, 'val':val}
        df = pd.DataFrame(d)

        res  = pg.anova(data=df, dv='val', between='state')
        res2 = pg.pairwise_tukey(data=df, dv='val', between='state')
 
        def _get_mean(s):
            return df[df['state']==s]['val'].mean()
 
        rmean = _get_mean('R')
        wmean = _get_mean('W')
        nmean = _get_mean('N')
        
        
        if class_mode == 'basic':
            roi_type = 'X'
            # REM-max
            if (rmean > wmean) and (rmean > nmean):
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'R-max'

            #REM-Off (R-Off) 
            elif (rmean < wmean) and (rmean < nmean):  
               cond1 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
               cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'N')]
   
               if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                   roi_type = 'R-Off'        
                   
            # W-max
            elif (wmean > nmean) and (wmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'W-max'
                    
            # N-max 
            elif (nmean > wmean) and (nmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'N-max'
            else:
                roi_type = 'X'
                            
            tmp = [unit, rmean, wmean, nmean, res.F.iloc[0], res['p-unc'].iloc[0], res2['p-tukey'].iloc[0], roi_type,depth,unit_quality]
            data.append(tmp)
            
        # REM mode:
        else:            
            roi_type = 'X'

            if res['p-unc'].iloc[0] < 0.05:
                
                p_nr = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]['p-tukey'].iloc[0] 
                p_rw = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]['p-tukey'].iloc[0] 
                p_nw = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]['p-tukey'].iloc[0] 
                
                # R>W>N
                if (rmean > wmean) and (rmean > nmean) and (wmean > nmean) and p_nr < 0.05 and p_rw<0.05 and p_nw < 0.05:                      
                        roi_type = 'R>W>N'
                # R>N>W
                elif (rmean > wmean) and (rmean > nmean) and (nmean > wmean) and p_nr < 0.05 and p_rw<0.05 and p_nw < 0.05:  
                        roi_type = 'R>N>W'
                # NEW:R>N=W #####################################################
                # The remaining REM-max units: R>N and R>W, but N and W are not significantly different
                # I'm calling these units R>N=W
                elif (rmean > wmean) and (rmean > nmean) and p_nr < 0.05 and p_rw<0.05 and p_nw >= 0.05:
                    roi_type = 'R>N=W'
                # END[NEW:R>N=W] ###############################################

                # Rem-off 
                elif (rmean < wmean) and (rmean < nmean) and p_nr < 0.05 and p_rw<0.05:  
                    roi_type = 'R-Off'  
                # W-max
                elif (wmean > nmean) and (wmean > rmean) and p_nw < 0.05 and p_rw < 0.05:  
                    roi_type = 'W-max' 
                    # N-max
                elif (nmean > wmean) and (nmean > rmean) and p_nw < 0.05 and p_nr < 0.05:  
                    roi_type = 'N-max'
                # NEW:Z ######################################################
                # significantly modulated by brainstate (according to ANOVA),
                # but not part of any of these subclasses
                else:
                    roi_type = 'Z'
            else:
                roi_type = 'X'
                            
            tmp = [unit, rmean, wmean, nmean, res.F.iloc[0], res['p-unc'].iloc[0], res2['p-tukey'].iloc[0], roi_type,depth,unit_quality]
            data.append(tmp)
            
        if pearson:
            # band=[10,15]
            # state_map = {1:'REM', 2:'Wake', 3:'NREM'}
            # ddir = os.path.join(sleep_path, sleep_rec)
            # P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % sleep_rec), squeeze_me=True)
            # SP = P['SP']
            # freq = P['freq']
            # ifreq = np.where((freq >= band[0]) & (freq <= band[1]))[0]
            # df = freq[1] - freq[0]
            
            # if pnorm_spec:
            #     sp_mean = SP.mean(axis=1)
            #     SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            #     pow_band = SP[ifreq, :].mean(axis=0)
            # else:
            #     pow_band = SP[ifreq,:].sum(axis=0)*df
            for s in [1,2,3]:
                idx = np.where(M==s)[0]
                r,p = scipy.stats.pearsonr(units[unit][idx], pow_band[idx])
                if p < 0.05:
                    sig = 'yes'
                else:
                    sig = 'no'
                pearson_temp=[unit, r, p, sig, state_map[s],depth,roi_type,unit_quality]
                data_p.append(pearson_temp)
                
    df_class = pd.DataFrame(data, columns=columns)
    df_pearson=pd.DataFrame(data_p,columns=['unit', 'r', 'p', 'sig', 'state','depth','Type','Quality'])

    if pplot:
        # plotting for unit type 
        mice = [mouse]
        j = 0
        mdict = {}
        for m in mice:
            mdict[m] = j
            j+=1
        clrs = sns.color_palette("husl", len(mice))
    
        types = df_class['Type'].unique()
        types = [i for i in types if not (i=='X')]
        types.sort()
        
        j = 1
        plt.figure()
        for typ in types:
            mouse_shown = {m:0 for m in mice}
            plt.subplot(int('1%d%d' % (len(types), j)))
            df = df_class[df_class['Type']==typ][['R', 'N', 'W']]
            
            sns.barplot(data=df[['R', 'N', 'W']], color='gray')
            for index, row in df.iterrows():
                if single_mice:
                    if mouse_shown[m] > 0:
                        plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color=clrs[mdict[m]])
                    else:
                        plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color=clrs[mdict[m]], label=m)
                        mouse_shown[m] += 1   
                else:
                    plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color='black')
    
            sns.despine()
            plt.title(typ)
            plt.legend()
            if j == 1:
                if not pzscore:
                    plt.ylabel('DF/F (%)')
                else:
                    plt.ylabel('Firing Rate (z-scored)')
            j += 1
        # plot swarm plot with depth vs type 
        df_class_good=df_class.loc[df_class['Quality']=='good']
        plt.figure()
        plt.title('Unit type sorted by depth ')
        sns.swarmplot(data=df_class_good, x='Type', y='Depth',palette="husl")    
        if pearson:
            #plotting pearson plot
            test_df=df_pearson.loc[df_pearson['sig']=='yes']
            test_df=test_df.loc[test_df['Quality']=='good']
            plt.figure()
            sns.swarmplot(data=test_df, x='Type', y='r', hue='state',palette="husl")
            plt.title('correllation of firing rate to'+ ' '+ 'eeg band:'+ str(band))
           
            plt.figure()
            sns.swarmplot(data=test_df, x='Type', y='depth', hue='state',palette="husl")
            plt.title('correllation of firing rate to'+ ' '+ 'eeg band:'+ str(band))
        
    return df_class, df_pearson



def load_config(config):
    """
    Load a config file specifying the file locations for each mouse of
    the sleep recording and the neuropixel recording; 


    Syntax:
        
    MOUSE: Mouse_name
    SL_PATH: Sleep_recording_folder
    NP_PATH: Neuropixels_data_folder
    
    MOUSE: Mouse_name
    SL_PATH: Sleep_recording_folder
    NP_PATH: Neuropixels_data_folder
    KCUT: t0-t1;t2-t3
    NO_REGION: A-B-C
    EXCLUDE: unit_ID1,unit_ID2,...,unit_ID2
    
    NOTE:
    Different mice are separated by new lines
    After a newline the first entry must be MOUSE.
    KCUT: is optional and allows to specify a time frame to be discarded in the
    recording.
    NO_REGION: brain_regions to exclude

    Parameters
    ----------
    config : str
        text file (including path)

    Returns
    -------
    recordings : dict
        dictionary: mouse_ID --> SL_PATH, NP_PATH

    """

    fid = open(config, 'r')
    lines = fid.readlines()

    mouse = 'X'
    recordings = dict()
    newline = True

    for l in lines:
        if re.match(r'^\s*#', l):
            continue

        if len(l)==0  or re.match(r'^\s*$', l):
            newline = True
            continue

        a = re.split(r'\s+', l)

        # cut away the ':' or ' :'
        field = re.split(r'\s*:', a[0])[0]
        value  = a[1]

        if not newline:
            recordings[mouse][field] = value
        else: # newline == True
            mouse = a[1]
            
            if not mouse in recordings:
                recordings[mouse] = {}
            
            newline = False

    for m in recordings:
        if 'KCUT' in recordings[m]:
            a = recordings[m]['KCUT']

            a = re.split(';', a)
            kcuts = []
            for b in a:
                c = re.split('-', b)
                c = [s.strip() for s in c]
                k1 = float(c[0])
                if re.match(r'[\d\.]+', c[1]):
                    k2 = float(c[1])
                else:
                    k2 = c[1]
                    
                kcuts.append( [k1, k2] )
            
            recordings[m]['KCUT'] = kcuts
            
    for m in recordings:
        if 'EXCLUDE' in recordings[m]:
            a = recordings[m]['EXCLUDE']
            k = re.split(r',\s*', a)

            recordings[m]['EXCLUDE'] = k
    
    return recordings



def load_mouse(mouse_id, config_file):
    """
    Load neuropixels recordings as described in mouse_config.txt  

    Parameters
    ----------
    mouse_id : TYPE
        DESCRIPTION.
    config_file : TYPE
        DESCRIPTION.

    Returns
    -------
    units : pd.DataFrame
        unit DataFrame: The columns correspond to the firing rates; 
        the column name is the ID of the unit, the rows correspond to single time points.
    cell_info : pd.DataFrame
        DESCRIPTION.

    """
    dt = 2.5
    
    recs = load_config(config_file)

    sl_path = recs[mouse_id]['SL_PATH']
    (sl_path, sl_name) = os.path.split(sl_path)
    np_path = recs[mouse_id]['NP_PATH']

    # Get sleep annotation:
    M = sleepy.load_stateidx(sl_path, sl_name)[0]

    # load kcuts, i.e. regions at beginning or end to discard from the recording --
    if 'KCUT' in recs[mouse_id]:
        kcut = recs[mouse_id]['KCUT']
        for k in kcut:
            if k[1] == '$':
                k[1] = len(M)*dt
    else:
        kcut = ()

    traind_file = ''
    if os.path.isfile(os.path.join(np_path, 'traind.csv')):
        traind_file = 'traind.csv'
    elif os.path.isfile(os.path.join(np_path, 'traind.csv')):
        traind_file = 'spike_train.csv'
    else:
        traind_file = 'traind_lfp.csv'

    units = pd.read_csv(os.path.join(np_path, traind_file)) 

    if os.path.isfile(os.path.join(np_path,'channel_locations.json')):
        regions=pd.read_json(os.path.join(np_path,'channel_locations.json')).T
        regions=regions.iloc[0:-1]
        regions['ch']=regions.index.str.split('_').str[-1].astype('int64')


    cell_info = pd.read_csv(os.path.join(np_path,'cluster_info.TSV'),delimiter="\t")
    cell_info['group']=cell_info['group'].fillna(cell_info['KSLabel'])

    cl_id = ''
    if cell_info.columns.isin(['cluster_id']).sum():
        cl_id = 'cluster_id'
    else:
        cl_id = 'id'
    
    cell_info['ID'] = cell_info[cl_id].astype(str) +'_'+cell_info['group'].astype(str)
    
    if os.path.isfile(os.path.join(np_path,'channel_locations.json')):    
        cell_info=cell_info.merge(regions,on='ch')

    # NEW 07/15/23
    if len(M) > units.shape[0]:
        M = M[0:units.shape[0]]
        
    return units, cell_info, M, kcut



def exclude_units(units, mouse, config_file):
    """
    Exclude units listed in mouse_config.txt under 'EXCLUDE:'

    Parameters
    ----------
    units : pd.pandas
        each colums hold the firing rate of a unit.
    mouse : string
        the mouse name.
    config_file : string
        string of the name of the config file as loaded by &load_config().

    Returns
    -------
    None.

    """
    paths = load_config(config_file)[mouse]
    if 'EXCLUDE' in paths:
        ex_units = paths['EXCLUDE']
        units.drop(columns=ex_units, inplace=True)



def fr_corr_state(units, M, idx1=[], idx2=[], win=60, state=3, ma_thr=10, mode='cross', pzscore=True, pplot=True, dt=2.5):
    """
    Perform cross-correlation between firing rates for a given brain state. 
    Calculate the correlation for each pair of the provided neurons unitIDs (in idx1 and idx2)

    Parameters
    ----------
    units : pd.DataFrame
        Each column corresponds to one unit. The column name is the unitID.
        So to get all unit ID, get all column names
    M : np.array
        hypnogram.
    idx1 : list, optional
        List of unitIDs for neuron1. If empty, use all neurons (unitIDs)
        The default is [].
    idx2 : list, optional
        List of unitIDs for neuron1. If empty, use all neurons (unitIDs)
        The default is [].
    win : float, optional
        DESCRIPTION. The default is 120.
    state : TYPE, optional
        DESCRIPTION. The default is 3.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    mode : TYPE, optional
        DESCRIPTION. The default is 'cross'.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    pplot : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    # If no IDs are provided for unit1, use all units as "unit1"
    if len(idx1) == 0:
        idx1 = units.columns.unique()
        
    # If no IDs are provided for unit2, use all units as "unit2"
    if len(idx2) == 0:
        idx2 = units.columns.unique()
        

    # get all firing rates and cast them to np.array    
    arr1 = units[idx1]
    arr2 = units[idx2]
    fr1 = np.array(arr1)
    fr2 = np.array(arr2)

    n1 = fr1.shape[1]
    n2 = fr2.shape[1]

    # z-scoring
    if pzscore:
        for i in range(n1):
            fr1[:,i] = (fr1[:,i]-fr1[:,i].mean()) / fr1[:,i].std()
            
        for i in range(n2):
            fr2[:,i] = (fr2[:,i]-fr2[:,i].mean()) / fr2[:,i].std()
            

    iwin = int(win/dt)
    t = np.arange(-iwin, iwin+1) * dt
    
    seq = sleepy.get_sequences(np.where(M==2)[0])
    if ma_thr > 0:
        seq = sleepy.get_sequences(np.where(M == 2)[0])
        for s in seq:
            if len(s) * dt < ma_thr:
                M[s] = 3
    
    seq = sleepy.get_sequences(np.where(M==state)[0])
    seq = [s for s in seq if len(s)*dt >= 2*win]
    
    data = []
    for i in range(n1):
        for j in range(n2):
            CC = []
            for s in seq:
                m = len(s)
                fr1_cut = fr1[s,i]
                fr2_cut = fr2[s,j]

                fr1_cut = fr1_cut - fr1_cut.mean()
                fr2_cut = fr2_cut - fr2_cut.mean()
                
                norm = np.nanstd(fr1_cut) * np.nanstd(fr2_cut)
                # for used normalization, see: https://en.wikipedia.org/wiki/Cross-correlation          
                
                if norm > 0:
                    xx = (1/m) * scipy.signal.correlate(fr1_cut, fr2_cut) / norm
                    ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
                    ii = [int(i) for i in ii]
                    
                    ii = np.concatenate((np.arange(m-iwin-1, m), np.arange(m, m+iwin, dtype='int')))
                    # note: point ii[iwin] is the "0", so xx[ii[iwin]] corresponds to the 0-lag correlation point
                    CC.append(xx[ii])
            
            if len(CC) > 0:
                CC = np.array(CC).mean(axis=0)
                #pdb.set_trace()
                m = CC.shape[0]
                un1 = list(idx1)[i]
                un2 = list(idx2)[j]
                
                label = r'%s~%s' % (un1, un2)
                data += zip(t, CC, [un1]*m, [un2]*m, [label]*m)
            
    df = pd.DataFrame(data=data, columns=['time', 'cc', 'unit1', 'unit2', 'label'])
    
    return df
            

            
def sort_xcorr_byregion(type1, type2, corr_frame, unit_info):
    """
    Go through all brain regions contained in DataFrame unit_info['brain_region'],
    For each pair of brain regions take all neurons of type1 in area A and all neurons
    of type2 in area B and determine the average cross-correlation for this type/area pair.

    Parameters
    ----------
    type1 : TYPE
        
    type2 : TYPE
        DESCRIPTION.
    corr_frame : pd.DataFrame with columns ['time', 'cc', 'unit1', 'unit2', 'label']
        DataFrame as returned by the function neuropyx.fr_corr_state()
        The DataFrame contains for pairs of neurons (indicated by the IDs in columns 'unit1' and 'unit2')
        the cross-correlation
    unit_info : pd.DataFrame        
        Necessary columns: 'ID' (IDs of units), 'brain_region' (brain region of unit 'ID'), 
        'Type' (neuron subclass as computed by neuropyx.brstate_fr())

    Returns
    -------
    None.

    """
    
    regions = unit_info.brain_region.unique()
    
    nregions = len(regions)
    
    data = []
    for i in range(nregions):
        for j in range(0, nregions):
            region1 = regions[i]
            region2 = regions[j]
            
            ids1 = unit_info[(unit_info['brain_region']==region1) & (unit_info['Type']==type1)]['ID']
            ids2 = unit_info[(unit_info['brain_region']==region2) & (unit_info['Type']==type2)]['ID']
            
            print(ids2)
            if len(ids1) > 0 and len(ids2) > 0:            
                df = corr_frame[(corr_frame.unit1.isin(ids1)) & (corr_frame.unit2.isin(ids2))]
                dfm = df.groupby(['time']).mean()
                
                time = np.array(dfm.index)
                cc = dfm['cc']
                m = len(cc)
    
                label = '%s~%s' % (region1, region2)
                
                data += zip(time, cc, [region1]*m, [region2]*m, [label]*m)

    df = pd.DataFrame(data=data, columns=['time', 'cc', 'region1', 'region2', 'label'])            
 
    return df           



def fr_transitions(units, M, unit_info, transitions, pre, post, si_threshold, sj_threshold, 
                   ma_thr=10, ma_rem_exception=False, sdt=2.5, pzscore=True, sf=0, ma_mode=False, 
                   attributes=[], kcuts=[], 
                   pspec=False, fmax=20, spe_filt=[], mouse='', config_file='mouse_config.txt'):
    """
    
    
    Note: If you like to also calculate the average EEG specotrogram for each transition and unit,
    set $psec=True, set the $mouse name and select the right $config_file
    

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    unit_info : TYPE
        DESCRIPTION.
    transitions : TYPE
        DESCRIPTION.
    pre : TYPE
        DESCRIPTION.
    post : TYPE
        DESCRIPTION.
    si_threshold : TYPE
        DESCRIPTION.
    sj_threshold : TYPE
        DESCRIPTION.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is False.
    sdt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    sf : TYPE, optional
        DESCRIPTION. The default is 0.
    ma_mode : TYPE, optional
        DESCRIPTION. The default is False.
    attributes : list of strings, optional
        Allows you to transfer columns in DataFrame $unit_info to the returned DataFrame. The default is [].
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    pspect : bool
        if True, also claculate EEG spectrogram
    mouse : str,
        If $pspec == True, needs to be set to an existing
        mouse name
    config_file : str
        Name of mouse recording configuration file
    fmax : float
        Maximum frequency for EEG spectrogram

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """

    states = {1:'R', 2:'W', 3:'N', 4:'M'}
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, units, kcuts)
    M = M[tidx]
    units = units.iloc[tidx,:]
    ################################
        
    ipre  = int(np.round(pre/sdt))
    ipost = int(np.round(post/sdt))
    m = ipre + ipost
    t = np.arange(-ipre, ipost) * sdt
    
    unitIDs = units.columns.unique()
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*sdt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>=1) and (M[s[0] - 1] != 1):
                        if ma_mode:
                            M[s] = 4
                        else:
                            M[s] = 3
                else:
                    if ma_mode:
                        M[s] = 4
                    else:
                        M[s] = 3    

    # NEW: load spectrogram
    # load spectrogram and normalize
    if pspec:
        path = load_config(config_file)[mouse]['SL_PATH']
        ppath, name = os.path.split(path)

        
        P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where(freq <= fmax)[0]
        
        if len(spe_filt) > 0:
            filt = np.ones(spe_filt)
            filt = np.divide(filt, filt.sum())
            SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')            
            
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        SP[:,tidx]
        
        
        
        unit_transspe = {}
        mx_transspe = {}
        for (si,sj) in transitions:
            # string label for type of transition:
            sid = states[si] + states[sj]
            unit_transspe[sid] = {r:[] for r in unitIDs}
            mx_transspe[sid] = np.zeros((len(ifreq), len(t), len(unitIDs)))        
    ##########################################################################

    data = []
    for unit in unitIDs:
        unit_annotation = unit_info[unit_info.ID == unit]
        if unit_annotation.shape[0] > 0:
            attr = unit_annotation[attributes].values.tolist()[0]
            
        else:
            attr = ['X', 'X']
        
        fr = np.array(units[unit])
        if sf > 0:
            fr = sleepy.smooth_data(fr, sf)
        if pzscore:
            fr = (fr-fr.mean()) / fr.std()
        
        for (si,sj) in transitions:
            # string label for type of transition:
            sid = states[si] + states[sj]
            seq = sleepy.get_sequences(np.where(M==si)[0])
    
            for s in seq:
                ti = s[-1]
    
                # check if next state is sj; only then continue
                if ti < len(M)-1 and M[ti+1] == sj:
                    # go into future
                    p = ti+1
                    while p<len(M)-1 and M[p] == sj:
                        p += 1
                    p -= 1
                    sj_idx = list(range(ti+1, p+1))
                    # so the indices of state si are seq
                    # the indices of state sj are sj_idx
    
                    if ipre <= ti < len(M)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:
                        act = fr[ti-ipre+1:ti+ipost+1]
                        # Note: ti+1 is the first time point of the "post" state

                        # i = 10, ipre = 2, ipost = 2
                        # 8,9,10
                        # np.arange(8,12) = 8,9,10,11,12

                        if pspec:                        
                            #spe_si = SP[ifreq,ti-ipre+1:ti+1]
                            #spe_sj = SP[ifreq,ti+1:ti+ipost+1]
                            #spe = np.concatenate((spe_si, spe_sj), axis=1)
                            spe = SP[ifreq, ti-ipre+1:ti+ipost+1]                            
                            unit_transspe[sid][unit].append(spe)
                            
                        #spm_si = emg_ampl[ti-ipre+1:ti+1]
                        #spm_sj = emg_ampl[ti+1:ti+ipost+1]
                        #spm = np.concatenate((spm_si, spm_sj))

                        #t = np.arange(-ipre*sdt, ipost*sdt - sdt + sdt / 2, sdt)
                        new_data = zip([unit]*m, t, act, [sid]*m)                                                
                        new_data = [list(x) + attr for x in list(new_data)]

                        data += new_data

    df = pd.DataFrame(data=data, columns=['ID', 'time', 'fr', 'trans'] + attributes)

    if pspec:
        for (si,sj) in transitions:
            for i,unit in enumerate(unitIDs):
                # string label for type of transition:
                sid = states[si] + states[sj]
                tmp = np.array(unit_transspe[sid][unit])
                mx_transspe[sid][:,:,i] = np.nanmean(tmp, axis=0)
            
    if not pspec:
        return df, []
    else:
        return df, mx_transspe



def fr_transitions_stats(df_trans, base_int, unit_avg=True, dt=2.5, time_mode='midpoint'):
    """
    
    Parameters
    ----------
    df_trans : TYPE
        DESCRIPTION.
    base_int : TYPE
        DESCRIPTION.
    unit_avg : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    time_mode : str, optional
        options: 'midpoint' or 'endpoint'. The default is 'midpoint'.

    Returns
    -------
    df_stats : TYPE
        DESCRIPTION.

    """
    
    # test if df has column ms_id

    if not 'ms_id' in df_trans.columns:
        mice = list(df_trans['mouse'])
        ids = list(df_trans['ID'])
        
        ms_ids = [m + '-' + i for m,i in zip(mice, ids)]
        df_trans['ms_id'] = ms_ids


    ids = list(df_trans.ms_id.unique())

    #df_trans = df_trans[df_trans.trans == trans]
    dfm_trans = df_trans[['ms_id', 'time', 'fr', 'trans']].groupby(['ms_id', 'time', 'trans',]).mean().reset_index()

    #t = np.array(dfm_trans.loc[df_trans.ms_id == ids[0], 'time'])
    t = dfm_trans.time.unique()
    
    # number of bins per time bin
    ibin = int(base_int / dt)
    pre = t[0]
    post = t[-1]
    nbin = int(np.floor((abs(pre)+post)/base_int))            
    trans_dict = {}
    for tr in df_trans.trans.unique():
        trans_mx = np.zeros((len(ids), len(t)))

        for i,ID in enumerate(ids):
            fr = dfm_trans.loc[(dfm_trans.ms_id == ID) & (dfm_trans.trans==tr), 'fr']
            trans_mx[i,:] = fr
        trans_dict[tr] = trans_mx

    tinit = t[0]
    # Statistics: When does activity becomes significantly different from baseline?
    ibin = int(np.round(base_int / dt))
    nbin = int(np.floor((abs(pre)+post)/base_int))
    data = []
    for tr in trans_dict:
        trans = trans_dict[tr]
        base = trans[:,0:ibin].mean(axis=1)
        for i in range(1,nbin):
            p = stats.ttest_rel(base, trans[:,i*ibin:(i+1)*ibin].mean(axis=1))
            sig = 'no'
            if p.pvalue < (0.05 / (nbin-1)):
                sig = 'yes'
            if time_mode == 'midpoint':
                tpoint = i*(ibin*dt)+tinit + ibin*dt/2
            else:
                tpoint = i*(ibin*dt)+tinit + ibin*dt
            tpoint = float('%.2f'%tpoint)
            
            data.append([tpoint, p.pvalue*(nbin-1), sig, tr])
    df_stats = pd.DataFrame(data = data, columns = ['time', 'p-value', 'sig', 'trans'])

    return df_stats



def pc_transitions(PC, M, transitions, pre, post, si_threshold, sj_threshold, 
                   ma_thr=10, ma_rem_exception=False, sdt=2.5, ma_mode=False, 
                   kcuts=[], allowed_idx=[], pzscore_pc=False):
    """
    Calculate timecourse of PCs in population activity relative to brain state 
    transitions.

    Parameters
    ----------
    PC : np.array 
        Number of PCx x number of time bins
        PCs; each row corresponds to one PC.
    M : np.array
        hynpogram.
    transitions : TYPE
        DESCRIPTION.
    pre : TYPE
        DESCRIPTION.
    post : TYPE
        DESCRIPTION.
    si_threshold : TYPE
        DESCRIPTION.
    sj_threshold : TYPE
        DESCRIPTION.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : bool, optional
        If True, don't touch wake following REM sleep. The default is False.
    sdt : float, optional
        time bin duration in seconds of one brain state. The default is 2.5.
    ma_mode : bool, optional
        If True, then specifically analysis transitions from and to MAs. 
        Note that when considering transitions from and to NREM, MAs are 
        considered as NREM sleep.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    allowed_idx : TYPE, optional
        DESCRIPTION. The default is [].
    pzscore_pc : bool
        If True, z-score PCs across entire recording

    Returns
    -------
    df : pd.DataFrame
        with columns ['event', 'pc', 'time', 'fr', 'trans'].

    """

    states = {1:'R', 2:'W', 3:'N', 4:'M'}
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)
    M = M[tidx]
    ################################
        
    ipre  = int(np.round(pre/sdt))
    ipost = int(np.round(post/sdt))
    m = ipre + ipost
    t = np.arange(-ipre, ipost) * sdt
    
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*sdt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>=1) and (M[s[0] - 1] != 1):
                        if ma_mode:
                            M[s] = 4
                        else:
                            M[s] = 3
                else:
                    if ma_mode:
                        M[s] = 4
                    else:
                        M[s] = 3    

    Mrepr = M.copy()
    M[M==4] = 3

    if not ma_mode:
        #just forget about MAs:
        Mrepr = M

    if len(allowed_idx) == 0:
        allowed_idx = range(0, len(M))

    # zscore pcs: #############################################################
    if pzscore_pc:
        for i in range(PC.shape[0]):
            PC[i,:] = (PC[i,:]- PC[i,:].mean()) / PC[i,:].std()
    ###########################################################################

    data = []
    ev = 0
    for count,fr in enumerate(PC):
        label = 'pc%d' % int(count+1)
        
        for (si,sj) in transitions:
            # string label for type of transition:
            sid = states[si] + states[sj]
            if si == 3 and not ma_mode:
                seq = sleepy.get_sequences(np.where(M==si)[0])
            elif si==3 and ma_mode:
                seq = sleepy.get_sequences(np.where(Mrepr==si)[0])
            else:
                seq = sleepy.get_sequences(np.where(Mrepr==si)[0])
            
            for s in seq:
                # ti is the last bin in the current sequence
                ti = s[-1]
    
                if si == 3 and ma_mode:
                    p = s[0]
                    
                    p = p-1
                    while p >0 and M[p] == 3:
                        p = p-1
                    p = p+1
                    s = np.arange(p, ti+1)
    
                # check if next state is sj; only then continue
                if ti < len(M)-1 and Mrepr[ti+1] == sj:
                    # go into future
                    p = ti+1
                    if sj == 3:
                        # if sj == 3, we're treating MAs as NREM
                        while p<len(M)-1 and M[p] == sj:
                            p += 1
                    else:
                        while p<len(M)-1 and Mrepr[p] == sj:
                            p += 1
                    p -= 1
                    sj_idx = list(range(ti+1, p+1))
                    # so the indices of state si are seq
                    # the indices of state sj are sj_idx
    
                    if ti in allowed_idx and ipre <= ti < len(M)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:
                        act = fr[ti-ipre+1:ti+ipost+1]
                        # Note: ti+1 is the first time point of the "post" state
                        # i = 10, ipre = 2, ipost = 2
                        # 8,9,10
                        # np.arange(8,12) = 8,9,10,11,12
                        data += zip([s[0]]*m, [label]*m, t, act, [sid]*m)                                               
                        ev += 1

    df = pd.DataFrame(data=data, columns=['event', 'pc', 'time', 'fr', 'trans'])

    return df



def pc_transitions_laser(mouse, PC, M, transitions, pre, post, si_threshold, sj_threshold, 
                   ma_thr=10, ma_rem_exception=False, sdt=2.5, ma_mode=False, 
                   kcuts=[], allowed_idx=[], pzscore_pc=True, config_file=''):
    """
    Compare spontaneous and laser-induced brain state transitions

    Parameters
    ----------
    PC : np.array 
        Number of PCx x number of time bins
        PCs; each row corresponds to one PC.
    M : np.array
        hynpogram.
    transitions : TYPE
        DESCRIPTION.
    pre : TYPE
        DESCRIPTION.
    post : TYPE
        DESCRIPTION.
    si_threshold : TYPE
        DESCRIPTION.
    sj_threshold : TYPE
        DESCRIPTION.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : bool, optional
        If True, don't touch wake following REM sleep. The default is False.
    sdt : float, optional
        time bin duration in seconds of one brain state. The default is 2.5.
    ma_mode : bool, optional
        If True, then specifically analysis transitions from and to MAs. 
        Note that when considering transitions from and to NREM, MAs are 
        considered as NREM sleep.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    allowed_idx : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    dt = 2.5
    states = {1:'R', 2:'W', 3:'N', 4:'M'}
    
    nhypno = M.shape[0]
    ndim = PC.shape[0]
    tidx = np.arange(0, nhypno)
    
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################   
    
        
    ipre  = int(np.round(pre/sdt))
    ipost = int(np.round(post/sdt))
    m = ipre + ipost
    t = np.arange(-ipre, ipost) * sdt
    
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*sdt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>=1) and (M[s[0] - 1] != 1):
                        if ma_mode:
                            M[s] = 4
                        else:
                            M[s] = 3
                else:
                    if ma_mode:
                        M[s] = 4
                    else:
                        M[s] = 3    

    Mrepr = M.copy()
    M[M==4] = 3

    if not ma_mode:
        #just forget about MAs:
        Mrepr = M

    if len(allowed_idx) == 0:
        allowed_idx = range(0, len(M))

        

    #######################################################################
    # get laser start and end index after excluding kcuts: ################
    ddir =  load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(ddir)
    
    sr = sleepy.get_snr(ppath, name)
    nbin = int(np.round(sr)*dt)
    lsr = so.loadmat(os.path.join(ddir, 'laser_%s.mat' % name), squeeze_me=True)['laser']

    idxs, idxe = sleepy.laser_start_end(lsr)
    idxs = [int(i/nbin) for i in idxs]
    idxe = [int(i/nbin) for i in idxe]

    laser_idx = []
    for (si,sj) in zip(idxs, idxe):
        laser_idx += list(range(si,sj+1))

    nlsr = int(np.floor(lsr.shape[0]/nbin))
    laser = np.zeros((nlsr,))
    laser[laser_idx] = 1
    laser = laser[tidx]
    # get again indices after kcut
    laser_idx = np.where(laser == 1)[0]

    idxs = [s[0]  for s in sleepy.get_sequences(np.where(laser == 1)[0])]
    idxe = [s[-1] for s in sleepy.get_sequences(np.where(laser == 1)[0])]
    #######################################################################
    
    # zscore pcs:
    if pzscore_pc:
        for i in range(PC.shape[0]):
            PC[i,:] = (PC[i,:]- PC[i,:].mean()) / PC[i,:].std()
    
    
    data = []
    ev = 0
    for count,fr in enumerate(PC):
        label = 'pc%d' % int(count+1)
        
        for (si,sj) in transitions:
            # string label for type of transition:
            sid = states[si] + states[sj]
            if si == 3 and not ma_mode:
                seq = sleepy.get_sequences(np.where(M==si)[0])
            elif si==3 and ma_mode:
                seq = sleepy.get_sequences(np.where(Mrepr==si)[0])
            else:
                seq = sleepy.get_sequences(np.where(Mrepr==si)[0])
            
            for s in seq:
                # ti is the last bin in the current sequence
                ti = s[-1]
    
                if si == 3 and ma_mode:
                    p = s[0]
                    
                    p = p-1
                    while p > 0 and M[p] == 3:
                        p = p-1
                    p = p+1
                    s = np.arange(p, ti+1)
    
                # check if next state is sj; only then continue
                if ti < len(M)-1 and Mrepr[ti+1] == sj:
                    # go into future
                    p = ti+1
                    if sj == 3:
                        # if sj == 3, we're treating MAs as NREM
                        while p<len(M)-1 and M[p] == sj:
                            p += 1
                    else:
                        while p<len(M)-1 and Mrepr[p] == sj:
                            p += 1
                    p -= 1
                    sj_idx = list(range(ti+1, p+1))
                    
                    
                    # so the indices of state si are seq
                    # the indices of state sj are sj_idx
    
                    if ti in allowed_idx and ipre <= ti < len(M)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:
                        act = fr[ti-ipre+1:ti+ipost+1]
                        # Note: ti+1 is the first time point of the "post" state
                        # i = 10, ipre = 2, ipost = 2
                        # 8,9,10
                        # np.arange(8,12) = 8,9,10,11,12
                        
                        laser_on = 'no'
                        if ti+1 in laser_idx:
                            laser_on = 'yes'

                        
                        data += zip([s[0]]*m, [label]*m, t, act, [sid]*m, [laser_on]*m)                                               
                        ev += 1

    df = pd.DataFrame(data=data, columns=['event', 'pc', 'time', 'fr', 'trans', 'laser_on'])

    return df





def fr_svd(units, nsmooth=0, pzscore=False):
    """
    perform SVD on firing rates of a population of neurons
    
    Parameters
    ----------
    units : pd.DataFrame
        each column is a unit; the column names are the unitIDs.
    ndim : int, optional
        DESCRIPTION. The default is 3.
    nsmooth : float, optional
        If > 0, som. The default is 0.

    Returns
    -------
    PC : np.array
        Each row vector corresponds to one principal component.
        dimensions: ndim x timepoints
    V : np.array
        Variance captured by the principal components

    """
    
    # first transform pd.DataFrame units to np.array:
    # Matrix arrangement:
    # rows - neurons; columns - time points
    # NOTE: the rows (neurons) are the variables (dimensions), 
    # the time points are the samples (trials)
    # through PCA we want to keep the number of samples, but we want to reduce
    # the dimensions. Again, for our matrix R, we have the arrangement:
    #              samples (=time)
    # variables x
    #
    # In our case, that's
    #              timepoints
    # units     x
        
    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units

    R = np.zeros((nvar, nsample))
    
    i = 0
    for unit in unitIDs:
        R[i,:] = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (R[i,:] - R[i,:].mean()) / R[i,:].std()

        i += 1

    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R.T / np.sqrt(nsample-1)

    U,S,Vh = scipy.linalg.svd(Y)

    return U, S, Vh    



def pr_components(units, ndim=3, nsmooth=0, pzscore=False):
    """
    Calculate principal components of simultaneously recorded firing rates.
    
    Parameters
    ----------
    units : pd.DataFrame
        each column is a unit; the column names are the unitIDs.
    ndim : int, optional
        DESCRIPTION. The default is 3.
    nsmooth : float, optional
        If > 0, som. The default is 0.

    Returns
    -------
    PC : np.array
        Each row vector corresponds to one principal component.
        dimensions: ndim x timepoints
    V : np.array
        Variance captured by the principal components

    """
    
    # first transform pd.DataFrame units to np.array:
    # Matrix arrangement:
    # rows - neurons; columns - time points
    # NOTE: the rows (neurons) are the variables (dimensions), 
    # the time points are the samples (trials)
    # through PCA we want to keep the number of samples, but we want to reduce
    # the dimensions. Again, for our matrix R, we have the arrangement:
    #              samples (=time)
    # variables x
    #
    # In our case, that's
    #              timepoints
    # units     x
      
    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units

    R = np.zeros((nvar, nsample))
    
    i = 0
    for unit in unitIDs:
        R[i,:] = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (R[i,:] - R[i,:].mean()) / R[i,:].std()
        i += 1

    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R.T / np.sqrt(nsample-1)
    U,S,Vh = scipy.linalg.svd(Y)

    # project data @R onto principal direction @Vh
    # Vh: nvar x nvar; each row is an eigenvector of the covariance matrix.
    # So using Vh * R, we project the data into the covariance space. 
    PC = np.dot(Vh, R)[0:ndim,:]
    V = S**2
    
    # Side note; the same result you can get also with:
    # U * S ... in more detail:
    # SM = np.zeros((nsample, nsample))
    # for i in range(nvar):
    #    SM[i,i] = S[i]
    # PC = np.dot(U, SM) * np.sqrt(nsample-1)


    ###########################################################################
    # plots
    plt.figure()
    plt.subplot(211)
    plt.plot(V, '.')
    
    plt.subplot(212)
    plt.plot(PC.T)

    var = S**2
    var_total = np.sum(S**2)
    
    # Calculate the cumulative variance explained, i.e. how much
    # of the variance is captured by the first i principal components. 
    p = []
    for i in range(1,len(var)+1):
        s = np.sum(var[0:i])
        p.append(s/var_total)
    
    plt.figure()
    plt.plot(p, '.')

    return PC, V



def sleep_components(units, M, wake_dur=600, wake_break=60, ndim=3, nsmooth=0, 
                     pzscore=False, detrend=False, pplot=True, pc_sign=[], 
                     kcuts=[], mode='w', ppath='', mouse='', collapse=False,
                     ylim=[], config_file='mouse_config.txt'):
    """
    Calculate principal components, excluding long wake periods in the recording.
    Long wake periods are defined by the parameters $wake_dur (duration of wake episodes)
    and $wake_break (wake episodes separated by less than $wake_break seconds are fused).

    Parameters
    ----------
    units : pd.DataFrame
        each column corresponds to a unit; each row is a time bin
    M : np.array
        hypnogram.
    wake_dur : float, optional
        Exclude wake periods that are longer than $wake_dur seconds. The default is 600.
    wake_break : float, optional
        Two wake periods that are separated by less than $wake_break seconds 
        are merged to one period. The default is 60.
    ndim : int, optional
        reduce data (matrix of firing rate vector) to $ndim dimensions using PCA. The default is 3.
    nsmooth : float, optional
        Smooth firing rate vector. The default is 0.
    pzscore : boolean, optional
        If True, zscore firing rates. The default is False.
    detrend : boolean, optional
        If True, detrend each firing rate vector.
    pc_sign : list with $ndim elements, either 1 or -1.
        The sign of PCs is ambiguous, so if preferred multiply, PC i with pc_sign[i]
    kcuts : list of tuples or lists with two elements.
        Discard the time interval ranging from kcuts[i][0] to kcuts[i][1] seconds
    mode : string with characters 'r' and/or 'w'
        if 'r' in mode, remove all REM indices for PC computation
        if 'w' in mode, remove all Wake indices for PC computation
    collapse: boolean
        if True, plot each PC in its own axis
    ylim: empty list, or tuple
        if empty list, don't fix ylims, otherwise set plt.ylim(ylim) for each PC
    config_file: str
        mouse configuration file as loaded by &load_config()
        
    Returns
    -------
    PC : np.array
        The $ndim principal components. Note although the PCs have been calculated
        without long wake periods, the returned PCs do include all wake periods.

    V : np.array
        Eigenvalues of the covariance matrix = Variance associated with each PC
    
    Vh : np.array
        each row in Vh is an eigenvector of the covariance matrix
            
    idx : np.array 
        Indices of time bins used for PC calculation. 
        NOTE that @idx are the indices obtained
        AFTER cutting out the KCUT intervals! 

    """
    dt = 2.5  
    
    nhypno = np.min((len(M), units.shape[0]))
    Morig = M.copy()
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
        
        print(len(tidx))
    ###########################################################################    
    
    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nsample = nhypno           # number of time points
    nvar    = len(unitIDs)     # number of units
    # Note the number of samples stays the same; while the number is variables
    # (or dimensions) is reduced! We want to keep the same number of time points,
    # but have only a few 'modes'.    
    R = np.zeros((nvar, nsample))
    #@tidx are the indices we're further considering.
    
    for i,unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        tmp = tmp[tidx]
        if detrend:
            tmp = scipy.signal.detrend(tmp)

        if pzscore:
            R[i,:] = (tmp - tmp.mean()) / tmp.std()
        else:
            R[i,:] = tmp
        
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    #for i in range(nvar):
    #    R[i,:] = R[i,:] - R[i,:].mean()

    # find long wake blocks:
    widx = sleepy.get_sequences(np.where(M==2)[0], ibreak=int(wake_break/dt))
    # all REM sequences
    ridx = sleepy.get_sequences(np.where(M==1)[0])

    nidx = sleepy.get_sequences(np.where(M==3)[0])

    tmp = []
    for w in widx:
        if len(w) * dt > wake_dur:
            tmp += list(w)            
    widx = tmp

    tmp = []
    for r in ridx:
        tmp += list(r)            
    ridx = tmp

    tmp = []
    for r in nidx:
        tmp += list(r)            
    nidx = tmp

    nhypno = np.min((len(M), units.shape[0]))
    idx = np.arange(0, np.min((len(M), units.shape[0])))
    if 'w' in mode:
        idx = np.setdiff1d(idx, widx)
    else:
        widx = []
    if 'r' in mode:
        idx = np.setdiff1d(idx, ridx)
    if 'n' in mode:
        idx = np.setdiff1d(idx, nidx)
    else:
        ridx = []
                    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R[:,idx].T / np.sqrt(nsample-1)
    # make sure the the columns of Y are mean-zero
    for i in range(nvar):
        Y[:,i] = Y[:,i] - Y[:,i].mean()
        R[i,:] = R[i,:] - R[i,idx].mean()
    
    # SVD
    # Note that each column is a neuron, 
    # and each row is a time point
    U,S,Vh = scipy.linalg.svd(Y)
    
    # each row in Vh is an eigenvector of the COV matrix;
    # to get the PCs, we project R onto the eigenvectors:
    PC = np.dot(Vh, R)[0:ndim,:]
    V = S**2
    
    if len(pc_sign) > 0:
        i = 0
        for s in pc_sign:
            PC[i,:] = PC[i,:] * s
            i += 1
        
    if pplot:
        add_laser = False
        t = np.arange(0, nhypno) * dt
        plt.figure()
        tmp = widx+ridx
        tmp.sort()
        widx = sleepy.get_sequences(np.array(tmp))
        axes_exc = plt.axes([0.2, 0.9, 0.7, 0.05])
        for w in widx:
            if len(w) > 1:
                if w[-1] < len(M):
                    plt.plot([t[w[0]], t[w[-1]]], [1, 1], 'k', lw=2)
        plt.ylim([0, 2])
        plt.xlim((t[0], t[-1]))
                
        # if laser exists also add laser here --
        if mouse:
            ddir = load_config(config_file)[mouse]['SL_PATH']
            ppath, name = os.path.split(ddir)

            if os.path.isfile(os.path.join(ddir, 'laser_%s.mat' % name)):
                lsr = so.loadmat(os.path.join(ddir, 'laser_%s.mat' % name), squeeze_me=True)['laser']
    
                sr = sleepy.get_snr(ppath, name)
                nbin = int(np.round(sr)*2.5)
            
                idxs, idxe = sleepy.laser_start_end(lsr)
                idxs = [int(i/nbin) for i in idxs]
                idxe = [int(i/nbin) for i in idxe]
                laser_idx = []
                for (si,sj) in zip(idxs, idxe):
                    laser_idx += list(range(si,sj+1))
    
                nlsr = int(np.floor(lsr.shape[0]/nbin))
                laser = np.zeros((nlsr,))
                laser[laser_idx] = 1
                laser = laser[tidx]
    
                lsr_seq = sleepy.get_sequences(np.where(laser==1)[0])
                
                for w in lsr_seq:
                    if w[-1] < len(M):
                        plt.plot([t[w[0]], t[w[-1]]], [1.5, 1.5], 'b', lw=2)
                plt.xlim((t[0], t[-1]))
                
                add_laser = True
                        
        sleepy._despine_axes(axes_exc)                
        axes_brs = plt.axes([0.2, 0.85, 0.7, 0.05], sharex=axes_exc)        
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        sleepy._despine_axes(axes_brs)

        if collapse:    
            plt.axes([0.2, 0.2, 0.7, 0.6], sharex=axes_brs)
            plt.plot(t, PC[:,:].T)
            plt.xlim((t[0], t[-1]))
            sns.despine()
            plt.xlabel('Time (s)')
            plt.ylabel('PC')
        
        else:
            clrs = sns.color_palette("husl", ndim)
            
            d = (0.6 / ndim) * 0.3
            ny = (0.6 / ndim)-d
            for i in range(ndim):
                ax = plt.axes([0.2, 0.2+i*(ny+d), 0.7, ny], sharex=axes_exc)
                ax.plot(t, PC[ndim-1-i,:], color=clrs[i])    
                plt.xlim([t[0], t[-1]])
                sleepy.box_off(ax)
                plt.ylabel('PC%d' % (ndim-i))
    
                if add_laser:
                    ylims = ax.get_ylim()

                    for w in lsr_seq:
                        if w[-1] < len(M):
                            if 1 in M[w]:
                                color = 'b'
                            else:
                                color = 'r'
                            #if M[w[0]] == 3:
                            #plt.plot(t[w[0]], PC[ndim-1-i,w[0]-1:w[0]+1].mean(), '.', color=color, lw=2)
                            plt.plot(t[w[0]], PC[ndim-1-i,w[0]], '.', color=color, lw=2)
                            
                            laser_tend = (w[-1] - w[0]) * dt
                            dy = ylims[1] - ylims[0]
                            ax.add_patch(patches.Rectangle(
                                (t[w[0]], ylims[0]), laser_tend, dy, 
                                facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))

                if i > 0:
                    ax.spines["bottom"].set_visible(False)
                    ax.axes.get_xaxis().set_visible(False)
                else:
                    plt.xlabel('Time (s)')
                
                if len(ylim) > 0:
                    plt.ylim(ylim)
    
        var = S**2
        var_total = np.sum(S**2)
        
        # Calculate the cumulative variance explained, i.e. how much
        # of the variance is captured by the first i principal components. 
        p = []
        for i in range(1,len(var)+1):
            s = np.sum(var[0:i])
            p.append(s/var_total)
                
        plt.figure(figsize=(4,4))
        plt.plot(p, '.', color='gray')
        plt.xlabel(r'$\mathrm{PC_i}$')
        plt.ylabel('Cum. variance')    
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.ylim([0, 1.1])
        sns.despine()
        
    if len(kcuts) > 0:
        # find long wake blocks:
        widx = sleepy.get_sequences(np.where(Morig==2)[0], ibreak=int(wake_break/dt))
        tmp = []
        for w in widx:
            if len(w) * dt > wake_dur:
                tmp += list(w)
        widx = tmp
        
        nhypno = np.min((len(Morig), units.shape[0]))
        idx_total = np.arange(0, nhypno)
        
        idx_total = np.setdiff1d(idx_total, kidx)
        idx_total = np.setdiff1d(idx_total, widx)

    return PC, V, Vh, idx



def sleep_components_fine(ids, wake_dur=600, wake_break=60, ndim=3, nsmooth=0, 
                     pzscore=False, detrend=False, pplot=True, pc_sign=[], 
                     kcuts=[], mode='w', ppath='', mouse='', collapse=False,
                     ylim=[], config_file='mouse_config.txt'):
    dt = 2.5  
    NDOWN = 100
    NUP = int(dt / (0.001 * NDOWN))
    
    fine_scale = False

    if len(config_file) == 0:
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]

    ###########################################################################
    fine_scale = True
    
    tr_path = load_config(config_file)[mouse]['TR_PATH']
    units = np.load(os.path.join(tr_path,'1k_train.npz')) 
    unitIDs = [unit for unit in list(units.keys()) if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    dt = dt / NUP
    M = upsample_mx(M, NUP)
    nhypno = int(np.min((len(M), units[unitIDs[0]].shape[0]/NDOWN)))

    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
    
    nsample = nhypno           # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample))
    #@tidx are the indices we're further considereing.
    
    
    print('Starting downsampling, smoothing and z-scoring')

    if fine_scale:
        fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % NDOWN)
        if not os.path.isfile(fr_file):
            for i,unit in enumerate(unitIDs):
                tmp = sleepy.downsample_vec(np.array(units[unit]), NDOWN)            
                R[i,:] = tmp[tidx]
            so.savemat(fr_file, {'R':R, 'ndown':NDOWN})
        else:
            R = so.loadmat(fr_file, squeeze_me=True)['R']
            
        for i,unit in enumerate(unitIDs):
            tmp = R[i,:]
            tmp = sleepy.smooth_data(tmp, nsmooth)
            if pzscore:
                R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
            else:
                R[i,:] = tmp[tidx]

    print('Starting downsampling, smoothing and z-scoring')

    if fine_scale:
        fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % NDOWN)
        if not os.path.isfile(fr_file):
            for i,unit in enumerate(unitIDs):
                tmp = sleepy.downsample_vec(np.array(units[unit]), NDOWN)            
                R[i,:] = tmp[tidx]
            so.savemat(fr_file, {'R':R, 'ndown':NDOWN})
        else:
            R = so.loadmat(fr_file, squeeze_me=True)['R']
            
        for i,unit in enumerate(unitIDs):
            tmp = R[i,:]
            tmp = sleepy.smooth_data(tmp, nsmooth)
            if pzscore:
                R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
            else:
                R[i,:] = tmp[tidx]


    # find long wake blocks:
    widx = sleepy.get_sequences(np.where(M==2)[0], ibreak=int(wake_break/dt))
    # all REM sequences
    ridx = sleepy.get_sequences(np.where(M==1)[0])

    nidx = sleepy.get_sequences(np.where(M==3)[0])

    tmp = []
    for w in widx:
        if len(w) * dt > wake_dur:
            tmp += list(w)            
    widx = tmp

    tmp = []
    for r in ridx:
        tmp += list(r)            
    ridx = tmp

    tmp = []
    for r in nidx:
        tmp += list(r)            
    nidx = tmp

    nhypno = np.min((len(M), units.shape[0]))
    idx = np.arange(0, np.min((len(M), units.shape[0])))
    if 'w' in mode:
        idx = np.setdiff1d(idx, widx)
    else:
        widx = []
    if 'r' in mode:
        idx = np.setdiff1d(idx, ridx)
    if 'n' in mode:
        idx = np.setdiff1d(idx, nidx)
    else:
        ridx = []
                    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R[:,idx].T / np.sqrt(nsample-1)
    # make sure the the columns of Y are mean-zero
    for i in range(nvar):
        Y[:,i] = Y[:,i] - Y[:,i].mean()
        R[i,:] = R[i,:] - R[i,idx].mean()
    
    # SVD
    # Note that each column is a neuron, 
    # and each row is a time point
    U,S,Vh = scipy.linalg.svd(Y)
    
    # each row in Vh is an eigenvector of the COV matrix;
    # to get the PCs, we project R onto the eigenvectors:
    PC = np.dot(Vh, R)[0:ndim,:]
    V = S**2
    
    if len(pc_sign) > 0:
        i = 0
        for s in pc_sign:
            PC[i,:] = PC[i,:] * s
            i += 1


    ## add figure
    if pplot:
        t = np.arange(0, nhypno) * dt

        plt.figure()
        axes_brs = plt.axes([0.2, 0.85, 0.7, 0.05])        
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        sleepy._despine_axes(axes_brs)

        if collapse:    
            plt.axes([0.2, 0.2, 0.7, 0.6], sharex=axes_brs)
            plt.plot(t, PC[:,:].T)
            plt.xlim((t[0], t[-1]))
            sns.despine()
            plt.xlabel('Time (s)')
            plt.ylabel('PC')

    


    return PC, V



def kcut_idx(M, X, kcuts, dt=2.5):
    """
    Using the values defined in the mouse_config.txt file (field KCUT:), determine
    the indices used for further calculations.
    
    @param: np.array, brainstate sequence
    @param X: np.pandas or np.array, array or DataFrame with time axis using same binning as @M
    @param kcuts: list of tuples, areas at the beginning or end that should be discarded

    @return tidx: np.array, list of indices in @M used for further calculation, i.e. indices 
            that are NOT within the ranges defined in @kcuts.
    """

    #n = np.max(X.shape)
    #nhypno = np.min((len(M), n))
    nhypno = len(M)    
    #M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
                
    return tidx



def kcut_idx2(M, kcuts, X = [], dt=2.5):
    """
    Using the values defined in the mouse_config.txt file (field KCUT:), determine
    the indices used for further calculations.
    
    @param: np.array, brainstate sequence
    @param X: np.pandas or np.array, array or DataFrame with time axis using same binning as @M
    @param kcuts: list of tuples, areas at the beginning or end that should be discarded

    @return tidx: np.array, list of indices in @M used for further calculation, i.e. indices 
            that are NOT within the ranges defined in @kcuts.
    """


    nhypno = len(M)    
    if len(X) > 0:
        n = np.max(X.shape)
        nhypno = np.min((len(M), n))
        
    tidx = np.arange(0, nhypno)
    
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
                
    return tidx


   
def align_pcsign(PC, mouse, rem_pca=0, kcuts=[], align_pc3=True, config_file='', pnorm_spec=True):
    """
    Automatically determine the sign of the given PCs. 
    Fix PC1 (normally increased during REM) though activity during REM sleep.
    Fix PC2 (normally positively correlated with sigma power) through its
    correlation with the sigma power.
    Fix PC3...PCn the same way as PC1.

    Parameters
    ----------
    PC : np.array
        Matrix with PCs; each row corresponds to one PC.
    mouse : str
        mouse name.
    rem_pca : int, optional
        0 or 1. If 0, assume that PC1 (PC[0,:]) is the "REM-PC".
    kcuts : list of tuples
        Areas at beginning or end to remove from recording. The default is [].
    align_pc3 : bool, optional
        If True, adjust sign of PC3, ... PCn using the same strategy as for PC1
    config_file : str, optional
        File name of mouse config file. The default is ''.
    pnorm_spec : bool, optional
        If True, normalize EEG spectrogram. The default is True.

    Returns
    -------
    pc_sign : list of length PC.shape[0]
        1 or -1 depending on whether the orientation of the PC should be changed or not.

    """    
    ndim = PC.shape[0]        
    pc_sign = [1 for i in range(ndim)]
    
    if len(config_file) == 0:        
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]
                            
    sigma = [10,15]
    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat'%name), squeeze_me=True)
    SP = P['SP']
    freq  = P['freq']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)

    if tidx[-1] > PC.shape[1]-1:
        tidx = tidx[0:-1]

    M = M[tidx]
    
    #PC = PC[:,tidx]
    SP = SP[:,tidx]
    ################################    
    
    if SP.shape[1] != PC.shape[1]:
        print('Check kcut!!!')
        print('Shape of SP = %d; shape of PC = %d' % (SP.shape[1], PC.shape[1]))
                            
    if rem_pca == 0:
        sigma_pca = 1
    else:
        sigma_pca = 0
        
    # fix PC1
    mmin = np.min((len(M), PC.shape[1]))
    M = M[0:mmin]
    PC = PC[:,0:mmin]
    
    pc1 = PC[rem_pca,:]
    rem_idx = np.where(M==1)[0]
    a = pc1[rem_idx].mean()
    if a < 0:
        pc_sign[rem_pca] = -1
                
    # fix PC3...PCn the same way
    if align_pc3:
        if PC.shape[0] >= 3:
            for j in range(2, PC.shape[0]):
                pc3 = PC[j,:]
                rem_idx = np.where(M==1)[0]
                a = pc3[rem_idx].mean()
                if a < 0:
                    pc_sign[j] = -1
    
    # fix PC2 through correlation with sigma power
    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        sigma_pow = SP[isigma,:].mean(axis=0)
    else:
        sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
        
    CC, t = state_correlation(PC[sigma_pca,:], sigma_pow, M, win=60, pplot=False)
    
    CC = np.nanmean(CC, axis=0)
    i = np.argmax(np.abs(CC))
    
    if CC[i] < 0:
        pc_sign[sigma_pca] = -1

    return pc_sign
    
    
    
def plot_pcs_withsigma(PC, M, mouse, ndim=2, pc_sign=[], kcuts=[], 
                       ma_thr=10, ma_rem_exception=False,
                       tstart=0, tend=-1,
                       dt=2.5, sigma=[10,15], fmax=20, vm=[],
                       box_filt=[], pnorm_spec=True, 
                       reverse_pcs=False, tlegend=120, zoomin=[], pplot=True, config_file=''):
    """
    Plot principal components along with sigma power.

    Parameters
    ----------
    PC : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    ndim : TYPE, optional
        DESCRIPTION. The default is 2.
    pc_sign : TYPE, optional
        DESCRIPTION. The default is [].
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is False.
    tstart : TYPE, optional
        DESCRIPTION. The default is 0.
    tend : TYPE, optional
        DESCRIPTION. The default is -1.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    sigma : TYPE, optional
        DESCRIPTION. The default is [10,15].
    fmax : TYPE, optional
        DESCRIPTION. The default is 20.
    vm : list, optional
        List with two elements defining vmin and vmax for the EEG spectrogram colormap. 
        If [], matplotlib will automatically set the color range.
    box_filt : list, optional
        Dimensions of box filter to smooth EEG spectrogram. If [], no filteringThe default is [].
    pnorm_spec : bool, optional
        If True, normalize spectrogram. 
    reverse_pcs : TYPE, optional
        DESCRIPTION. The default is False.
    tlegend : TYPE, optional
        DESCRIPTION. The default is 120.
    zoomin : TYPE, optional
        DESCRIPTION. The default is [].
    pplot : TYPE, optional
        DESCRIPTION. The default is True.
    config_file : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    PC_orig : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    sigma_pow : TYPE
        DESCRIPTION.

    """

    if len(config_file) == 0:        
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]
    
    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat'%name), squeeze_me=True)
    SP = P['SP']
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)
    tidx = tidx[0:PC.shape[1]]

    M = M[tidx]
    #PC = PC[:,tidx]
    SP = SP[:,tidx]
    
    if not(len(M) == SP.shape[1] == PC.shape[1]):        
        print('Something went wrong with KCUT')
        print('returning')
        return
    ################################
    
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################
    
    freq  = P['freq']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]
 
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        sigma_pow = SP[isigma,:].mean(axis=0)
    else:
        sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
        
    istart = int(tstart/dt)
    if tend == -1:
        iend = len(M)
    else:
        iend = int(tend/dt)
        
    # cut out time interval istart - iend:
    M = M[istart:iend]
    PC = PC[:,istart:iend]
    PC_orig = PC.copy()
    PC_orig = PC_orig[istart:iend]
    sigma_pow = sigma_pow[istart:iend]
    t = np.arange(0, len(M))*dt
    
    if pplot:
        # plot figure
        plt.figure()
        # show brainstate
        axes_brs = plt.axes([0.1, 0.9, 0.8, 0.05])
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)

        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        sleepy._despine_axes(axes_brs)

        # show EEG spectrogram
        axes_spec = plt.axes([0.1, 0.72, 0.8, 0.15], sharex=axes_brs)
        # axes for colorbar
        axes_cbar = plt.axes([0.9, 0.72, 0.05, 0.15])
        
        # calculate median for choosing right saturation for heatmap
        med = np.median(SP.max(axis=0))
        if len(vm) == 0:
            vm = [0, med*2.0]
        ifreq = np.where(freq <= fmax)[0]
        im = axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq, istart:iend], cmap='jet', vmin=vm[0], vmax=vm[1])
        axes_spec.axis('tight')
        axes_spec.set_xticklabels([])
        axes_spec.set_xticks([])
        axes_spec.spines["bottom"].set_visible(False)
        axes_spec.set_ylabel('Freq (Hz)')
        sleepy.box_off(axes_spec)
        axes_spec.set_xlim([t[0], t[-1]])

        # colorbar for EEG spectrogram
        cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0)
        if pnorm_spec:
            cb.set_label('Norm. power')
        else:
            cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
        #if len(cb_ticks) > 0:
        #    cb.set_ticks(cb_ticks)
        axes_cbar.set_alpha(0.0)
        sleepy._despine_axes(axes_cbar)
        #im = axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq, istart:iend], cmap='jet', vmin=vm[0], vmax=vm[1])
        
        # show sigmapower
        axes_sig = plt.axes([0.1, 0.57, 0.8, 0.1], sharex=axes_brs)
        plt.plot(t, sigma_pow, color='gray')
        plt.xlim([t[0], t[-1]])
        plt.ylabel('$\mathrm{\sigma}$ power')

        axes_sig.spines["top"].set_visible(False)
        axes_sig.spines["right"].set_visible(False)
        axes_sig.spines["bottom"].set_visible(False)
        axes_sig.axes.get_xaxis().set_visible(False)

        a = np.percentile(sigma_pow, 99.5)
        plt.ylim([0, a+a*0.1])

        # show PCs
        for i in range(ndim):
            PC[i,:] = PC[i,:] - np.min(PC[i,:])

        pc_max = []
        for i in range(ndim):
            p = np.max(PC[i,:])
            pc_max.append(p)

        mmax = np.max(np.array(pc_max))
        pos = [0]
        for i in range(1,ndim):
            if reverse_pcs:
                PC[i,:] = PC[i,:] + i*mmax 
                pos.append(i*mmax)
            else:
                PC[i,:] = PC[i,:] - i*mmax 
                pos.append(-i*mmax)

        # axes for PCs
        axes_pcs = plt.axes([0.1,0.05,0.8,0.5], sharex=axes_brs)
        # colors for PCs
        cmap = sns.color_palette("husl", ndim)
        for i in range(ndim):
            axes_pcs.plot(t, PC[i,:], c=cmap[i])    
            plt.text(t[-1], pos[i], 'PC%d' % (i+1), fontsize=14, color=cmap[i])
        plt.xlim((t[0], t[-1]))

        axes_pcs.spines["left"].set_visible(False)
        axes_pcs.spines["right"].set_visible(False)
        axes_pcs.axes.get_yaxis().set_visible(False)
        sleepy._despine_axes(axes_pcs)

        if len(zoomin) > 0:
            zoomin = [int(z/dt) for z in zoomin]
            for z in zoomin:
                pos = np.array(pos)
                plt.plot([t[z], t[z]], [mmax, -(ndim-1)*mmax], 'k--')

        # axes for time legend
        axes_legend = plt.axes([0.1,0.04,0.8,0.04], sharex=axes_brs)
        plt.plot([0, tlegend], [1, 1], lw=2, color='k')
        plt.ylim([-1, 1])
        axes_legend.text(0, -2, '%d s' % tlegend, verticalalignment='bottom', horizontalalignment='left')

        plt.xlim((t[0], t[-1]))
        sleepy._despine_axes(axes_legend)
    
    return PC_orig, M, sigma_pow



def sleep_subspaces(units, mouse, ndim=2, nsmooth=0, ma_thr=10, ma_rem_exception=False,
                    pzscore=False, detrend=False, pplot=True, coords=[], traj_mode='full', trig_state=1, 
                    local_rotation=True, pspec=True,
                    kcuts=[], proj_3d=False, config_file=''):
    """
    Calculate (and plot) PCA separately for each state (REM, Wake, NREM). 

    Parameters
    ----------
    units : np.DataFrame or []
        If [], use 1 ms spike trains to calculate firing rates.
    mouse : TYPE
        DESCRIPTION.
    ndim : int, optional
        Number of PCs to keep for dimensionality reduction. 
        The default is 2.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    ma_thr : float, optional
        Wake sequences <= $ma_thr s are interpreted as NREM (3). The default is 10.
    ma_rem_exception : bool, optional
        If True, then the MA rule does not apply for wake episodes directly following REM. 
        The default is False.
    pzscore : TYPE, optional
        DESCRIPTION. The default is False.
    pplot : TYPE, optional
        DESCRIPTION. The default is True.
    coords : list, optional
        Specific the two PCs that should be shown on the x and y-axis. 
        PC1 corresponds to "0".
        The default is [].
    traj_mode : string, optional
        If 'full', show complete trajectory. 
        If 'trig', only show trajectories for the specific state transitions (-> trig_state)
        The default is 'full'.
    trig_state : int, optional
        1,2, or 3. If traj_mode == 'trig', only show the the transitions to state $trig_state 
        instead of the full trajectories across the whole recording session
        The default is 1.
    local_rotation : bool, optional
        If True, rotate the ndim dimensional space (by performing another PCA) and
        use the resulting first two dimensions to get the PCs.
        If True, the parameter @coords has no effect.
        The default is True.
    pspec : TYPE, optional
        DESCRIPTION. The default is True.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    proj_3d : bool, optional
        If True, plot 3D subspaces
    config_file : str
        file name of mouse configuration file, as loaded by &load_config()
    

    Returns
    -------
    pc_dict : TYPE
        DESCRIPTION.
    vh_dict : TYPE
        DESCRIPTION.
    idx_dict : TYPE
        DESCRIPTION.

    """    
    dt = 2.5  
    NDOWN = 500
    NUP = int(dt / (0.001 * NDOWN))
    
    fine_scale = False

    if len(config_file) == 0:
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]

    ###########################################################################
    if len(units) == 0:
        fine_scale = True
        
        tr_path = load_config(config_file)[mouse]['TR_PATH']
        units = np.load(os.path.join(tr_path,'1k_train.npz')) 
        unitIDs = [unit for unit in list(units.keys()) if '_' in unit]
        unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

        dt = dt / NUP
        M = upsample_mx(M, NUP)
        nhypno = int(np.min((len(M), units[unitIDs[0]].shape[0]/NDOWN)))

    else:
        unitIDs = [unit for unit in units.columns if '_' in unit]
        unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
        nhypno =         np.min((len(M), units.shape[0]))

    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    # OLD:
    # if len(kcut) > 0:
    #     kidx = np.arange(int(kcut[0]/dt), int(kcut[-1]/dt))
    #     tidx = np.setdiff1d(tidx, kidx)
    #     M = M[tidx]
    #     nhypno = len(tidx)
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
    #print(len(tidx))
    
    nsample = nhypno           # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample))
    #@tidx are the indices we're further considereing.
    
    
    print('Starting downsampling, smoothing and z-scoring')

    if fine_scale:
        fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % NDOWN)
        if not os.path.isfile(fr_file):
            for i,unit in enumerate(unitIDs):
                tmp = sleepy.downsample_vec(np.array(units[unit]), NDOWN)            
                R[i,:] = tmp[tidx]
            so.savemat(fr_file, {'R':R, 'ndown':NDOWN})
        else:
            R = so.loadmat(fr_file, squeeze_me=True)['R']
            
        for i,unit in enumerate(unitIDs):
            tmp = R[i,:]
            tmp = sleepy.smooth_data(tmp, nsmooth)
            if pzscore:
                R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
            else:
                R[i,:] = tmp[tidx]
                
    if not fine_scale:
        for i,unit in enumerate(unitIDs):
            tmp = sleepy.smooth_data(np.array(units[unit]), nsmooth)
            tmp = tmp[tidx]

            if detrend:
                tmp = scipy.signal.detrend(tmp)

            if pzscore:
                #R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
                R[i,:] = (tmp - tmp.mean()) / tmp.std()
            else:
                #R[i,:] = tmp[tidx]
                R[i,:] = tmp

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 4
    
    # all REM sequences
    ridx = sleepy.get_sequences(np.where(M==1)[0])
    # find long wake blocks:
    widx = sleepy.get_sequences(np.where(M==2)[0])
    # all NREM sequences    
    nidx = sleepy.get_sequences(np.where(M>=3)[0])
    
    midx = sleepy.get_sequences(np.where(M==4)[0])
    
    ridx = [list(a) for a in ridx]
    widx = [list(a) for a in widx]
    nidx = [list(a) for a in nidx]
    midx = [list(a) for a in midx]
    
    ridx = sum(ridx, [])
    widx = sum(widx, [])
    nidx = sum(nidx, [])
    midx = sum(midx, [])
    idx_dict = {'REM':ridx, 'Wake':widx, 'NREM':nidx, 'MA':midx}

    pc_dict = {}
    vh_dict = {}
    labels = ['REM', 'Wake', 'NREM', 'MA']
    for idx,label in zip([ridx, widx, nidx], labels):
        # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
        Y = R[:,idx].T / np.sqrt(nsample-1)
        # make sure the the columns of Y are mean-zero
        for i in range(nvar):
            Y[:,i] = Y[:,i] - Y[:,i].mean()
            # Note the PCs are orthogonal to each other, but
            # only for the segments for which we calculate PCA
            # and only if we center the firing rates across these
            # segments!
            R[i,:] = R[i,:] - R[i,idx].mean()
            #R[i,:] = R[i,:] - R[i,:].mean()
        
        # SVD
        U,S,Vh = scipy.linalg.svd(Y)
        
        # each row in Vh is an eigenvector of the COV matrix:    
        PC = np.dot(Vh, R)[0:ndim,:]

        if local_rotation:
            if not proj_3d:
                PC = pca(PC.copy().T, dims=2)[0].T
                coords = [0,1]            
            else:
                PC = pca(PC.copy().T, dims=3)[0].T
                coords = [0,1,2]            
                
        pc_dict[label] = PC
        vh_dict[label] = Vh[0:ndim,:]
    
    if coords == []:
        coords = list(range(ndim))

    if pplot:

        sleepy.set_fontsize(12)
        plt.figure(figsize=(12,10))

        t = np.arange(0, len(M))*dt

        if pspec:
            # load spectrogram
            tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
            SP = tmp['SP']
            freq = tmp['freq']
            ifreq = np.where(freq < 30)[0]
    
            axes_spec = plt.axes([0.4, 0.85, 0.55, 0.08])
            axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq,:], vmin=0, vmax=2000, cmap='jet')
            sleepy._despine_axes(axes_spec)
        
            axes_brs = plt.axes([0.4, 0.8, 0.55, 0.025], sharex=axes_spec)
            
        else:
            axes_brs = plt.axes([0.4, 0.8, 0.55, 0.025])
                    
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        axes_brs.axes.get_xaxis().set_visible(False)
        axes_brs.axes.get_yaxis().set_visible(False)
        axes_brs.spines["top"].set_visible(False)
        axes_brs.spines["right"].set_visible(False)
        axes_brs.spines["bottom"].set_visible(False)
        axes_brs.spines["left"].set_visible(False)
                        
        ax1 = plt.axes([0.4, 0.6, 0.55, 0.2], sharex=axes_brs)
        sns.despine()
        ax2 = plt.axes([0.4, 0.35, 0.55, 0.2], sharex=ax1)
        sns.despine()
        ax3 = plt.axes([0.4, 0.1, 0.55, 0.2], sharex=ax1)
        sns.despine()
        
        axes = [ax1, ax2, ax3]
        i = 0
        for ax, label in zip(axes, pc_dict):
            ax.plot(t, pc_dict[label][coords].T)
            ax.set_xlim([t[0], t[-1]])
            if i < 2:
                #ax.set_xticklabels([])
                pass
            else:
                ax.set_xlabel('Time (s)')
            i += 1
        
        if not proj_3d:        
            ax1 = plt.axes([0.1, 0.6, 0.2, 0.2])
            sns.despine()
            ax2 = plt.axes([0.1, 0.35, 0.2, 0.2])
            sns.despine()
            ax3 = plt.axes([0.1, 0.1, 0.2, 0.2])
            sns.despine()
        else:
            ax1 = plt.axes([0.1, 0.6, 0.2, 0.2], projection='3d')
            sns.despine()
            ax2 = plt.axes([0.1, 0.35, 0.2, 0.2], projection='3d')
            sns.despine()
            ax3 = plt.axes([0.1, 0.1, 0.2, 0.2], projection='3d')
            sns.despine()
            
        
        axes = [ax1, ax2, ax3]
        clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0.2, 0.2]]
        if traj_mode == 'full':
            for ax, label in zip(axes, pc_dict):
                PC = pc_dict[label]
                
                
                p = M[0]
                k = 0
                kold = k
                while k < len(M)-1:
                    while M[k] == p and k < len(M)-1:
                        k+=1
                    if not proj_3d:
                        ax.plot(PC[coords[0],kold:k], PC[coords[1], kold:k], color=clrs[int(p)], lw=0.5)
                    else:
                        ax.plot3D(PC[coords[0],kold:k], PC[coords[1], kold:k], PC[coords[2], kold:k], color=clrs[int(p)], lw=0.5)                        
                    p = M[k]
                    kold = k-1
                
                if local_rotation:
                    ax.set_xlabel("PC1'")
                    ax.set_ylabel("PC2'")
                else:
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")                    
                
        else:        
            for ax, label in zip(axes, pc_dict): 
                PC = pc_dict[label]
                plot_trajectories(PC, M, 300, 0, dt=dt, min_dur=20, istate=trig_state, pre_state=2, 
                                     state_num=[], ma_thr=20, kcuts=(), coords=coords, ax=ax, lw=0.5)    
                
                if local_rotation:
                    ax.set_xlabel("PC1'")
                    ax.set_ylabel("PC2'")
                else:
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                            
        
    return pc_dict, vh_dict, idx_dict



def subspace_mixing_mx(units, mouse, ndim=2, nsmooth=0, ma_thr=10, ma_rem_exception=False,
                    pzscore=False, detrend=False, pplot=True, coords=[], traj_mode='full', trig_state=1, 
                    pspec=True, kcuts=[], proj_3d=False, config_file=''):
    """
    Determine coordindates systems for REM, NREM, and Wake subspace. 
    Then, project the firing rate vectors for REM, NREM, and Wake into each subspace 
    and determine the variance of the projected data. 

    See also function &sleep_subspaces()

    Calculate (and plot) PCA separately for each state (REM, Wake, NREM). 

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    ndim : int, optional
        Number of PCs to keep for dimensionality reduction. 
        The default is 2.
    nsmooth : float, optional
        Smooth firing rates using sleepy.smooth_data(data, nsmooth). The default is 0.
    ma_thr : float, optional
        Wake sequences <= $ma_thr s are interpreted as NREM (3). The default is 10.
    ma_rem_exception : bool, optional
        If True, then the MA rule does not apply for wake episodes directly following REM. 
        The default is False.
    pzscore : TYPE, optional
        DESCRIPTION. The default is False.
    pplot : TYPE, optional
        DESCRIPTION. The default is True.
    coords : list, optional
        Specific the two PCs that should be shown on the x and y-axis. 
        PC1 corresponds to "0".
        The default is [].
    traj_mode : string, optional
        If 'full', show complete trajectory. 
        If 'trig', only show trajectories for the specific state transitions (-> trig_state)
        The default is 'full'.
    trig_state : int, optional
        1,2, or 3. If traj_mode == 'trig', only show the the transitions to state $trig_state 
        instead of the full trajectories across the whole recording session
        The default is 1.
    pspec : TYPE, optional
        DESCRIPTION. The default is True.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    proj_3d : bool, optional
        If True, plot 3D subspaces
    config_file : str
        file name of mouse configuration file, as loaded by &load_config()
    

    Returns
    -------
    pc_dict : dict
        dict: state --> PCs.
    vh_dict : TYPE
        DESCRIPTION.
    idx_dict : TYPE
        DESCRIPTION.

    """    
    dt = 2.5  

    if len(config_file) == 0:
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]

    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nhypno =         np.min((len(M), units.shape[0]))

    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
        
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
        #@tidx are the indices we're further considereing.
    ###########################################################################    

    
    nsample = nhypno           # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample))    
    # R: each row is one neuron with firing rates for len(tidx) time points
    for i,unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]), nsmooth)
        tmp = tmp[tidx]
        if detrend:
            tmp = scipy.signal.detrend(tmp)

        if pzscore:
            R[i,:] = (tmp - tmp.mean()) / tmp.std()
        else:
            R[i,:] = tmp

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    
    # all REM sequences
    ridx = sleepy.get_sequences(np.where(M==1)[0])
    # find long wake blocks:
    widx = sleepy.get_sequences(np.where(M==2)[0])
    # all NREM sequences    
    nidx = sleepy.get_sequences(np.where(M>=3)[0])
    
    midx = sleepy.get_sequences(np.where(M==4)[0])
    
    ridx = [list(a) for a in ridx]
    widx = [list(a) for a in widx]
    nidx = [list(a) for a in nidx]
    midx = [list(a) for a in midx]
    
    ridx = sum(ridx, [])
    widx = sum(widx, [])
    nidx = sum(nidx, [])
    midx = sum(midx, [])
    idx_dict = {'REM':ridx, 'Wake':widx, 'NREM':nidx, 'MA':midx}

    pc_dict = {}
    vh_dict = {}
    cc = np.sqrt(nsample-1)
    cc = 1
    labels = ['REM', 'Wake', 'NREM']
    for idx,label in zip([ridx, widx, nidx], labels):
        Rsub = R.copy()
        # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
        Y = R[:,idx].T / cc
        # make sure the the columns of Y are mean-zero
        for i in range(nvar):
            Y[:,i] = Y[:,i] - Y[:,i].mean()
            # Note the PCs are orthogonal to each other, but
            # only for the segments for which we calculate PCA
            # and only if we center the firing rates across these
            # segments!
            Rsub[i,:] = Rsub[i,:] - Rsub[i,idx].mean()            
        # SVD
        U,S,Vh = scipy.linalg.svd(Y)
        
        # each row in Vh is an eigenvector of the COV matrix:    
        PC = np.dot(Vh, Rsub)[0:ndim,:]
                
        pc_dict[label] = PC
        vh_dict[label] = Vh[0:ndim,:] * cc

    MX = np.zeros((len(labels), len(labels)))
    # Build mixing matrix
    for i in range(len(labels)):
        PCsub = pc_dict[labels[i]]

        # For example take NREM subspace and project
        # NREM, Wake and REM firing rates into this space, and
        # then calculate the total variance of NREM, Wake, and REM
        # neurons within this space

        for j in range(0, len(labels)):            
            state_idx = idx_dict[labels[j]]
            A = PCsub[:,state_idx]            
            MX[i,j] = np.trace(np.cov(A))

    if coords == []:
        coords = list(range(ndim))

    if pplot:

        sleepy.set_fontsize(12)
        plt.figure(figsize=(12,10))

        t = np.arange(0, len(M))*dt

        if pspec:
            # load spectrogram
            tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
            SP = tmp['SP']
            freq = tmp['freq']
            ifreq = np.where(freq < 30)[0]
    
            axes_spec = plt.axes([0.4, 0.85, 0.55, 0.08])
            axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq,:], vmin=0, vmax=2000, cmap='jet')
            sleepy._despine_axes(axes_spec)
        
            axes_brs = plt.axes([0.4, 0.8, 0.55, 0.025], sharex=axes_spec)
            
        else:
            axes_brs = plt.axes([0.4, 0.8, 0.55, 0.025])
                    
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        axes_brs.axes.get_xaxis().set_visible(False)
        axes_brs.axes.get_yaxis().set_visible(False)
        axes_brs.spines["top"].set_visible(False)
        axes_brs.spines["right"].set_visible(False)
        axes_brs.spines["bottom"].set_visible(False)
        axes_brs.spines["left"].set_visible(False)
                        
        ax1 = plt.axes([0.4, 0.6, 0.55, 0.2], sharex=axes_brs)
        sns.despine()
        ax2 = plt.axes([0.4, 0.35, 0.55, 0.2], sharex=ax1)
        sns.despine()
        ax3 = plt.axes([0.4, 0.1, 0.55, 0.2], sharex=ax1)
        sns.despine()
        
        axes = [ax1, ax2, ax3]
        i = 0
        for ax, label in zip(axes, pc_dict):
            ax.plot(t, pc_dict[label][coords,:].T)
            ax.set_xlim([t[0], t[-1]])
            if i < 2:
                #ax.set_xticklabels([])
                pass
            else:
                ax.set_xlabel('Time (s)')
            i += 1
        
        if not proj_3d:        
            ax1 = plt.axes([0.1, 0.6, 0.2, 0.2])
            sns.despine()
            ax2 = plt.axes([0.1, 0.35, 0.2, 0.2])
            sns.despine()
            ax3 = plt.axes([0.1, 0.1, 0.2, 0.2])
            sns.despine()
        else:
            ax1 = plt.axes([0.1, 0.6, 0.2, 0.2], projection='3d')
            sns.despine()
            ax2 = plt.axes([0.1, 0.35, 0.2, 0.2], projection='3d')
            sns.despine()
            ax3 = plt.axes([0.1, 0.1, 0.2, 0.2], projection='3d')
            sns.despine()
            
        
        axes = [ax1, ax2, ax3]
        clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0.2, 0.2]]
        if traj_mode == 'full':
            for ax, label in zip(axes, pc_dict):
                PC = pc_dict[label]
                
                
                p = M[0]
                k = 0
                kold = k
                while k < len(M)-1:
                    while M[k] == p and k < len(M)-1:
                        k+=1
                    if not proj_3d:
                        ax.plot(PC[coords[0],kold:k], PC[coords[1], kold:k], color=clrs[int(p)], lw=0.5)
                    else:
                        ax.plot3D(PC[coords[0],kold:k], PC[coords[1], kold:k], PC[coords[2], kold:k], color=clrs[int(p)], lw=0.5)                        
                    p = M[k]
                    kold = k-1
                
                
        else:        
            for ax, label in zip(axes, pc_dict): 
                PC = pc_dict[label]
                plot_trajectories(PC, M, 300, 0, dt=dt, min_dur=20, istate=trig_state, pre_state=2, 
                                     state_num=[], ma_thr=20, kcuts=(), coords=coords, ax=ax, lw=0.5)    
                
                                    
    return pc_dict, vh_dict, idx_dict, MX



def pc_reconstruction(units, cell_info, ndim=3, nsmooth=0, pnorm=True, pzscore=False, 
                      pc_sign=[], dt=2.5):

    
    unitIDs = [unit for unit in units.columns if '_' in unit]    
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample))
    
    # OLD:
    i = 0
    for unit in unitIDs:
        R[i,:] = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (R[i,:] - R[i,:].mean()) / R[i,:].std()
        i += 1
    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R.T / np.sqrt(nsample-1)

    U,S,Vh = scipy.linalg.svd(Y)

    PC = np.dot(Vh, R)[0:ndim,:]
    plt.figure()
    plt.plot(PC.T)
    
    # reconstruct neural responses with the first $ndim eigenvectors
    SM = np.zeros((nsample, ndim))
    for i in range(ndim):
        SM[i,i] = S[i]

    # That's the reconstruction:
    Yhat = (np.dot(U[:,0:ndim], np.dot(SM[0:ndim,:], Vh[0:ndim,:]))) * np.sqrt(nsample-1)

    # Alternative:
    #      neurons x dim  *  [dim x neurons *  neurons x time]       
    # Yhat2 = np.dot(Vh[0:ndim,:].T, np.dot(Vh[0:ndim,:], R)).T

    C = Vh[0:ndim,:].T
    if pnorm:
        for i in range(C.shape[0]):
            C[i,:] = C[i,:] / np.sqrt(np.sum(C[i,:]**2))

    if len(pc_sign) > 0:
        i = 0
        for s in pc_sign:
            C[:,i] = C[:,i] * s
            i += 1

    labels = ['c'+str(i) for i in range(1, ndim+1)]
    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs

    labels = ['c'+str(i) for i in range(1, ndim+1)]
    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs
    #df['brain_region'] = list(cell_info[cell_info.ID.isin(unitIDs)]['brain_region'])
    df['brain_region'] = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in unitIDs]

    plt.figure()    
    plt.subplot(211)
    sns.histplot(data=df, x='brain_region', y='c1')
    plt.subplot(212)
    sns.histplot(data=df, x='brain_region', y='c2')
    
    return C, unitIDs, df, Yhat



def pc_reconstruction2(units, cell_info, time_idx=[], ndim=3, nsmooth=0, detrend=False, pnorm=True, pzscore=False, 
                      pc_sign=[], dim_reconstr=[], dt=2.5, kcuts=[], pearson=False, pplot=True, sign_plot=True):
    """
    Use whole time axis for smoothing and z-scoring.
    Calculate SVD only using time points in @time_idx and reconstruct firing rates
    only for timepoints in @time_idx.
    When calculating the coefficients for each PC this function
    takes only time points in time_idx into account!

    Note on SVD:
    Assume Y is a matrix with each column corresponding to the firing rate vector
    of one recorded unit (Y ~ time bins x units).
    Assume
        U,S,Vh = scipy.linalg.svd(Y)
    is the SVD of matrix Y
    
    Then, PC = U * S are the PCs;
    PC[i,:] is the i-th PC
    
    Vh[i,:] are the coefficients of each neurons for PCi
    
    The firing rates of unit fr_i can be reconstructed using,
    
        fr_i = PC1 * c[0,i] + PC2 * c2[1,i] + ...
    
    which we rewrite as
    
        fr_i = PC1 * c1_i + PC2 * c2_i + ...
    


    Parameters
    ----------
    units : pd.DataFrame
        Each column is a unit, with the unit ID (str) as column name.
    cell_info : pd.DataFrame
        Columns 'ID' lists all units, 'brain_region' describes the brain region of each unit.
    time_idx : list or array of indices, optional
        Indices used (after kcut) to calculate PCs. If [], use all indices.
    ndim : int, optional
        Number of dimensions to reconstruct firing rates. The default is 3.
    nsmooth : float, optional
        Smoothing factor for firing rates using Gaussian kernel. The default is 0.
    pnorm : bool, optional
        If True, normalize PC coefficients. 
    pzscore : bool, optional
        If True, z-score firing rates. The default is False.
    pc_sign : list or np.array, optional
        Describes the sign of the PCs. PCi is multplied by pc_sign[i-1]
    dt : float, optional
        Time bin for firing rates. The default is 2.5.
    kcuts : tuple/list, optional
        list of tuples.
        Discard the time interval ranging from kcuts[i][0] to kcuts[i][1] seconds
    sign_plot: bool
        It True, plot PCs after multiplying with pc_sign
    pearson: bool
        If True, calculate pearson correlation between each PC and firing rate. 
        The r value and p value are reported in the returned DataFrame df;
        columns r1, r2,... and p1, p2, ...


    Returns
    -------
    C : np.array with dimension: number of units x $ndim
        Coefficient c1_i, c_i, ... for each unit i.
        DESCRIPTION.
    df : pd.DataFrame with columns ['c1', 'c2', 'ID', 'r1', 'p1', 'r2', 'p2']
        c1, c2, ... are the coefficient c1, c2 for each PC. ID are the unit IDs
        p1, p2, ... and r1, r2, ... are the p-values and r coefficients, when fitting
        the firing rates using PC1, PC2, ... using linear regression.
    units : pd.DataFrame
        Original firing rates, each columns corresponds to one unit.
    units_hat : pd.DataFrame
        Reconstructed firing rates, each colums corresponds to one unit.

    """
    nhypno = units.shape[0]
    tidx = np.arange(0, nhypno)
    
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        nhypno = len(tidx)
    ###########################################################################    
            
    unitIDs = [unit for unit in units.columns if '_' in unit]    
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nhypno)) # dimensions: number of units x time bins
    
    for i,unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        tmp = tmp[tidx]
        if detrend:
            tmp = scipy.signal.detrend(tmp)

        if pzscore:
            R[i,:] = (tmp - tmp.mean()) / tmp.std()
        else:
            R[i,:] = tmp
    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    R = R[:,time_idx]
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = R.T / np.sqrt(nsample-1)
    
    U,S,Vh = scipy.linalg.svd(Y)
    PC = np.dot(Vh, R)[0:ndim,:]
    
    if len(pc_sign) == []:
        pc_sign = np.ones((ndim,))

    if sign_plot:
        plt.figure()
        for i in range(ndim):
            if i < len(pc_sign):
                plt.plot(PC[i,:]*pc_sign[i])
            else:
                plt.plot(PC[i,:])
    
    # reconstruct neural responses with the first $ndim eigenvectors
    SM = np.zeros((nsample, ndim))
    for i in range(ndim):
        SM[i,i] = S[i]

    if len(dim_reconstr) == 0:
        dim_reconstr = range(0, ndim)

    # That's the reconstruction:
    Yhat = (np.dot(U[:,dim_reconstr], np.dot(SM[dim_reconstr,:], Vh[dim_reconstr,:]))) * np.sqrt(nsample-1)

    # Alternative:
    #      neurons x dim  *  [dim x neurons *  neurons x time]       
    # Yhat2 = np.dot(Vh[0:ndim,:].T, np.dot(Vh[0:ndim,:], R)).T

    # 11/08/23 added the factor '* np.sqrt(nsample-1)'
    C = Vh[0:ndim,:].T  #* np.sqrt(nsample-1)
    if pnorm:
        for i in range(C.shape[0]):
            C[i,:] = C[i,:] / np.sqrt(np.sum(C[i,:]**2))

    if len(pc_sign) > 0:
        i = 0
        for s in pc_sign:
            C[:,i] = C[:,i] * s
            i += 1

    labels = ['c'+str(i) for i in range(1, ndim+1)]
    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs

    labels = ['c'+str(i) for i in range(1, ndim+1)]
    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs

    if pplot:
        df['brain_region'] = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in unitIDs]
        plt.figure()    
        plt.subplot(211)
        sns.histplot(data=df, x='brain_region', y='c1')
        plt.subplot(212)
        sns.histplot(data=df, x='brain_region', y='c2')

    #convert Yhat to pd.DataFrame with rows of Yhat as columns
    units_hat = pd.DataFrame(data=Yhat, columns=unitIDs)
    units = pd.DataFrame(data=Y*np.sqrt(nsample-1), columns=unitIDs)
    
    if pearson:
        for j in range(ndim):
            p = []
            cc = []
            for i,ID in enumerate(unitIDs):
                r = R[i,:]        
                res = scipy.stats.pearsonr(r, PC[j,:])
                p.append(res.pvalue)
                cc.append(res.statistic)
            df['r' + str(j+1)] = cc
            df['p' + str(j+1)] = p
                    
    return C, df, units, units_hat



def partialpc_reconstruction(units, cell_info, time_idx, neuron_ids=[], ndim=3, nsmooth=0, 
                             pnorm=True, pzscore=False, pc_sign=[], fit_timeidx=True, pplot=True):
    """
    
    NOTE: In its current implementation, the function fits the PC coefficients 
    to the complete time axis, although the principal axes have been fit using
    a (potentially smaller) time range.

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    cell_info : TYPE
        DESCRIPTION.
    time_idx : TYPE
        DESCRIPTION.
    neuron_ids : TYPE, optional
        DESCRIPTION. The default is [].
    ndim : TYPE, optional
        DESCRIPTION. The default is 3.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    pnorm : TYPE, optional
        DESCRIPTION. The default is True.
    pzscore : TYPE, optional
        DESCRIPTION. The default is False.
    pc_sign : TYPE, optional
        DESCRIPTION. The default is [].
    fit_timeidx: boolean, optional
        If True, take for fit of firing rates using the PCs only time points in time_idx into account

    Returns
    -------
    C : TYPE
        DESCRIPTION.
    PC : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.
    units: pd.DataFrame
        smoothed and centered firing rates
    units_hat: pd.DataFrame
        fitted firing rates using $ndim PCs

    """

    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample))
    
    #all_units = False
    if neuron_ids == []:
        neuron_ids = unitIDs
        #all_units = True
    
    i = 0
    neuron_idx = []
    for unit in unitIDs:
        R[i,:] = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (R[i,:] - R[i,:].mean()) / R[i,:].std()
            
        if unit in neuron_ids:
            neuron_idx.append(i)
        i += 1
    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    R2 = R[neuron_idx,:]
    Y = R2[:,time_idx].T / np.sqrt(len(time_idx)-1)
    # make sure the the columns of Y are mean-zero
    for i in range(Y.shape[1]):
        Y[:,i] = Y[:,i] - Y[:,i].mean()
    
    U,S,Vh = scipy.linalg.svd(Y)

    # that's the projections of all neuron_idx neurons (rows) into the PCA space,
    # giving us the principal components
    PC = np.dot(Vh, R2)[0:ndim,:]

    # Now, for ALL neurons and time points the to optimal coefficients to
    # reconstruct the original firing rates using the $ndim PCs
    A = np.ones((PC.shape[1], ndim+1))
    A[:,0:ndim] = PC.T

    Rhat = np.zeros(R.shape)
    C = np.zeros((nvar, ndim+1))
    for i in range(nvar):
        r = R[i,:]       
        if fit_timeidx:
            w = np.linalg.lstsq(A[time_idx], r[time_idx])[0]        
        else:
            w = np.linalg.lstsq(A, r)[0]        
        C[i,:] = w
        
        # reconstruction of units using ndeim PCs
        rhat = np.dot(A, w)
        Rhat[i,:] = rhat
    
    C = C[:,0:ndim]
    #C = Vh[0:ndim,:].T
            
    if pnorm:
        for i in range(C.shape[0]):
            C[i,:] = C[i,:] / np.sqrt(np.sum(C[i,:]**2))
            
    if len(pc_sign) > 0:
        for i,s in enumerate(pc_sign):
            C[:,i] = C[:,i] * s


    labels = ['c'+str(i) for i in range(1, ndim+1)]

    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs

    labels = ['c'+str(i) for i in range(1, ndim+1)]
    df = pd.DataFrame(data=C, columns=labels)
    df['ID'] = unitIDs
    #df['brain_region'] = list(cell_info[cell_info.ID.isin(unitIDs)]['brain_region'])
    df['brain_region'] = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in unitIDs]

    ### FIGURE ################################################################
    if pplot:
        plt.figure()
        plt.subplot(211)
        sns.histplot(data=df, x='brain_region', y='c1')
        plt.subplot(212)
        sns.histplot(data=df, x='brain_region', y='c2')

    # reconstruct neural responses with the first $ndim eigenvectors
    #SM = np.zeros((nsample, ndim))
    #for i in range(ndim):
    #    SM[i,i] = S[i]

    # That's the reconstruction:
    #Yhat = (np.dot(U[:,0:ndim], np.dot(SM[0:ndim,:], Vh[0:ndim,:]))) * np.sqrt(nsample-1)

    # UNDER CONSTRUCTION    

    units_hat = pd.DataFrame(data=Rhat.T, columns=unitIDs)
    units = pd.DataFrame(data=R.T, columns=unitIDs)

    return C, PC, df, units, units_hat



def optimal_direction(dfc, pref_direction, thr=0, pplot=True, ax='', brain_region=False):
    """
    
    Parameters
    ----------
    dfc : pd.DataFrame
        DataFrame with coefficients, c1, ... c_n (loadings) for PCs.
        Columns: 'c1', ... 'c_n', 'ID', 'brain_region'
    pref_direction : TYPE
        DESCRIPTION.
    thr : TYPE
        DESCRIPTION.

    Returns
    -------
    df2 : TYPE
        DESCRIPTION.

    """
    
    pref_direction = pref_direction / scipy.linalg.norm(pref_direction)
    
    cols = ['c'+str(i+1) for i in range(pref_direction.shape[0])]
  
    data = []
    for index, row in dfc.iterrows():
        vec = np.array(row[cols])
        # normalize each vector
        vec = vec / scipy.linalg.norm(vec)        
        d = np.dot(vec, pref_direction)
        
        if brain_region:
            data += [[d, row['ID'], row['brain_region']]]
        else:
            data += [[d, row['ID']]]

    if brain_region:        
        df2 = pd.DataFrame(data=data, columns=['proj', 'ID', 'brain_region'])
    else:
        df2 = pd.DataFrame(data=data, columns=['proj', 'ID'])
    
    
    if thr > 0:
        dfs = dfc[(df2.proj > thr)]
    else:
        dfs = dfc[(df2.proj < thr)]
    
    if pplot:
        if ax == '':
            plt.figure()
            ax = plt.axes([0.2, 0.2, 0.7, 0.7])
            #ax.axis('equal')
                    
        ax.axhline(0, linestyle='--', color='k') # horizontal lines
        ax.axvline(0, linestyle='--', color='k') # vertical lines
        
        if thr != 0:
            sns.scatterplot(data=dfc, x='c1', y='c2', color='gray')
            sns.scatterplot(data=dfs, x='c1', y='c2', color='red')
        else:
            sns.scatterplot(data=dfc, x='c1', y='c2', hue='brain_region')
            
        plt.grid(False)
        sns.despine()
        
    return df2    
    


def laser_triggered_pcs(PC, pre, post, M, mouse, kcuts=[], min_laser=20, pzscore_pc=False, local_pzscore=True,
                        pplot=True, ci=None, refractory_rule=False, ma_thr=10, ma_rem_exception=False, rnd_laser=False, seed=1,
                        config_file='mouse_config.txt'):
    """
    Calculated the time course of the provided PCs relative to the laser onset.

    Parameters
    ----------
    PC : np.array
        Each row corresponds to one PC.
    pre : float
        Time before laser onset.
    post : float
        Time after laser onset.
    M : np.array
        Hypnogram.
    mouse : str
        Mouse name.
    kcuts : list of tuples, optional
        DESCRIPTION. The default is [].
    min_laser : float, optional
        Minimum duration of laser train. If laser duration < $min_laser, 
        disregard the laser trial.
    pzscore_pc : bool, optional
        If true, z-score the PCs (across entire recording)
    local_pzscore : bool, optional
        DESCRIPTION. The default is True.
    pplot : bool, optional
        If True, plot figures summarizing results.
    ci : float or None, optional
        Confidence interval for plots.
    refractory_rule : TYPE, optional
        DESCRIPTION. The default is False.
    config_file : str, optional
        Mouse recordings configuration file.

    Returns
    -------
    df : pandas.DataFrame
        with columns ['mouse', 'time', 'val', 'valz', 
                      'pc', 'lsr_start', 'start_state', 'start_state_int',
                      'state', 'success', 'refractory']
        
        
        'state': brain state sequence 
        'start_state_int': How long does it take till the brain state at laser onset
                           switches to a different state
        'valz': PC values during laser trial, for each trial the PC vector
                is z-scored.
        'rem_delay': Delay from laser to REM onset; if there's no REM the value is set to -1

    """
    if rnd_laser:
        np.random.seed(seed)
    
    dt = 2.5
    
    ddir = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(ddir)
    nhypno = M.shape[0]
    ndim = PC.shape[0]
    tidx = np.arange(0, nhypno)
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>0) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################
            
    if os.path.isfile(os.path.join(ddir, 'laser_%s.mat' % name)):
        sr = sleepy.get_snr(ppath, name)
        nbin = int(np.round(sr)*dt)
        dt = nbin * (1.0/sr)
        
        ipre = int(pre/dt)
        ipost = int(post/dt)
        t = np.arange(-ipre, ipost)*dt
        nt = len(t)
    
        #######################################################################
        # get laser start and end index after excluding kcuts: ################
        lsr = so.loadmat(os.path.join(ddir, 'laser_%s.mat' % name), squeeze_me=True)['laser']

        idxs, idxe = sleepy.laser_start_end(lsr)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]

        dur = int(60/2.5)                
        if rnd_laser:
            
            idxs_rnd = []
            idxe_rnd = []
            
            tmp = np.random.randint(dur, idxs[0]-dur)
            idxs_rnd.append(tmp)
            idxe_rnd.append(tmp+dur)
            for (a,b) in zip(idxe[0:-1], idxs[1:]):
                
                if a+2*dur < b-dur:
                    
                    tmp = np.random.randint(a+2*dur,b-dur)
                    idxs_rnd.append(tmp)
                    idxe_rnd.append(tmp+dur)

            idxs = idxs_rnd 
            idxe = idxe_rnd                                       
                    
        laser_idx = []
        for (si,sj) in zip(idxs, idxe):
            laser_idx += list(range(si,sj+1))

        nlsr = int(np.floor(lsr.shape[0]/nbin))
        laser = np.zeros((nlsr,))
        laser[laser_idx] = 1
        laser = laser[tidx]

        idxs = [s[0]  for s in sleepy.get_sequences(np.where(laser == 1)[0])]
        idxe = [s[-1] for s in sleepy.get_sequences(np.where(laser == 1)[0])]
        #######################################################################

        label = []
        for p in range(1, ndim+1):
            l = 'pc' + str(p)
            label.extend([l]*nt)

        # zscore pcs:
        if pzscore_pc:
            for i in range(PC.shape[0]):
                PC[i,:] = (PC[i,:]- PC[i,:].mean()) / PC[i,:].std()

        data = []
        ev = 0
        for i,j in zip(idxs, idxe):
            if i > ipre and i+ipost < nhypno:
                
                if (j-i)*dt < min_laser:
                    continue
                
                idx = np.arange(i-ipre, i+ipost).astype('int')
                pc_cut = PC[:,idx]
                
                pc_cut_z = pc_cut.copy()
                for k in range(ndim):
                    pc_cut_z[k,:] = (pc_cut[k,:] - pc_cut[k,:].mean()) / pc_cut[k,:].std()
                
                m_lsr = M[i:j+1]
                # repeat m_cut ndim-times 
                m_cut = np.tile(M[idx], (ndim,))
                
                vec = np.reshape(pc_cut, (ndim*nt,))
                
                vecz = np.reshape(pc_cut_z, (ndim*nt,))
                tvec = np.tile(t, (ndim,))
                
                lsr_rem = 'no'
                if 1 in m_lsr:
                    lsr_rem='yes'
                    
                start_state = M[i]
                
                rem_delay = -1
                if lsr_rem == 'yes':
                    rem_delay = np.where(m_lsr == 1)[0][0] * dt
                
                # duration of interval of brainstate start_state till the
                # brain_state switches:
                start_state_int = len(m_lsr)*dt                
                a = np.where(m_lsr != start_state)[0]
                if len(a) > 0:
                    start_state_int = len(a) * dt
                
                l = i-1
                while M[l] != 1 and l>0:
                    l = l-1
                
                refr = 'no'
                if M[l] == 1 and l != i-1:

                    v = l
                    while M[v] == 1 and v > 0:                    
                        v = v-1
                    
                    v = v+1
                    dur_rem_pre = (l-v+1)*dt
                    inrem = len(np.where(M[l+1:i] == 3)[0]) * dt
                
                
                    if inrem <= dur_rem_pre * 2:
                        refr = 'yes'
                               
                data += zip([mouse]*nt*ndim, [ev]*nt*ndim, tvec, vec, vecz, 
                            label, [i]*nt*ndim, [start_state]*nt*ndim, [start_state_int]*nt*ndim,
                            m_cut, [lsr_rem]*nt*ndim, [rem_delay]*nt*ndim, [refr]*nt*ndim)
                ev += 1
                                                        
        df = pd.DataFrame(data=data, columns=['mouse', 'ev', 'time', 'val', 'valz', 
                                              'pc', 'lsr_start', 'start_state', 'start_state_int',
                                              'state', 'success', 'rem_delay', 'refractory'])
        
    if pplot:
        plt.figure() 
        if local_pzscore:
            sns.lineplot(data=df, x='time', y='valz', hue='pc', palette='husl')
        else:
            sns.lineplot(data=df, x='time', y='val', hue='pc', palette='husl')
        plt.xlim([t[0], t[-1]])
        sns.despine()
                
        pcs = df.pc.unique()
        data = {p:[] for p in pcs}
        data_start = []
        for p in pcs:
            for si in df.lsr_start.unique():
                a = np.array(df[(df.lsr_start==si) & (df.pc == p)]['val'])
                m_cut = np.array(df[(df.lsr_start==si) & (df.pc == p)]['state'])
                t = np.array(df[(df.lsr_start==si) & (df.pc == p)]['time'])
                l = np.array(df[(df.lsr_start==si) & (df.pc == p)]['success'])[0]
                
                if p == 'pc1':
                    if l == 'yes':
                        rem_start = np.where((t >= 0) & (m_cut == 1) )[0][0]
                        data_start.append(t[rem_start])
                    else:
                        data_start.append(-1)
                
                data[p].append(a)
                                
        for p in pcs:
            data[p] = np.array(data[p])

        f, axes = plt.subplots(nrows=ndim, ncols=1, sharex='all')
            
        for ax,p in zip(axes, pcs):
            mx = data[p].copy()
            if local_pzscore:
                for i in range(mx.shape[0]):
                    mx[i,:] = (mx[i,:] - mx[i,:].mean()) / mx[i,:].std()
            
            ax.pcolorfast(t, range(0, mx.shape[0]+1), mx, cmap='jet')
            for ii in range(mx.shape[0]):
                tt = data_start[ii]
                if tt >= 0:
                    ax.plot([tt,tt], [ii,ii+1], color='black', lw=3)
            ax.set_ylabel('')
                                            
    return df
        


def laser_triggered_frs(units, pre, post, mouse, kcuts=[], ma_thr=10, ma_rem_exception=True,
                        pzscore=True, nsmooth=0, detrend=True, min_laser=20,
                        config_file='mouse_config.txt'):
    """
    

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    pre : TYPE
        DESCRIPTION.
    post : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is True.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    detrend : TYPE, optional
        DESCRIPTION. The default is True.
    min_laser : TYPE, optional
        DESCRIPTION. The default is 20.
    config_file : TYPE, optional
        DESCRIPTION. The default is 'mouse_config.txt'.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """    
    dt = 2.5
    
    ddir = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(ddir)
    M = sleepy.load_stateidx(ppath, name)[0]

    nhypno = M.shape[0]-1
    ndim = units.shape[1]
    tidx = np.arange(0, nhypno)
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>0) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################

    if os.path.isfile(os.path.join(ddir, 'laser_%s.mat' % name)):
        sr = sleepy.get_snr(ppath, name)
        nbin = int(np.round(sr)*dt)
        dt = nbin * (1.0/sr)
        
        ipre = int(pre/dt)
        ipost = int(post/dt)
        t = np.arange(-ipre, ipost)*dt
        nt = len(t)
    
        #######################################################################
        # get laser start and end index after excluding kcuts: ################
        lsr = so.loadmat(os.path.join(ddir, 'laser_%s.mat' % name), squeeze_me=True)['laser']

        idxs, idxe = sleepy.laser_start_end(lsr)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
                
        laser_idx = []
        for (si,sj) in zip(idxs, idxe):
            laser_idx += list(range(si,sj+1))

        nlsr = int(np.floor(lsr.shape[0]/nbin))
        laser = np.zeros((nlsr,))
        laser[laser_idx] = 1
        laser = laser[tidx]

        idxs = [s[0]  for s in sleepy.get_sequences(np.where(laser == 1)[0])]
        idxe = [s[-1] for s in sleepy.get_sequences(np.where(laser == 1)[0])]
        #######################################################################

        unitIDs = [unit for unit in units.columns if '_' in unit]    
        nvar    = len(unitIDs) 
        R = np.zeros((nvar, nhypno)) # dimensions: number of units x time bins
        for i,unit in enumerate(unitIDs):

            tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
            tmp = tmp[tidx]
            if detrend:
                tmp = scipy.signal.detrend(tmp)
    
            if pzscore:
                R[i,:] = (tmp - tmp.mean()) / tmp.std()
            else:
                R[i,:] = tmp
        
        label = []
        for p in range(0, nvar):
            l = unitIDs[p]
            label.extend([l]*nt)
    
        ms_id = [mouse + '_' + i for i in label]
    
        data = []
        for i,j in zip(idxs, idxe):
            if i > ipre and i+ipost < nhypno:
                
                if (j-i)*dt < min_laser:
                    continue
                
                idx = np.arange(i-ipre, i+ipost).astype('int')
                r_cut = R[:,idx]
    
                m_lsr = M[i:j+1]
                # repeat m_cut ndim-times 
                m_cut = np.tile(M[idx], (nvar,))
                
                vec = np.reshape(r_cut, (nvar*nt,))
                tvec = np.tile(t, (ndim,))
        
                lsr_rem = 'no'
                if 1 in m_lsr:
                    lsr_rem='yes'
                    
                start_state = M[i]

                # duration of interval of brainstate start_state till the
                # brain state switches:
                start_state_int = len(m_lsr)*dt                
                a = np.where(m_lsr != start_state)[0]
                if len(a) > 0:
                    start_state_int = len(a) * dt

                data += zip([mouse]*nt*nvar, tvec, vec, 
                            label, ms_id, [i]*nt*nvar, [start_state]*nt*nvar, [start_state_int]*nt*nvar, 
                            m_cut, [lsr_rem]*nt*nvar)
        
        df = pd.DataFrame(data=data, columns=['mouse', 'time', 'fr', 'ID', 
                                              'ms_id', 'lsr_start', 'start_state', 'start_state_int', 
                                              'state', 'success'])    
    return df
    

                
def plot_trajectories(PC, M, pre, post, istate=1, dt=2.5, ma_thr=10, ma_rem_exception=False,
                      kcuts=[],min_dur=0, pre_state=0, state_num=[], ax='', lw=1, coords=[0,1], mouse=''):
    
    tidx = np.arange(0, len(M))

    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    nhypno = np.min((len(M), PC.shape[1]))
    M = M[0:nhypno]

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    
    if ax == '':
        plt.figure()
        ax = plt.axes([0.15, 0.15, 0.7, 0.7])
    ax_3d = False
    if ax == '3D':
        ax_3d = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0.2, 0.2]]

    ipre = int(pre/dt)
    ipost = int(post/dt)

    if istate in [1,2,3]:
        if istate > 0:
            seq = sleepy.get_sequences(np.where(M==istate)[0])
        else:
            seq = [np.arange(0, len(M))]
    
        if len(state_num) > 0:
            seq = [seq[i] for i in state_num]
        
        for s in seq:
            si = s[0]
            sj = s[-1]
            
            if si-ipre > 0 and sj+ipost < nhypno and len(s)*dt > min_dur and M[si-1]==pre_state:
                #print(len(s)*dt)
                mcut = M[si-ipre:sj+ipost]
                
                k = si-ipre
                p = mcut[0]
                kold = k
                while k < sj+ipost:
                    while M[k] == p and k < sj+ipost:
                        k+=1
                    if ax_3d:
                        ax.plot(PC[coords[0],kold:k+1], PC[coords[1], kold:k+1], PC[coords[2], kold:k+1], color=clrs[int(p)], lw=lw)
                        
                    else:
                        ax.plot(PC[coords[0],kold:k+1], PC[coords[1], kold:k+1], color=clrs[int(p)], lw=lw)
                    p = M[k]
                    kold = k

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
    if istate == 'laser':
        ddir = load_config('mouse_config.txt')[mouse]['SL_PATH']
        ppath, name = os.path.split(ddir)

        sr = sleepy.get_snr(ppath, name)
        nbin = int(np.round(sr)*2.5)
        dt = nbin * (1.0/sr)

        #######################################################################
        # get laser start and end index after excluding kcuts: ################
        lsr = so.loadmat(os.path.join(ddir, 'laser_%s.mat' % name), squeeze_me=True)['laser']

        idxs, idxe = sleepy.laser_start_end(lsr)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
                
        laser_idx = []
        for (si,sj) in zip(idxs, idxe):
            laser_idx += list(range(si,sj+1))

        nlsr = int(np.floor(lsr.shape[0]/nbin))
        laser = np.zeros((nlsr,))
        laser[laser_idx] = 1
        laser = laser[tidx]

        idxs = np.array([s[0] for s in sleepy.get_sequences(np.where(laser == 1)[0])])
        idxe = np.array([s[-1] for s in sleepy.get_sequences(np.where(laser == 1)[0])])
        #######################################################################
    
        for (si,sj) in zip(idxs[state_num], idxe[state_num]):
            if (sj-si) * dt < 20:
                continue
            
            mcut = M[si:sj+1]
            if 1 in mcut:
            
                ax.plot(PC[coords[0],si-ipre:si], PC[coords[1], si-ipre:si], color='cornflowerblue', lw=lw)
                ax.plot(PC[coords[0],si-1:sj], PC[coords[1], si-1:sj], color='blue', lw=lw)
                ax.plot(PC[coords[0],si-1], PC[coords[1], si-1], color='orange', marker='o', lw=lw)
                
            else:
                ax.plot(PC[coords[0],si-ipre:si], PC[coords[1], si-ipre:si], color='pink', lw=lw)
                ax.plot(PC[coords[0],si-1:sj], PC[coords[1], si-1:sj], color='red', lw=lw)
                ax.plot(PC[coords[0],si-1], PC[coords[1], si-1], color='green', marker='o', lw=lw)
                


def pc_state_space(PC, M, ma_thr=10, ma_rem_exception=False, kcuts=[], dt=2.5, ax='', nrem2wake=False, nrem2wake_step=4,
                   pscatter=True, local_coord=False, outline_std=True, rem_onset=False, rem_offset=False, rem_offset_only_nrem=False, show_avgtraj=False,
                   pre_win=30, post_win=0, rem_min_dur=0, break_out=True, break_in=False, prefr=False, scale=1.645):
    """
    Plot for each time point the population activity within the 2D state space spanned
    by PC[0,:] and PC[1,:]

    Parameters
    ----------
    PC : np.array
        each row are the PC coefficients or scores
    M : np.array
        brain state annotation.
        1 - REM, 2 - Wake, 3 - NREM
    ma_thr : float, optional
        Microarousal threshold. The default is 10.
    ma_rem_exception : bool, optional
        If True, don't set a wake episodes after REM that is shorter then $ma_thr to NREM. 
        The default is False.
    kcuts : list of tuples, optional
        Each tuple describes the start and end of an interval to be discarded. The default is [].
    dt : float, optional
        Time bin of firing rates and hypnogram. The default is 2.5.
    ax : figure axis handle, optional
        If $as is provided, use it to draw all plots on it. 
        Otherwise generate new figure.
        The default is ''.
    nrem2wake : Bool, optional
        if True, show NREM->Wake transitions
    nrem2wake_step: int, optional
        Show every nrem2wake_step-th NREM->Wake transition; otherwise
        it gets too clusttered
    pscatter : bool, optional
        If True, draw each (dimensionally reduced) population vector in a scatter plot. 
        The default is True.
    local_coord : bool, optional
        If True, draw into the NREM ellipse a local coordinate system. The default is False.
    outline_std : bool
        If True, draw outline of one std using an ellipse;
        if False, use sns.kdeplot to draw outline of data spread
    rem_onset : bool
        If True, color-code the REM onset and the preceding $pre_win seconds 
    rem_offset : bool
        If True, plot REM->Wake->NREM transitions
    rem_offset_only_nrem: bool
        If True, plot only REM->Wake->NREM transitions where NREM occurs within 
        the $post_win interval following the REM offset.
    show_avgtraj : bool
        If True, show average across all trajectories instead of each individual
        trajectory
    rem_mindur: float
        Only REM episodes >= rem_mindur are considered
    break_out: bool
        if True, draw a dot where a NREM to REM trajectory
        leaves the NREM subspace, defined by $scale
    prefr: bool,
        If True, also draw outline of refractory period within state space
    scale: float
        $scale == 1 means that the drawn ellipse outlines one standard deviation
        for each subspace. 
        $scale == 1.645 outlines the area (of the fitted Gaussian)
        that comprises 90% of the data distribution.                    
        $scale == 1.96 outlines 95% of the distribution
        
    # ax, df_breakout, df_breakin, df_ampl
        
    Returns
    -------
    ax : plt.axes
        return axes of current figure.
        
    df_breakout : pd.DataFrame
        with columns ['angle':first_angles, 'pc1':c1, 'pc2':c2, 'pc1_org':porg1, 'pc2_org':porg2]
        Describes along each NREM->REM trajectory the first point and angle leaving the
        NREM subspace
    df_breakin : pd.DataFrame
        Describes for each REM->Wake->NREM transition trajectory the first point within NREM. 
    df_ampl: pd.DataFrame
        Describes for each NREM->REM ($nrem2wake=False) or NREM->Wake ($nrem2wake=True) the maximum
        PC1 and PC2 values of the preceding trajectory of during $pre_win. 

    """        
    bs_map = {'rem':[0, 1, 1], 'wake':[0.6, 0, 1], 'nrem':[0.8, 0.8, 0.8]}
    tidx = np.arange(0, len(M))
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    nhypno = np.min((len(M), PC.shape[1]))
    M = M[0:nhypno]

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3

    if ax == '':
        plt.figure()
        ax = plt.axes([0.2, 0.15, 0.7, 0.7])
    clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.6, 0.6, 0.6], [1, 0.2, 0.2]]

    state_idx = {}
    for s in [1,2,3]:
        idx = np.where(M==s)[0]
        state_idx[s] = idx

    # get all indices for REM, Wake, and NREM    
    for s in [1,2,3]:
        idx = state_idx[s]
        C = PC[0:2,idx].T
    
        if outline_std:            
            mean = C.mean(axis=0)
            covar = np.cov(C.T)
        
            v, w = linalg.eigh(covar)
            # columns of w are the eigenvectors
            # v are the eigenvalues in ascending order
            
            # 2 * scale * std (the 2 is to transform the radius to diameter)
            v = 2.0 * scale * np.sqrt(v) 
            u = w[0] / linalg.norm(w[0])
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=clrs[s],  lw=2)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.3)
            ax.add_artist(ell)
        
        else:        
            sns.kdeplot(x=C[:, 0], y=C[:, 1], ax=ax, color=clrs[s], fill=True, alpha=0.8, levels=[0.25, 0.5, 0.75, 1])

        # Add refractory period to state space
        if prefr:
            refr_color = 'maroon'
            df_refr, refr_vec, _ = add_refr(M)
        
            refr_idx = np.where(refr_vec == 1)[0]
            nr_idx = np.where(M==3)[0]
            idx = np.intersect1d(refr_idx, nr_idx)
                        
            C = PC[0:2,idx].T

            if outline_std:
                mean = C.mean(axis=0)
                covar = np.cov(C.T)
            
                v, w = linalg.eigh(covar)
                # columns of w are the eigenvectors
                # v are the eigenvalues in ascending order
                
                # 2 * scale * std (the 2 is to transform the radius to diameter)
                v = 2.0 * scale * np.sqrt(v) 
                u = w[0] / linalg.norm(w[0])
                
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=refr_color,  lw=2)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.3)
                ax.add_artist(ell)
            else:
                sns.kdeplot(x=C[:, 0], y=C[:,1], ax=ax, color=refr_color, fill=True, alpha=0.8, levels=[0.25, 0.5, 0.75, 1])

        if local_coord:
            idx = state_idx[3]
            C = PC[0:2,idx].T
            
            pca = PCA(n_components=2)
            pca.fit(C)            
            mean_x, mean_y = np.mean(C, axis=0)
            pc1, pc2 = pca.components_        
            if pc1[1] < 0:
                pc1 = -pc1
        
            # Calculate the standard deviations along the principal components
            std_x, std_y = np.sqrt(pca.explained_variance_)
            
            # Scale the principal components by the standard deviations
            pc1_scaled = pc1 * std_x * 1 * scale
            pc2_scaled = pc2 * std_y * 1 * scale
            
            origin = [mean_x], [mean_y]
            ax.quiver(*origin, pc1_scaled[0], pc1_scaled[1], angles='xy', scale_units='xy', scale=1, color='black', label='PC1 (Scaled)')
            ax.quiver(*origin, pc2_scaled[0], pc2_scaled[1], angles='xy', scale_units='xy', scale=1, color='black', label='PC2 (Scaled)')
        
        pc1_min = PC[0,:].min()
        pc1_max = PC[0,:].max()
        pc2_min = PC[1,:].min()
        pc2_max = PC[1,:].max()   
        d1 = pc1_max - pc1_min
        d2 = pc2_max - pc2_min
        ax.set_xlim(pc1_min - 0.1*d1, pc1_max + 0.1*d1)
        ax.set_ylim(pc2_min - 0.1*d2, pc2_max + 0.1*d2)
        
    if pscatter:
        for s in [1,2,3]:
            idx = state_idx[s]
            ax.scatter(PC[0,idx], PC[1,idx], color=clrs[s], s=1, alpha=1)            
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    sns.despine()

    # just the onset of REM as single dot:
    ipre_win = int(pre_win/dt)
    ipost_win = int(post_win/dt)
    rem_start = [s[0] for s in sleepy.get_sequences(np.where(M==1)[0]) if len(s)*dt >= rem_min_dur and s[0]*dt >= pre_win and s[0]+ipost_win < len(M)]    
    if nrem2wake:
        rem_start = [s[0] for s in sleepy.get_sequences(np.where(M==2)[0]) if len(s)*dt >= rem_min_dur and s[0]*dt >= pre_win and M[s[0]-1]==3]    
        rem_start = rem_start[1::nrem2wake_step]

    # show trajectories for REM onset
    if rem_onset:
        if not show_avgtraj:
            for r in rem_start[:]:
                if not nrem2wake:
                    plt.plot(PC[0,r], PC[1,r], '*', color=bs_map['rem'], markersize=10, zorder=3)
                else:
                    plt.plot(PC[0,r], PC[1,r], '*', color=bs_map['wake'], markersize=10, zorder=3)
        
            for r in rem_start:
                # NEW - 04/17/24:
                pc1, pc2 = PC[0,r-ipre_win:r+ipost_win+1], PC[1,r-ipre_win:r+ipost_win+1]            
                sm = _jet_plot(pc1, pc2, ax, lw=2, cmap='magma')
        else:
            tmp1, tmp2 = [], []
            for r in rem_start:
                pc1, pc2 = PC[0,r-ipre_win:r+ipost_win+1], PC[1,r-ipre_win:r+ipost_win+1]            
                tmp1.append(pc1)
                tmp2.append(pc2)
            
            pc1 = np.array(tmp1).mean(axis=0)
            pc2 = np.array(tmp2).mean(axis=0)
            sm = _jet_plot(pc1, pc2, ax, lw=2, cmap='magma')
            
            
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.6)  # pad adjusts the distance between the plot and colorbar
        sm.set_clim(-pre_win, post_win)
        cbar.set_ticks([-pre_win, post_win])
        cbar.set_label("Time (s)")
    
    if rem_offset:
        ipre_win = int(pre_win/dt)
        ipost_win = int(post_win/dt)
        rem_end_all = [s[-1]+1 for s in sleepy.get_sequences(np.where(M==1)[0]) if len(s)*dt >= rem_min_dur and s[-1]+ipost_win < len(M) and s[-1]>ipre_win]    
        tmp = []
        
        # for each REM offset search for the end of the following wake episode
        for r in rem_end_all:
            i = r
            while i < len(M) and M[i] != 3:
                i += 1            
            dur = (i - r) * dt            
            if dur <= post_win:
                tmp.append(r)                
        rem_end_nrem = tmp

        if rem_offset_only_nrem:
            rem_end = rem_end_nrem
        else:
            rem_end = rem_end_all
                
        for r in rem_end:
            pc1, pc2 = PC[0,r-ipre_win:r+ipost_win+1], PC[1,r-ipre_win:r+ipost_win+1]
            sm = _jet_plot(pc1, pc2, ax, lw=2, cmap='magma')
            
            plt.plot(PC[0,r+1], PC[1,r+1], '*', color=bs_map['wake'], markersize=10, zorder=3)

            
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.6)  # pad adjusts the distance between the plot and colorbar
        sm.set_clim(-pre_win, post_win)
        cbar.set_ticks([-pre_win, post_win])
        cbar.set_label("Time (s)")
            
    df_breakout = []
    data_ampl = []
    df_ampl = []
    if break_out:
        seq = sleepy.get_sequences(np.where(M==1)[0])
        idx = state_idx[3]
        C = PC[0:2,idx].T
        
        meanc = C.mean(axis=0)
        covar = np.cov(C.T)
        
        pca = PCA(n_components=2)
        pca.fit(C)        
        # get the eigenvectors of the covariance matrix:
        pc1, pc2 = pca.components_        
        if pc1[1] < 0:
            pc1 = -pc1
        w = np.zeros((2,2))
        w[:,0] = pc1
        w[:,1] = pc2
                                        
        first = []
        if not nrem2wake:
            # NREM -> REM
            for r in rem_start:
                ifirst = last_subspace_point(r, PC[0:2,:], meanc, covar, scale=scale)
                first.append(ifirst)
                plt.plot(PC[0,ifirst], PC[1,ifirst], 'ro')
        else:
            # NREM -> Wake
            for r in rem_start:
                if not is_in_ellipse(r, PC[0:2,:], meanc, covar, scale=scale):
                    ifirst = last_subspace_point(r, PC[0:2,:], meanc, covar, scale=scale)
                    first.append(ifirst)            
                    #plt.plot(PC[0,ifirst], PC[1,ifirst], 'ro')
            
        for r in rem_start:
            tmp1 = PC[0,r-ipre_win:r+1]
            tmp2 = PC[1,r-ipre_win:r+1]
            
            data_ampl += [[tmp1.max(), tmp2.max()]]
            
        df_ampl = pd.DataFrame(data=data_ampl, columns=['pc1', 'pc2'])
        
        # Alternative way of calculating the eigenvectors of the
        # covariance matrix:            
        # v, w = linalg.eigh(covar)
        # ii = np.argsort(v)[::-1]
        # v = v[ii]
        # w = w[:,ii]
        
        first_angles = []
        c1, c2 = [], []
        porg1, porg2 = [], []
        porg1_rel, porg2_rel = [], []
        for ifirst in first:
            # take the first point outside the NREM subspace
            p = PC[0:2,ifirst]
            # center the point
            pctr = p - meanc
                        
            # project it onto the eigenvectors
            a = np.dot(pctr, w)
            x, y = a[0], a[1]
            # calculate the angle in radians
            theta = math.atan2(x, y)                                    
            # Convert the angle to degrees;
            # NOTE: 0 deg. corresponds to 3; 90 deg. corresponds to 12
            angle_degrees = math.degrees(theta)
            first_angles.append(angle_degrees)
            c1.append(a[0])
            c2.append(a[1]) 
            
            porg1_rel.append(pctr[0])
            porg2_rel.append(pctr[1])
            porg1.append(p[0])
            porg2.append(p[1])
            
        df_breakout = pd.DataFrame({'angle':first_angles, 'pc1':c1, 'pc2':c2, 
                                    'pc1_org':porg1, 'pc2_org':porg2, 
                                    'pc1_rel':porg1_rel, 'pc2_rel':porg2_rel})

    df_breakin = []
    if break_in:        
        rem_end = rem_end_nrem
        
        seq = sleepy.get_sequences(np.where(M==1)[0])
        idx = state_idx[3]
        C = PC[0:2,idx].T
        
        meanc = C.mean(axis=0)
        covar = np.cov(C.T)
        
        pca = PCA(n_components=2)
        pca.fit(C)        
        # get the eigenvectors of the covariance matrix:
        pc1, pc2 = pca.components_        
        if pc1[1] < 0:
            pc1 = -pc1
        w = np.zeros((2,2))
        w[:,0] = pc1
        w[:,1] = pc2
                                        
        first = []
        for r in rem_end:
            ifirst = first_subspace_point(r, PC[0:2,:], meanc, covar, scale=scale)
            first.append(ifirst)
            plt.plot(PC[0,ifirst], PC[1,ifirst], 'bo')
                            
        first_angles = []
        c1, c2 = [], []
        porg1, porg2 = [], []
        porg1_rel, porg2_rel = [], []
        for ifirst in first:
            # take the first point outside the NREM subspace
            p = PC[0:2,ifirst]
            # center the point
            pctr = p - meanc
                        
            # project it onto the eigenvectors
            a = np.dot(pctr, w)
            x, y = a[0], a[1]
            # calculate the angle in radians
            theta = math.atan2(x, y)                                    
            # Convert the angle to degrees;
            # NOTE: 0 deg. corresponds to 3; 90 deg. corresponds to 12
            angle_degrees = math.degrees(theta)
            first_angles.append(angle_degrees)
            c1.append(a[0])
            c2.append(a[1]) 
            
            porg1_rel.append(pctr[0])
            porg2_rel.append(pctr[1])
            porg1.append(p[0])
            porg2.append(p[1])
            
        df_breakin = pd.DataFrame({'angle':first_angles, 'pc1':c1, 'pc2':c2, 
                                    'pc1_org':porg1, 'pc2_org':porg2, 
                                    'pc1_rel':porg1_rel, 'pc2_rel':porg2_rel})
                                    
    return ax, df_breakout, df_breakin, df_ampl
    


def state_space_geometry(PC, M, ma_thr=10, ma_rem_exception=False, kcuts=[], dt=2.5, ax='', 
                   outline_std=True, prefr=True, show_nrem=True, scale=1.645):
    """
    (1) Distance between different subspaces
    (2) Refractory and permissive state space; draw ellipses capturing the distribution
        of the refractory and permissive period. Note that these periods only include 
        NREM sleep

    Returns
    -------
    df_geom: pd.DataFrame
        with columns ['pc1', 'pc2', 'area', 'state'] that
        describe for each $state the coordindates of the mean of its subspace 
        spanned by 'pc1' and 'pc2' and the 'area' of this subspace.
    df_distr: pd.DataFrame
        with columns ['pc1', 'pc2', 'state']
        All PC1 and PC2 values within refractory and permissive state space
    """
    
    state_map = {1:'REM', 2:'Wake', 3:'NREM'}
    
    # KCUTS
    tidx = np.arange(0, len(M))
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    nhypno = np.min((len(M), PC.shape[1]))
    M = M[0:nhypno]

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3

    if ax == '':
        plt.figure()
        ax = plt.axes([0.2, 0.15, 0.7, 0.7])
    clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0.2, 0.2]]

    state_idx = {}
    for s in [1,2,3]:
        idx = np.where(M==s)[0]
        state_idx[s] = idx

    # get all indices for REM, Wake, and NREM   
    data_geom = []
    if not show_nrem:
        shown_states = [1,2]
    else:
        shown_states = [1,2,3]
    for s in shown_states:
        idx = state_idx[s]
        C = PC[0:2,idx].T

        if outline_std:
            
            mean = C.mean(axis=0)
            covar = np.cov(C.T)
        
            v, w = linalg.eigh(covar)
            # columns of w are the eigenvectors
            # v are the eigenvalues in ascending order
            
            # 2 * scale * std (the 2 is to transform the radius to diameter)
            v = 2.0 * scale * np.sqrt(v) 
            u = w[0] / linalg.norm(w[0])
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=clrs[s],  lw=2, fill=True)
            #else:
            #    ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=clrs[s],  lw=2)
                
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.3)
            
            ax.add_artist(ell)
            
            area = np.pi * v[0] * v[1]
            data_geom += [list(C.mean(axis=0)) + [area] + [state_map[s]]]
        
        else:        
            sns.kdeplot(x=C[:, 0], y=C[:, 1], ax=ax, color=clrs[s], fill=True, alpha=0.8, levels=[0.25, 0.5, 0.75, 1])

    # Add refractory period to state space
    if prefr:
        refr_color = 'maroon'
        perm_color = 'dodgerblue'
        rp_clrs = [perm_color, refr_color]
        df_refr, refr_vec, _ = add_refr(M)
    
        refr_idx = np.where(refr_vec == 1)[0]
        perm_idx = np.where(refr_vec == 2)[0]
        nr_idx = np.where(M==3)[0]
        nr_idx_refr = np.intersect1d(refr_idx, nr_idx)
        nr_idx_perm = np.intersect1d(perm_idx, nr_idx)
        
        Crefr = PC[0:2,nr_idx_refr].T
        Cperm = PC[0:2,nr_idx_perm].T
        
        data_distr = []
        data_distr += zip(Crefr[:,0], Crefr[:,1], ['refr']*Crefr.shape[0])
        data_distr += zip(Cperm[:,0], Cperm[:,1], ['perm']*Cperm.shape[0])
        df_distr = pd.DataFrame(data=data_distr, columns=['pc1', 'pc2', 'state'])
        
                
        if outline_std:
            for C,clr,label in zip([Cperm, Crefr], rp_clrs, ['perm', 'refr']):
                mean = C.mean(axis=0)
                covar = np.cov(C.T)
            
                v, w = linalg.eigh(covar)
                # columns of w are the eigenvectors
                # v are the eigenvalues in ascending order
                
                # 2 * scale * std (the 2 is to transform the radius to diameter)
                v = 2.0 * scale * np.sqrt(v) 
                u = w[0] / linalg.norm(w[0])
                
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=clr,  lw=2)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.3)
                ax.add_artist(ell)
                
                area = np.pi * v[0] * v[1]
                data_geom += [list(C.mean(axis=0)) + [area] + [label]]                
        else:
            for C,clr in zip([Cperm, Crefr], rp_clrs):
                sns.kdeplot(x=C[:, 0], y=C[:,1], ax=ax, color=clr, fill=True, alpha=0.8, levels=[0.25, 0.5, 0.75, 1])

        if outline_std:
            df_geom = pd.DataFrame(data=data_geom, columns=['pc1', 'pc2', 'area', 'state'])
        else:
            df_geom = []
                
    return df_geom, df_distr



def _jet_plot(x, y, ax, color='', cmap="YlOrBr", lw=1):
    from matplotlib.cm import ScalarMappable
    n = len(x)
    if len(color) == 0 and type(cmap) == str:
        clrs = sns.color_palette(cmap, n)
    elif len(color) > 0:
        clrs = [color]*n
    else:
        values = np.linspace(0, 1, n)
        clrs = cmap(values)
    
    for (i,j) in zip(range(0,n-1), range(1, n)):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=clrs[i], lw=lw)

    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])  # You need to set an array, but it can be empty
    
    # Return the ScalarMappable object
    return sm  


def mahalanobis_distance(point, mean, covariance_matrix):
    diff = point - mean
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, inv_covariance_matrix), diff))
    return mahalanobis_distance


def last_subspace_point(istate, PC, meanc, Ci, scale=np.sqrt(2)):    
    i = istate    
    while mahalanobis_distance(PC[:,i], meanc, Ci) > scale and i > 0:
        #print(mahalanobis_distance(PC[:,i], meanc, Ci))
        i = i-1
    
    ifirst = i+1
    return ifirst
    

def is_in_ellipse(istate, PC, meanc, Ci, scale=np.sqrt(2)):
    return mahalanobis_distance(PC[:,istate], meanc, Ci) <= scale


def first_subspace_point(istate, PC, meanc, Ci, scale=np.sqrt(2)):    
    i = istate    
    while mahalanobis_distance(PC[:,i], meanc, Ci) > scale and i < PC.shape[1]:
        i = i+1    
    #ifirst = i-1
    return i

    

def svc_components(units, ndim=3, nsmooth=0):

    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']
    nsample = units.shape[0]   # number of time points
    nvar    = len(unitIDs)     # number of units

    R = np.zeros((nvar, nsample))
    
    i = 0
    for unit in unitIDs:
        R[i,:] = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        i += 1

    
    # first, for each varible (dimension), we first need to remove the mean:
    # mean-zero rows:
    for i in range(nvar):
        R[i,:] = R[i,:] - R[i,:].mean()

    if np.mod(R.shape[0],2) == 1:
        R = R[0:-1,:]
    if np.mod(R.shape[1],2) == 1:
        R = R[:,0:-1]
    
    n = int(R.shape[1]/2)
        
        
    train1 = R[0::2,:n]
    train2 = R[1::2,:n]
    
    test1 = R[0::2,n:2*n]
    test2 = R[1::2,n:2*n]
    
    C = np.dot(train1, train2.T) * np.sqrt(1/n)
    # Then, calculate the eigenvalues and eigenvectors of C
    evals, evecs = scipy.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    evecs = evecs[:,:ndim]
    
    # predict the SVDs for both sets of neurons on the test set
    svd1 = np.dot(evecs.T, test1)
    svd2 = np.dot(evecs.T, test2)

    plt.figure()
    plt.plot(svd1[0:ndim,:].T) 
    #plt.plot(svd2[0:ndim,:].T)

    return svd1, svd2, evals



def irem_trends(units, M, nsmooth=0, pzscore=False, kcut=[], irem_dur=120, wake_prop_threshold=0.5):

    dt = 2.5
    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']

    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)

    
    if len(kcut) > 0:
        kidx = np.arange(int(kcut[0]/dt), int(kcut[-1]/dt))
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)


    R = np.zeros((units.shape[1], len(tidx)))
    i = 0
    for unit in unitIDs:
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]
        i += 1


    seq = sleepy.get_sequences(np.where(M==1)[0])
    
    irem = 0
    data = []
    for (p,q) in zip(seq[0:-1], seq[1:]):
        # indices of current inter-REM period:
        idx = np.arange(p[-1]+1, q[0])
        
        widx = np.where(M[idx] == 2)[0]
        
                
        if len(idx)*dt < irem_dur or len(widx) / len(idx) > wake_prop_threshold:
            continue
        
        nridx = idx[np.where(M[idx]==3)]
        
        for i in range(len(unitIDs)):
            fr = R[i,:]            
            res = stats.linregress((nridx-nridx[0])*dt, fr[nridx])

            data += [[unitIDs[i], irem, res.slope, res.rvalue, res.pvalue, len(p)*dt]]
        irem += 1

    df = pd.DataFrame(data=data, columns=['ID', 'rem', 's', 'r', 'p', 'rem_dur'])

    data = []
    for ID in unitIDs:
        df2 = df[df.ID == ID]
        
        p = df2.p.mean()
        s = df2.s.mean()
        r = df2.r.mean()
        
        data += [[ID, s, r, p]]
        
    dfm = pd.DataFrame(data=data, columns=['ID', 's', 'r', 'p'])


    return df, dfm



def rem_prepost(units, M, pzscore=True,  ma_thr=20, ma_rem_exception=True, kcuts=[], nsmooth=0, postwake_thr=60):
    
    dt = 2.5
    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
            
    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    
    
    
    R = np.zeros((units.shape[1], len(tidx)))
    for i, unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]


    seq = sleepy.get_sequences(np.where(M==1)[0])
    data = []
    # go over all units
    for iunit,ID in enumerate(unitIDs):
        fr = R[iunit,:]
        #region = cell_info[(cell_info.ID == ID)]['brain_region'].iloc[0]
        
        # go over all inter-REM episodes
        if len(seq) >= 1:
            for s in seq:                
                
                istart = s[0]
                iend = s[-1]
                
                
                if M[istart-1] == 3:
                    a = istart-1
                    
                    while(M[a] == 3) and a>0:
                        a = a-1                    
                    a = a+1
                    nrempre_idx = np.arange(a, istart)
                    
                    b = iend+1
                    
                    while b<len(M)-1 and M[b] != 3:
                        b = b+1
                    
                    if  b<len(M)-1 and (b-iend)*dt < postwake_thr and M[b] == 3:
                        c = b
                        while M[c] == 3 and c<len(M)-1:
                            c += 1
                        nrempost_idx = np.arange(b, c)
                        
                        if len(nrempre_idx) > 0 and len(nrempost_idx) > 0:
                            rem_pre = len(s)*dt
                            dfr = fr[nrempost_idx].mean() -  fr[nrempre_idx].mean()
                            data += [[ID, istart, fr[nrempre_idx].mean(), fr[nrempost_idx].mean(), dfr, rem_pre]]
                    
                    
    df = pd.DataFrame(data=data, columns=['ID', 'rem_id', 'fr_pre', 'fr_post', 'dfr', 'rem_pre'])
    return df



def remrem_sections(units, cell_info, M, nsections=5, nsections_rem=0, nsmooth=0, pzscore=False, kcuts=[], 
                    irem_dur=120, refractory_rule=False, wake_prop_threshold=0.5, ma_thr=10, ma_rem_exception=False, 
                    border=0, linreg_cut=0, nan_check=False):
    """
    Calculate the everage NREM and Wake activity during consecutive sections of the
    inter-REM interval for each single unit and then perform regression analysis to test whether 
    the given unit significantly increases or decreases throughout inter-REM.

    Parameters
    ----------
    units : pd.DataFrame
        each columns is a unit; the column name is the unit ID used to access
        units in $cell_info
    cell_info : pd.DataFrame
        lists for each unit (cell_info.ID) information such as brain_region
    M : np.array
        hypnogram (as returned from sleepy.load_stateidx).
    nsections : int, optional
        DESCRIPTION. The default is 5.
    nsections_rem : int, optional
        If nsections_rem > 0, also include REM_pre and REM_post using $nsections_rem
        bins
    nsmooth : float, optional
        Smooth firing rate across entire recording.
    pzscore : bool, optional
        If True, zscore data. The default is False.
    kcut : tuple, optional
        If non-empty discard the recording from kcut[0] to kcut[1] seconds. The default is [].
    irem_dur : float, optional
        If an inter-REM interval is < $irem_dur discard this inter-REM interval. The default is 120.
    refractory_rule : bool, optional
        If True, apply refractory rule to exclude sequential REM episodes
    wake_prop_threshold : float, optional
        Maximum allowed proportion of wake states within inter-REM interval. 
        If proportion of wake > $wake_prop_threshold, then discard the given inter-REM interval
        The default is 0.5.
    border : float, optional
        If > 0, cut off the last $border seconds at the end of the interREM interval
    linreg_cut: float, optional
        value between 0 and 1. Cut away the first $*100 and last $*100 percent of each
        normalized firing rate vector for linear regression analysis that goes into
        df_stats
    nan_check: bool, optional
        If True, exclude trials (inter-REM intervals) that include nan.

    Returns
    -------
    df_trials : TYPE
        DESCRIPTION.
    dfm : pd.DataFrame
        with columns ['ID', 'section', 'fr', 'state', 'brain_region'].
    df_stats : TYPE
        DESCRIPTION.

    """
    refr_func = refr_period()
    
    dt = 2.5
    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
            
    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    
    
    
    R = np.zeros((units.shape[1], len(tidx)))
    for i, unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]


    seq = sleepy.get_sequences(np.where(M==1)[0])
    data = []
    ev = 0    
    # go over all units
    for iunit,ID in enumerate(unitIDs):
        fr = R[iunit,:]
        region = cell_info[(cell_info.ID == ID)]['brain_region'].iloc[0]
        
        # go over all inter-REM episodes
        if len(seq) >= 2:
            for (si, sj) in zip(seq[0:-1], seq[1:]):                
                irem_idx = np.arange(si[-1]+1, sj[0]-border, dtype='int')
                rem_id = si[0]
                
                # the refractory duration:
                refr_dur = np.exp(refr_func(len(si)*dt))
                                
                # Apply refractory period:
                if refractory_rule:
                    mcut = M[irem_idx]
                    nrem_dur = len(np.where(mcut==3)[0])*dt 
                    if nrem_dur <= len(si)*2*dt:
                        continue
                                
                # if inter-REM is too short, or if there's too much of wake, just continue with 
                # next inter-REM interval
                widx = np.where(M[irem_idx] == 2)[0]
                if len(irem_idx) * dt < irem_dur or len(widx) / len(irem_idx)  > wake_prop_threshold:
                    continue

                rem_pre = len(si)*dt
                # m - number of bins during inter-REM
                m = len(irem_idx)                
                M_up = upsample_mx( M[irem_idx], nsections)
                M_up = np.round(M_up)
                fr_up = upsample_mx(fr[irem_idx], nsections)
    
                single_event_nrem = []
                single_event_wake = []
                single_event_nrw  = []                
                for p in range(nsections):
                    # for each m consecutive bins calculate average NREM, REM, Wake activity
                    mi = list(range(p*m, (p+1)*m))
    
                    idcut = np.intersect1d(mi, np.where(M_up == 2)[0])
                    if len(idcut) == 0:
                        wake_fr = np.nan
                    else:
                        wake_fr = np.nanmean(fr_up[idcut])
                    single_event_wake.append(wake_fr)
    
                    idcut = np.intersect1d(mi, np.where(M_up == 3)[0])
                    if len(idcut) == 0:
                        nrem_fr = np.nan
                    else:
                        nrem_fr = np.nanmean(fr_up[idcut])
                    single_event_nrem.append(nrem_fr)
    
                    single_event_nrw.append(np.nanmean(fr_up[mi]))
    
                #inter-REM duration in seconds (!)
                idur = len(irem_idx)*dt
                
                mcut = M[irem_idx]
                nrem_dur = len(np.where(mcut==3)[0])*dt 
                wake_dur = len(np.where(mcut==2)[0])*dt 
                
                mcut2 = np.zeros(mcut.shape)
                mcut2[mcut==3] = dt
                mcut_csum = np.cumsum(mcut2)
                # isplit is the first point (index) that is not anymore in the refractory period
                try:
                    isplit = (np.where(mcut_csum >= refr_dur)[0][0]) * 1
                except:
                    isplit = len(mcut_csum)
                    print('HERE isplit')
                refr_perc = dt * isplit / idur
                
                add_nrem = True
                add_wake = True
                add_nrw = True
                if nan_check:
                    if np.any(np.isnan(np.array(single_event_nrem))):
                        add_nrem = False

                    if np.any(np.isnan(np.array(single_event_wake))): 
                        add_wake = False
                        
                    if np.any(np.isnan(np.array(single_event_nrw))):
                        add_nrw = False
                                        
                if add_nrem:
                    data += zip([ID]*nsections, [rem_id]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_nrem, ['NREM']*nsections, [nrem_dur]*nsections, [region]*nsections, [rem_pre]*nsections, [refr_perc]*nsections)
                if add_wake:
                    data += zip([ID]*nsections, [rem_id]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_wake, ['Wake']*nsections, [wake_dur]*nsections, [region]*nsections, [rem_pre]*nsections, [refr_perc]*nsections)
                if add_nrw:
                    data += zip([ID]*nsections, [rem_id]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_nrw,  ['NRW']*nsections,  [idur]*nsections,     [region]*nsections, [rem_pre]*nsections, [refr_perc]*nsections)

                if nsections_rem > 0:
                    # REM-pre
                    fr_pre = time_morph(fr[si], nsections_rem)
                    dur = len(si)*dt
                    
                    data += zip([ID]*nsections_rem, [rem_id]*nsections_rem, [ev]*nsections_rem, 
                                list(range(1, nsections_rem+1)), 
                                fr_pre, ['REM_pre']*nsections_rem, [rem_pre]*nsections_rem, 
                                [region]*nsections_rem, [dur]*nsections_rem, [refr_perc]*nsections_rem)            
                    #REM-post
                    fr_post = time_morph(fr[sj], nsections_rem)
                    dur = len(sj)*dt
                    data += zip([ID]*nsections_rem, [rem_id]*nsections_rem, [ev]*nsections_rem, 
                                list(range(1, nsections_rem+1)), 
                                fr_post, ['REM_post']*nsections_rem, [dur]*nsections_rem, 
                                [region]*nsections_rem, [rem_pre]*nsections, [refr_perc]*nsections_rem)            

                ev += 1
    
    df_trials = pd.DataFrame(columns = ['ID', 'rem_id', 'event', 'section', 'fr', 'state', 'idur', 'brain_region', 'rem_pre', 'refr_perc'], data=data)
    dfm = df_trials.groupby(['ID', 'section', 'state', 'brain_region']).mean().reset_index()
    dfm = dfm[['ID', 'section', 'fr', 'state', 'brain_region', 'refr_perc']]
    
    data = []
    for ID in dfm.ID.unique():
        for cond in ['NREM', 'Wake', 'NRW']:
            df2 = df_trials[(df_trials.ID == ID) & (df_trials.state == cond)]
            df2 = df2[~df2.fr.isna()]    
            X = df2['section']
            Y = df2['fr']
            
            ncut = int(nsections * linreg_cut)
            
            res = stats.linregress(X[ncut:-ncut], Y[ncut:-ncut])
            region = df2['brain_region'].iloc[0]
            data += [[ID, cond, res.slope, res.rvalue, res.pvalue, region]]
                        
        # df2 = df_trials[(df_trials.ID == ID) & (df_trials.state == 'Wake')]
        # df2 = df2[~df2.fr.isna()]    
        # res = stats.linregress(df2['section'], df2['fr'])
        # data += [[ID, 'Wake', res.slope, res.rvalue, res.pvalue, region]]            
        
        # df2 = df_trials[(df_trials.ID == ID) & (df_trials.state == 'NRW')]
        # df2 = df2[~df2.fr.isna()]    
        # res = stats.linregress(df2['section'], df2['fr'])
        # data += [[ID, 'NRW', res.slope, res.rvalue, res.pvalue, region]]            
        
                
    df_stats = pd.DataFrame(data=data, columns=['ID', 'state', 'slope', 'r', 'p', 'brain_region'])
    
    return df_trials, dfm, df_stats                



def remrem_purenrem(units, cell_info, M, nsections=5, nsections_rem=0, nsmooth=0, pzscore=False, kcuts=[], 
                    irem_dur=120, wake_prop_threshold=0.5, ma_thr=10, border=20, ma_rem_exception=False):
    """
    Calculate the average NREM activity during consecutive sections of the
    inter-REM interval for each single unit and then perform regression analysis to test whether 
    the given unit significantly increases or decreases throughout inter-REM.

    To calculate the NREM activity throughout inter-REM all the NREM episodes within
    one inter-REM interval are stitched together. 

    Parameters
    ----------
    units : pd.DataFrame
        each columns is a unit; the column name is the unit ID used to access
        units in $cell_info
    cell_info : pd.DataFrame
        lists for each unit (cell_info.ID) information such as brain_region
    M : np.array
        hypnogram (as returned from sleepy.load_stateidx).
    nsections : int, optional
        DESCRIPTION. The default is 5.
    nsections_rem : int, optional
        If nsections_rem > 0, also include REM_pre and REM_post using $nsections_rem
        bins
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    pzscore : bool, optional
        If True, zscore data. The default is False.
    kcut : tuple, optional
        If non-empty discard the recording from kcut[0] to kcut[1] seconds. The default is [].
    irem_dur : float, optional
        If an inter-REM interval is < $irem_dur discard this inter-REM interval. The default is 120.
    refractory_rule : bool, optional
        If True, apply refractory rule to exclude sequential REM episodes
    wake_prop_threshold : float, optional
        Maximum allowed proportion of wake states within inter-REM interval. 
        If proportion of wake > $wake_prop_threshold, then discard the given inter-REM interval
        The default is 0.5.
    border: float, optional
        Remove the last $border seconds from the inter-REM interval (to avoid potential "border" 
        effects).

    Returns
    -------
    df_trials : TYPE
        DESCRIPTION.
    dfm : pd.DataFrame
        with columns ['ID', 'section', 'fr', 'state', 'brain_region'].
    df_stats : TYPE
        DESCRIPTION.

    """
    refr_func = refr_period()
    
    dt = 2.5
    border = int(border/dt)

    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
            
    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    
    
    
    R = np.zeros((units.shape[1], len(tidx)))
    for i, unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]

    seq = sleepy.get_sequences(np.where(M==1)[0])
    data = []
    ev = 0    
    # go over all units
    for iunit,ID in enumerate(unitIDs):
        fr = R[iunit,:]
        region = cell_info[(cell_info.ID == ID)]['brain_region'].iloc[0]
        
        # go over all inter-REM episodes
        if len(seq) >= 2:
            for (si, sj) in zip(seq[0:-1], seq[1:]):                
                rem_id = si[0]
                # the refractory duration:
                rem_pre = len(si)*dt

                refr_dur = np.exp(refr_func(rem_pre))

                irem_idx = np.arange(si[-1]+1, sj[0]-border, dtype='int')                

                                
                # if inter-REM is too short, or if there's too much of wake, just continue with 
                # next inter-REM interval
                widx = np.where(M[irem_idx] == 2)[0]
                if len(irem_idx) * dt < irem_dur or len(widx) / len(irem_idx)  > wake_prop_threshold:
                    continue


                nrem_idx = np.where(M == 3)[0]
                inrem_idx = np.intersect1d(irem_idx, nrem_idx)
                
                if len(irem_idx) == 0:
                    continue

                # m - number of bins during inter-REM
                m = len(inrem_idx)                
                M_up = upsample_mx(M[inrem_idx], nsections)
                M_up = np.round(M_up)
                fr_up = upsample_mx(fr[inrem_idx], nsections)
    
                single_event_nrem = []
                for p in range(nsections):
                    # for each m consecutive bins calculate average NREM activity
                    mi = list(range(p*m, (p+1)*m))                        
                    nrem_fr = np.nanmean(fr_up[mi])
                    single_event_nrem.append(nrem_fr)
    
    
                #Total NREM duration during inter-REM duration in seconds (!)
                idur = len(inrem_idx)*dt
                
                mcut = M[inrem_idx]
                
                mcut2 = np.zeros(mcut.shape)
                mcut2[mcut==3] = dt
                mcut_csum = np.cumsum(mcut2)
                # isplit is the first point (index) that is not anymore in the refractory period
                try:
                    isplit = ((np.where(mcut_csum >= refr_dur)[0][0])-1) * 1
                    refr_perc = dt * isplit / idur

                except:
                    refr_perc = 1
                
                data += zip([ID]*nsections, [rem_id]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_nrem, ['NREM']*nsections, [idur]*nsections, [region]*nsections, [rem_pre]*nsections, [refr_perc]*nsections)

                if nsections_rem > 0:
                    # REM-pre
                    fr_pre = time_morph(fr[si], nsections_rem)
                    dur = len(si)*dt
                    
                    data += zip([ID]*nsections_rem, [rem_id]*nsections_rem, [ev]*nsections_rem, list(range(1, nsections_rem+1)), fr_pre, ['REM_pre']*nsections_rem, [rem_pre]*nsections_rem,[region]*nsections_rem, [dur]*nsections_rem, [refr_perc]*nsections_rem)            
                    #REM-post
                    fr_post = time_morph(fr[sj], nsections_rem)
                    dur = len(sj)*dt
                    data += zip([ID]*nsections_rem, [rem_id]*nsections_rem, [ev]*nsections_rem, list(range(1, nsections_rem+1)), fr_post, ['REM_post']*nsections_rem, [dur]*nsections_rem, [region]*nsections_rem, [rem_pre]*nsections, [refr_perc]*nsections_rem)            

                ev += 1
    
    df_trials = pd.DataFrame(columns = ['ID', 'rem_id', 'event', 'section', 'fr', 'state', 'idur', 'brain_region', 'rem_pre', 'refr_perc'], data=data)
    dfm = df_trials.groupby(['ID', 'section', 'state', 'brain_region']).mean().reset_index()
    dfm = dfm[['ID', 'section', 'fr', 'state', 'brain_region', 'refr_perc']]
    
    data = []
    for ID in dfm.ID.unique():
        df2 = df_trials[(df_trials.ID == ID) & (df_trials.state == 'NREM')]
        df2 = df2[~df2.fr.isna()]    
        res = stats.linregress(df2['section'], df2['fr'])
        region = df2['brain_region'].iloc[0]
        data += [[ID, 'NREM', res.slope, res.rvalue, res.pvalue, region]]
        
    df_stats = pd.DataFrame(data=data, columns=['ID', 'state', 's', 'r', 'p', 'brain_region'])
    
    return df_trials, dfm, df_stats                




def fr_stateseq(units, M, ids=[], ma_thr=10, ma_rem_exception=True, kcuts=[], 
                pzscore=True, nsmooth=0, sequence=[3,1,2], sign=['>','>','>'], 
                thres = [0,0,0], nstates=[10,10,10]):
    
    dt = 2.5
    nall = sum(nstates)
    
    
    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)

    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    

    if len(ids) == 0:
        unitIDs = [unit for unit in units.columns if '_' in unit]
        unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    else:
        unitIDs = ids

    
    R = np.zeros((units.shape[1], len(tidx)))
    for i, unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]

    seq = sleepy.get_sequences(np.where(M==1)[0])
    

    seqi = sleepy.get_sequences(np.where(M==sequence[0])[0])
    data = []
    for x in seqi:
        if x[-1]+1 < nhypno:
            if M[x[-1]+1] == sequence[1]:
                i = x[-1]+1
                while(i<nhypno-1) and M[i]==sequence[1]:
                    i+=1
                if M[i] == sequence[2]:
                    j = i
                    while (j<nhypno-1) and M[j] == sequence[2]:
                        j+=1
                    idxi = x
                    idxj = list(range(x[-1]+1,i))
                    idxk = list(range(i,j))

                    pass_thresholds = True
                    if sign[0] == '<':
                        if len(idxi)*dt >= thres[0]:
                            pass_thresholds = False
                    else:
                        if len(idxi)*dt <= thres[0]:
                            pass_thresholds = False

                    if sign[1] == '<':
                        if len(idxj)*dt >= thres[1]:
                            pass_thresholds = False
                    else:
                        if len(idxj)*dt <= thres[1]:
                            pass_thresholds = False

                    if sign[2] == '<':
                        if len(idxk)*dt >= thres[2]:
                            pass_thresholds = False
                    else:
                        if len(idxk)*dt <= thres[2]:
                            pass_thresholds = False

                    if pass_thresholds:
                        seq_id = str(x[0]) + '-' + str(j-1)
                        
                        for iunit,ID in enumerate(unitIDs):
                            fr = R[iunit,:]
                        
                            fri = time_morph(fr[idxi], nstates[0])
                            frj = time_morph(fr[idxj], nstates[1])
                            frk = time_morph(fr[idxk], nstates[2])
    
                            fr_seq = np.concatenate((fri, frj, frk))
                        
                            data += zip([ID]*nall, [seq_id]*nall, range(nall), fr_seq)
                            #pdb.set_trace()
                    
    df = pd.DataFrame(data=data, columns=['ID', 'seq_id', 'section', 'fr'])
    dfm = df[['ID', 'section', 'fr']].groupby(['ID', 'section']).mean().reset_index()

    return df, dfm



def add_refr(M, ma_thr=10, ma_rem_exception=False, kcuts=[]):
    """
    Determine all refractory periods for the given hypnogram

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is False.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    refr_vec : TYPE
        DESCRIPTION.
    M : np.array
         hypnogram without KCUTS

    """
    M = M.copy()
    
    refr_func = refr_period()

    dt = 2.5
    

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    

    nhypno = len(M)
    tidx = np.arange(0, nhypno)

    # KCUT
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
    
    
    refr_vec = np.zeros((nhypno,))
    
    data = []
    seq = sleepy.get_sequences(np.where(M==1)[0])
    if len(seq) >= 2:
        for (si, sj) in zip(seq[0:-1], seq[1:]):                
            rem_id = si[0]
            rem_pre = len(si)*dt
            refr_dur = np.exp(refr_func(rem_pre))


            irem_idx = np.arange(si[-1]+1, sj[0], dtype='int')                
            #nrem_idx = np.where(M == 3)[0]
            #inrem_idx = np.intersect1d(irem_idx, nrem_idx)

            mcut = M[irem_idx]            
            mcut2 = np.zeros(mcut.shape)
            mcut2[mcut==3] = dt
            mcut_csum = np.cumsum(mcut2)

            # inter-REM duration  
            idur = len(irem_idx)*dt

            # isplit is the last point (index) that is still in the refractory period
            isplit = ((np.where(mcut_csum >= refr_dur)[0][0])-1) * 1
            refr_perc = dt * isplit / idur

            a = si[-1]+1
            refr_vec[a:a+isplit+1] = 1
            refr_vec[a+isplit+1:sj[0]] = 2

            data += [[rem_id, rem_pre, isplit, idur, refr_perc, sj[0]]]

    df = pd.DataFrame(data=data, columns=['rem_id', 'rem_pre', 'isplit', 'irem_dur', 'ref_perc', 'irem_post'])
    return df, refr_vec, M



def refr_period(sig_level=0.01):
    from statistics import NormalDist
    
    # Define exact refractory period
    plight = True
    # Parameter for dark phase:
    # long cycles
    if not plight: 
        a_mu_long = 0.89
        b_mu_long = 80.23
        c_mu_long = 1.94
        
        a_sig_long = -0.16
        b_sig_long = 6.30
        c_sig_long = 1.25
        
        # short cycles
        a_mu_short = -0.72
        b_mu_short = 0
        c_mu_short = 7.11
        
        a_sig_short = -0.0059
        b_sig_short = 1.012
        c_sig_short = np.nan
        
        a_klong = 0.29
        b_klong = 0
        c_klong = -0.30
    
    else:
        a_mu_long = 0.62
        b_mu_long = 27.42
        c_mu_long = 3.4
        
        a_sig_long = -0.44
        b_sig_long = 134.42
        c_sig_long = 2.9
        
        # short cycles
        a_mu_short = -0.57
        b_mu_short = 0
        c_mu_short = 6.33
        
        a_sig_short = -0.0022
        b_sig_short = 0.7
        c_sig_short = np.nan
        
        a_klong = 0.17
        b_klong = 0
        c_klong = 0.14
        
    
    k_long = lambda x: np.min((a_klong * np.log(x + b_klong) + c_klong, 1))
    #k_short = lambda x: 1 - k_long(x)
            
    
    mu_long = lambda x: a_mu_long * np.log(x + b_mu_long) + c_mu_long
    #mu_short = lambda x: a_mu_short * np.log(x + b_mu_short) + c_mu_short
    
    sig_long = lambda x: a_sig_long * np.log(x + b_sig_long) + c_sig_long
    #sig_short = lambda x: a_sig_short * x + b_sig_short
    
    
    #pdf_long =  lambda ln,rem: k_long(rem)  * NormalDist(mu=mu_long(rem),  sigma=sig_long(rem)).pdf(ln)
    #pdf_short = lambda ln,rem: k_short(rem) * NormalDist(mu=mu_short(rem), sigma=sig_short(rem)).pdf(ln)
    
    
    # x_is = np.arange(15, 200)
    # ln = np.arange(0.1, 10, 0.01)
    # y_is = []
    # ln_last = 0
    # for rem in x_is:
    #     if rem < 150:
    #         y_long, y_short = [], []
    #         for n in ln:
    #             y_long.append(pdf_long(n, rem))
    #             y_short.append(pdf_short(n, rem))
                
    #         y_long = np.array(y_long)
    #         y_short = np.array(y_short)
            
    #         d = y_long-y_short
    #         i = np.where(d > 0)[0][0]
            
    #         y_is.append(ln[i])
    #         ln_last = ln[i]
    #     else:
    #         y_is.append(ln_last)
        
    ################################
    
    # #%%
    # # Define refractory period
    # x = np.arange(1, 240)
    # y1 = []
    # y2 = []
    # for i in x:
    #     y1.append( NormalDist(mu = mu_long(i), sigma = sig_long(i)).inv_cdf(0.01) )
    #     y2.append( NormalDist(mu = mu_long(i), sigma = sig_long(i)).inv_cdf(0.001) )
    

    return lambda x: NormalDist(mu = mu_long(x), sigma = sig_long(x)).inv_cdf(sig_level)



def remrem_trajectories(PC, M, nsmooth=0, pzscore=False, kcuts=[], 
                        irem_dur=120, lw=2, ma_thr=10, dt=2.5, ma_rem_exception=True,
                        ax='', coords=[0,1], m=5, wake_break=30, wake_dur=60):
    """
    
    Todo: Add option treat long wake episodes like rem
    
    """
    tidx = np.arange(0, len(M))

    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
    nhypno = np.min((len(M), PC.shape[1]))
    M = M[0:nhypno]

    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    
    # find long wake blocks:
    widx = sleepy.get_sequences(np.where(M==2)[0], ibreak=int(wake_break/dt))

    tmp = []
    for w in widx:
        if len(w) * dt > wake_dur:
            tmp += list(w)            
    widx = tmp
    M[widx] = 0
    
    if ax == '':
        plt.figure()
        ax = plt.axes([0.15, 0.15, 0.7, 0.7])
    ax_3d = False
    if ax == '3D':
        ax_3d = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    #bs_clrs = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0.2, 0.2]]

    
    # ADJUST
    seq = sleepy.get_sequences(np.where(M<=1)[0])
    for (si,sj) in zip(seq[0:m-1], seq[1:m]):
        a = si[-1]+1
        b = sj[0]
        
        s = np.arange(a,b)
        clrs = sns.color_palette("YlOrBr", len(s))
        
        if len(s)*dt >= irem_dur:
            k = a
            i = 0
            while k < b:
                ax.plot(PC[coords[0],[k,k+1]], PC[coords[1], [k,k+1]], color=clrs[i], lw=lw)
                k += 1
                i += 1
                
            #ax.plot(PC[coords[0],b:sj[-1]+1], PC[coords[1], b:sj[-1]+1], color=bs_clrs[1], lw=lw)
                
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    sns.despine()



def plot_is_trajectories(PC, mouse, kcuts=[], config_file='', ma_thr=10, pc_sign=[], lead_pc=1,
                         ma_rem_exception=True, dt=2.5, sigma=[10,15], tstart=0, tend=-1,
                         box_filt=[1,4], pnorm_spec=True, wfreq=[0.01, 0.03], 
                         wake_dur=120, wake_break=0, rem_mindur=0, irem_dur=120,
                         rem_ending=True, hilbert_lead_signal=False, clr_map='zink'):    
    """
    Based on the sigma power determine each infraslow cycle during NREM using Hilbert transform.
    Determine for each IS cycle the peak of PC1 and PC2 and 
    
    :param lead_pc: int or string,
        Use PC (lead_pc = 1, 2, ...) or sigma power (lead_pc = 'sigma')
    :param wake_dur: float, wake bouts > $wake_dur are assumed to interrupt inter-REM episodes. That is, 
        we assume that a new inter-REM episodes starts after each wake bout > $wake_dur.
    :param wake_break: 
        to define long wake bouts, we allow them to be interrupted by non-Wake episodes, of duration < $wake_break
    :param hilbert_lead_signal: bool, if True, filter and apply hilbert transform
        to the lead signal; otherwise, filter and apply Hilbert transform to sigma power.        
    """    
    from scipy.signal import hilbert
    
    if len(pc_sign) > 0:
        PC = PC.copy()
        for i,s in enumerate(pc_sign):
            PC[i,:] = PC[i,:] * s
    
    
    if len(config_file) == 0:        
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat'%name), squeeze_me=True)
    SP = P['SP']
    freq  = P['freq']
    t = P['t']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)
    M = M[tidx]
    SP = SP[:,tidx]
    t = t[tidx]
    t = t-t[0]
    Mrepr = M.copy()
    
    if len(M) > PC.shape[1]:
        M = M[0:PC.shape[1]]
        SP = SP[0:PC.shape[1]]
        t = t[0:PC.shape[1]]
        
    ################################
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(Mrepr==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (Mrepr[s[0] - 1] != 1):
                        Mrepr[s] = 3
                else:
                    Mrepr[s] = 3
                    
    ###############################

    # Calculate sigma power #########    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        sigma_pow = SP[isigma,:].mean(axis=0)
    else:
        sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
        
    istart = int(tstart/dt)
    if tend == -1:
        iend = len(M)
    else:
        iend = int(tend/dt)

    sigma_pow = sigma_pow[istart:iend]
    sigma_pow -= sigma_pow.mean()
    t = t[istart:iend]
    ####################################

    # detect rythmic peaks in PC or sigma power ###############################
    sr_is = 1/dt
    w1 = wfreq[0] / (sr_is/2.0)
    w2 = wfreq[1] / (sr_is/2.0)

    if type(lead_pc) == int:
        lead_signal = PC[lead_pc-1,:]
        if lead_pc == 1:
            tail_signal = PC[1,:]
        else:
            tail_signal = PC[0,:]
    else:
        lead_signal = sigma_pow
        tail_signal = sigma_pow
    
    if hilbert_lead_signal:
        sigma_pow_filt = sleepy.my_bpfilter(lead_signal, w1, w2)
    else:
        sigma_pow_filt = sleepy.my_bpfilter(sigma_pow, w1, w2)
    total_res = hilbert(sigma_pow_filt)

    thr = -2.5
    instantaneous_phase = np.angle(total_res)
    lidx  = np.where(instantaneous_phase[0:-2] > instantaneous_phase[1:-1])[0]
    ridx  = np.where(instantaneous_phase[1:-1] <= instantaneous_phase[2:])[0]
    thidx = np.where(instantaneous_phase[1:-1] < thr)[0]
    # sidx: indices for troughs of sigma power
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    
    #seq = sleepy.get_sequences(np.where(Mrepr==3)[0])
    #nrem_idx = []
    #for s in seq:
    #    nrem_idx += list(s)
    #idx = [s for s in sidx if s in nrem_idx]
    #idx = sidx
        
    # For each pair of troughs 
    midx = []
    for (i,j) in zip(sidx[0:-1], sidx[1:]):
        k = np.argmax(lead_signal[i:j]) + i        
        if Mrepr[i] == 3 or Mrepr[j] == 3:
            midx.append(k)
    idx = midx
    
    midx2 = []
    for (i,j) in zip(sidx[0:-1], sidx[1:]):
        k = np.argmax(tail_signal[i:j]) + i        
        if Mrepr[i] == 3 or Mrepr[j] == 3:
            midx2.append(k)
    idx2 = midx2
    
    
    ###########################################################################
    
    # idx = []
    # for s in seq:
    #     if len(s)*dt >= nrem_thr:
    #         res = total_res[s]
    #         instantaneous_phase = np.angle(res)
            
    #         # get minima in phase
    #         lidx  = np.where(instantaneous_phase[0:-2] > instantaneous_phase[1:-1])[0]
    #         ridx  = np.where(instantaneous_phase[1:-1] <= instantaneous_phase[2:])[0]
    #         thidx = np.where(instantaneous_phase[1:-1]<-1)[0]
    #         sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1

    #         sidx_corr = []
    #         for (a,b) in zip(sidx[0:-1], sidx[1:]):
    #             ii = np.where(instantaneous_phase[a:b]>1)[0]
    #             if len(ii) > 0:
    #                 sidx_corr.append(b)
            
    #         if len(sidx) >= 2:
    #             a = sidx[0]
    #             b = sidx[1]
    #             ii = np.where(instantaneous_phase[a:b]>1)[0]
    #             if len(ii) > 0:
    #                 sidx_corr = [a] + sidx_corr

    #         sidx = np.array(sidx_corr)+s[0]
    #         sidx = list(sidx)
    #         idx = idx + list(sidx)


    # figure showing EEG spectrogram, sigma power together with dots indicating 
    # the throuhgs in the sigma power
    fig, axs = plt.subplots(6, 1, sharex=True)    
    axes_brs = axs[0]
    cmap = plt.cm.jet

    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)

    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([Mrepr]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    sleepy._despine_axes(axes_brs)

    if type(lead_pc) == int:
        label = 'lead_pc' + str(lead_pc)
    else:
        label = r'$\sigma$'
    axs[1].plot(t, sigma_pow, label=label)
    axs[1].plot(t, sigma_pow_filt*3)
    axs[1].plot(t[idx], sigma_pow_filt[idx], 'r.')
    axs[1].set_ylabel('Sigma')
    
    axs[2].plot(t, PC[1,:])    
    axs[2].plot(t[idx], PC[1,idx], 'r.')
    axs[2].plot(t[sidx], PC[1,sidx], 'g.')    
    axs[2].set_ylabel('PC2')
    
    axs[3].plot(t, PC[0,:])    
    axs[3].plot(t[idx], PC[0,idx2], 'r.')
    axs[3].set_ylabel('PC1')

    axs[4].plot(t, instantaneous_phase, 'k')
    axs[4].set_ylabel('Phase')

    # find long wake blocks ###################################################
    widx = sleepy.get_sequences(np.where(Mrepr==2)[0], ibreak=int(wake_break/dt))    
    tmp = []
    for w in widx:
        v = np.arange(w[0], w[-1]+1)
        if len(v) * dt > wake_dur:            
            tmp += list(v)            
    widx = tmp
    M[widx] = 0
    
    idx = np.array(idx)
    idx2 = np.array(idx)
    seq = sleepy.get_sequences(np.where(M<=1)[0])
    irem_dur = 120
    coords = [0,1]

    plt.figure()
    ax = plt.subplot(111)
    incl_seq = []
    for (si,sj) in zip(seq[0:-1], seq[1:]):
        a = si[-1]+1
        b = sj[0]
        s = np.arange(a,b)
        
        if rem_ending and (M[b] != 1 or len(sj)*dt <= rem_mindur or len(s)*dt < irem_dur):
            continue
        
        incl_seq.append([a,b])
        sel_idx  = idx[np.where((midx >= a) & (midx < b))[0]]
        sel_idx2 = idx2[np.where((midx >= a) & (midx < b))[0]]
        
        if clr_map == 'zink':
            clrs = sns.color_palette("ch:s=-.2,r=.6", len(sel_idx))
        else:
            clrs = sns.color_palette("YlOrBr", len(sel_idx))
        
        #if len(s)*dt >= irem_dur:
        k = a
        i = 0
        for k,l in zip(sel_idx, sel_idx2):
            #ax.plot(PC[coords[0],k], PC[coords[1], k], 'o', color=clrs[i])
            ax.plot(PC[coords[0],k], PC[coords[1], l], 'o', color=clrs[i])

            axs[5].plot(t[k], PC[1,k], 'o', color=clrs[i])

            k += 1
            i += 1
                                
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    sns.despine()
    
    for (a,b) in incl_seq:
        axs[5].plot([t[a], t[b]], [0, 0], lw=2)        
    axs[5].set_ylim([-10, 20])
    


def is_cycle_outcome(PC, mouse, kcuts=[], pc_sign=[], config_file='', 
                     ma_thr=10, ma_rem_exception=True, sigma=[10,15], dt=2.5, 
                     wfreq=[0.01, 0.03], box_filt=[], pnorm_spec=True, 
                     nrem_thr=120, win=20, nstates=20, pplot=False):
    
    from scipy.signal import hilbert
    
    if len(pc_sign) > 0:
        PC = PC.copy()
        for i,s in enumerate(pc_sign):
            PC[i,:] = PC[i,:] * s
        
    if len(config_file) == 0:        
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat'%name), squeeze_me=True)
    SP = P['SP']
    freq  = P['freq']
    t = P['t']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]
    
    if SP.shape[1] > M.shape[0]:
        mmin = np.min((SP.shape[1], M.shape[0]))
        SP = SP[:,0:mmin]
        t = t[0:mmin]
        
    
    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)
    M = M[tidx]
    SP = SP[:,tidx]
    t = t[tidx]
    t = t-t[0]
    Mrepr = M.copy()
    Mma   = M.copy()
    
    if SP.shape[1] > PC.shape[1]:
        M = M[0:PC.shape[1]]
        SP = SP[:,0:PC.shape[1]]
        t = t[0:PC.shape[1]]


        
    ################################
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(Mrepr==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (Mrepr[s[0] - 1] != 1):
                        Mrepr[s] = 3
                        Mma[s] = 4
                else:
                    Mrepr[s] = 3
                    Mma[s] = 4
    ###############################

    # Calculate sigma power #########    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        sigma_pow = SP[isigma,:].mean(axis=0)
    else:
        sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
        
    sigma_pow -= sigma_pow.mean()
    ####################################

    # detect rythmic peaks in PC or sigma power ###############################
    sr_is = 1/dt
    w1 = wfreq[0] / (sr_is/2.0)
    w2 = wfreq[1] / (sr_is/2.0)

    sigma_pow_filt = sleepy.my_bpfilter(sigma_pow, w1, w2)
    total_res = hilbert(sigma_pow_filt)

    thr = -2.5
    instantaneous_phase = np.angle(total_res)
    lidx  = np.where(instantaneous_phase[0:-2] > instantaneous_phase[1:-1])[0]
    ridx  = np.where(instantaneous_phase[1:-1] <= instantaneous_phase[2:])[0]
    thidx = np.where(instantaneous_phase[1:-1] < thr)[0]
    # sidx: indices for troughs of sigma power
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1

    
    seq = sleepy.get_sequences(np.where(Mrepr==3)[0])    
    nrem_seq = [[s[0], s[-1]] for s in seq]

    def _nrem_dur(i):
        for a,b in nrem_seq:
            if a <= i <= b:
                dur =  (b-a+1)*dt
                break
            else:
                dur = -1
        
        return dur

    
    # go through each pair IS rhythm troughs
    # if the first trough is in NREM, go for it
    # check whether the second point overlaps with NREM, REM, Wake, or MA
    data = []
    iwin = int(win/dt)
    tridx = []
    for (i,j) in zip(sidx[0:-1], sidx[1:]):

        if Mrepr[i] == 3 and _nrem_dur(i) >= nrem_thr:
            m_cut = Mma[j-iwin:j+iwin]
            
            tmp = np.where(m_cut != 3)[0]
            if len(tmp) > 0:
                fstate = m_cut[tmp[0]]
            else:
                fstate = 3

            data += [[i,j,fstate]]
                        
            if not i in tridx:
                tridx.append(i)
            tridx.append(j)
            
    df_idx = pd.DataFrame(data=data, columns=['tra', 'trb', 'fstate'])

    ndim = PC.shape[0]
    state_map = {1:'REM', 2:'Wake', 3:'NREM', 4:'MA'}
    data = []
    for i, row in df_idx.iterrows():
        a = row.tra
        b = row.trb
        fstate = row.fstate
        
        for j in range(ndim):
            pc_cut = PC[j,a:b+1]
            pc_cut_morph = time_morph(pc_cut, nstates)
            pc_label = 'pc%d' % (j+1)
            
            data += zip([i]*nstates, np.arange(nstates), pc_cut_morph, [pc_label]*nstates, [state_map[fstate]]*nstates)
    
    df_is = pd.DataFrame(data=data, columns=['ev', 'bins', 'fr', 'pc', 'fstate'])

    if pplot:

        # figure showing EEG spectrogram, sigma power together with dots indicating 
        # the throuhgs in the sigma power
        fig, axs = plt.subplots(6, 1, sharex=True)    
        axes_brs = axs[0]
        cmap = plt.cm.jet
    
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([Mrepr]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        sleepy._despine_axes(axes_brs)
    
        label = r'$\sigma$'
        axs[1].plot(t, sigma_pow, label=label)
        axs[1].plot(t, sigma_pow_filt*3)
        axs[1].plot(t[sidx], sigma_pow_filt[sidx], 'r.')
        axs[1].set_ylabel('Sigma')
        
        axs[2].plot(t, PC[1,:])    
        axs[2].plot(t[sidx], PC[1,sidx], 'r.')    
        axs[2].set_ylabel('PC1')
        
        axs[3].plot(t, PC[0,:])    
        axs[3].plot(t[sidx], PC[0,sidx], 'r.')
        axs[3].set_ylabel('PC2')
    
        axs[4].plot(t, instantaneous_phase, 'k')        
        axs[4].set_ylabel('Phase')

                    
    return df_idx, df_is


    
def sigmaramp_pca(units, M, mouse, config_file='', kcuts=[], ma_thr=10, ma_rem_exception=True,
                  box_filt=[], pnorm_spec=True, wfreq=[0.01, 0.03], wake_dur=120, wake_break=0, 
                  irem_dur=120, rem_mindur=0, rem_ending=False,
                  nsmooth=0, detrend=False, pzscore=True, win=0,
                  pc_sign=[], ndim=2, pplot=False):
    """
    For each NREM block (ranging from the end of a REM episode or long wake bout, defined
    by $wake_dur and $wake_break, till the beginning of the next wake) determine all the infraslow (IS) peaks
    for each unit and then perform using only the firing rates at IS peaks.

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    config_file : TYPE, optional
        DESCRIPTION. The default is ''.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is True.
    box_filt : TYPE, optional
        DESCRIPTION. The default is [].
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is True.
    wfreq : TYPE, optional
        DESCRIPTION. The default is [0.01, 0.03].
    wake_dur : TYPE, optional
        DESCRIPTION. The default is 120.
    wake_break : TYPE, optional
        DESCRIPTION. The default is 0.
    irem_dur : TYPE, optional
        DESCRIPTION. The default is 120.
    rem_mindur : TYPE, optional
        DESCRIPTION. The default is 0.
    rem_ending : TYPE, optional
        DESCRIPTION. The default is False.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    detrend : TYPE, optional
        DESCRIPTION. The default is False.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    pc_sign : TYPE, optional
        DESCRIPTION. The default is [].
    ndim : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    PC : TYPE
        DESCRIPTION.

    """

    from scipy.signal import hilbert
    sigma = [0,15]
    dt = 2.5  
    
    if len(config_file) == 0:        
        config_file = 'mouse_config.txt'
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat'%name), squeeze_me=True)
    SP = P['SP']
    freq  = P['freq']
    t = P['t']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]

     
    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)        
    ###########################################################################    
    SP = SP[:,tidx]
    t = t[tidx]
    t = t-t[0]
    Mrepr = M.copy()
    
    # flatten out MAs ##############
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(Mrepr==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (Mrepr[s[0] - 1] != 1):
                        Mrepr[s] = 3
                else:
                    Mrepr[s] = 3
    ###############################


    # Calculate sigma power #########    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        sigma_pow = SP[isigma,:].mean(axis=0)
    else:
        sigma_pow = SP[isigma, :].sum(axis=0)*dfreq        
    ####################################

    # detect rythmic troughs in sigma power ###################################
    sr_is = 1/dt
    w1 = wfreq[0] / (sr_is/2.0)
    w2 = wfreq[1] / (sr_is/2.0)

    lead_signal = sigma_pow
    
    sigma_pow_filt = sleepy.my_bpfilter(lead_signal, w1, w2)
    total_res = hilbert(sigma_pow_filt)

    thr = -2.5
    instantaneous_phase = np.angle(total_res)
    lidx  = np.where(instantaneous_phase[0:-2] > instantaneous_phase[1:-1])[0]
    ridx  = np.where(instantaneous_phase[1:-1] <= instantaneous_phase[2:])[0]
    thidx = np.where(instantaneous_phase[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
                
    midx, data_idx = [], []
    for (i,j) in zip(sidx[0:-1], sidx[1:]):
        k = np.argmax(lead_signal[i:j]) + i
        if Mrepr[i] == 3 or Mrepr[j] == 3:
            midx.append(k)
            data_idx += [[k, i,j]]
            
    idx = midx
    df_idx = pd.DataFrame(data=data_idx, columns=['smax', 'tra', 'trb'])
    ###########################################################################

    # find long wake blocks ###################################################
    widx = sleepy.get_sequences(np.where(Mrepr==2)[0], ibreak=int(wake_break/dt)+1)    
    tmp = []
    for w in widx:
        v = np.arange(w[0], w[-1]+1)
        if len(v) * dt > wake_dur:
            
            tmp += list(v)            
    widx = tmp
    M[widx] = 0
    
    idx = np.array(idx)
    seq = sleepy.get_sequences(np.where(M<=1)[0])
    irem_dur = 120

    sel_idx = []
    df_idx['rem_end'] = -1
    for (si,sj) in zip(seq[0:-1], seq[1:]):
        a = si[-1]+1
        b = sj[0]
        s = np.arange(a, b)
            
        if M[b] != 1:
            print("here")
    
        if rem_ending and (M[b] != 1 or len(sj)*dt <= rem_mindur or len(s)*dt < irem_dur):
            print("here")
            continue
        
        conf_idx = idx[np.where((idx >= a) & (idx < b))[0]]
        sel_idx += list(conf_idx)
        
        df_idx.loc[df_idx.smax.isin(conf_idx),'rem_end'] = b
        
    sel_idx = np.array(sel_idx)
    df_idx = df_idx[df_idx['smax'].isin(sel_idx)]
    df_idx.reset_index(inplace=True)
    
    # process firing rates ####################################################
    unitIDs = [unit for unit in units.columns if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    nsample = nhypno           # number of time points
    nvar    = len(unitIDs)     # number of units
    R = np.zeros((nvar, nsample)) # neurons x time
    #@tidx are the indices we're further considering.

    for i,unit in enumerate(unitIDs):
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        #tmp = smooth_causal(np.array(units[unit]), nsmooth)
        tmp = tmp[tidx]
        if detrend:
            tmp = scipy.signal.detrend(tmp)

        if pzscore:
            R[i,:] = (tmp - tmp.mean()) / tmp.std()
        else:
            R[i,:] = tmp
            
    # For matrix Y (subjected to PCA) only keep firing rates at the peak of
    # each IS cycle
    iwin = int(win/dt)
    tmp = []
    for i in sel_idx:
        if iwin > 1:
            C = R[:,i-iwin:i+iwin].mean(axis=1)
        else:
            C = R[:,i]        
        tmp.append(C)        
    Y = np.array(tmp).T
    
    # divide by sqrt(nsample - 1) to make SVD equivalent to PCA
    Y = Y.T / np.sqrt(nsample-1)
    # make sure the the columns of Y are mean-zero
    for i in range(Y.shape[1]):
        Y[:,i] = Y[:,i] - Y[:,i].mean()
    
    # SVD
    U,S,Vh = scipy.linalg.svd(Y)
    
    # each row in Vh is an eigenvector of the COV matrix:    
    PC = np.dot(Vh, R)[0:ndim,:]
    V = S**2    
            
    if len(pc_sign) > 0:
        i = 0
        for s in pc_sign:
            PC[i,:] = PC[i,:] * s
            i += 1
    
    collapse=False
    if pplot:
        plt.figure()
        #sleepy._despine_axes(axes_exc)                
        axes_brs = plt.axes([0.2, 0.85, 0.7, 0.05])        
        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes_brs.axis('tight')
        sleepy._despine_axes(axes_brs)

        if collapse:    
            plt.axes([0.2, 0.2, 0.7, 0.6], sharex=axes_brs)
            plt.plot(t, PC[:,:].T)
            plt.xlim((t[0], t[-1]))
            sns.despine()
            plt.xlabel('Time (s)')
            plt.ylabel('PC')
        
        else:
            clrs = sns.color_palette("husl", ndim)
            
            d = (0.6 / ndim) * 0.3
            ny = (0.6 / ndim)-d
            for i in range(ndim):
                ax = plt.axes([0.2, 0.2+i*(ny+d), 0.7, ny], sharex=axes_brs)
                ax.plot(t, PC[ndim-1-i,:], color=clrs[i])    
                plt.xlim([t[0], t[-1]])
                sleepy.box_off(ax)
                plt.ylabel('PC%d' % (ndim-i))
    

                if i > 0:
                    ax.spines["bottom"].set_visible(False)
                    ax.axes.get_xaxis().set_visible(False)
                else:
                    plt.xlabel('Time (s)')
                

        var = S**2
        var_total = np.sum(S**2)
        
        # Calculate the cumulative variance explained, i.e. how much
        # of the variance is captured by the first i principal components. 
        p = []
        for i in range(1,len(var)+1):
            s = np.sum(var[0:i])
            p.append(s/var_total)
                
        plt.figure(figsize=(4,4))
        plt.plot(p, '.', color='gray')
        plt.xlabel(r'$\mathrm{PC_i}$')
        plt.ylabel('Cum. variance')    
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.ylim([0, 1.1])
        sns.despine()
    
    return PC, V, Vh, sel_idx, df_idx
            

    
def state_correlation(fr1, fr2, M, istate=3, win=60, tbreak=10, dt=2.5, pplot=True):
    """
    Calculate cross-correlation for the two vectors @fr1 and @fr2

    Note: Negative time points in the cross-correlation mean that 
    the fr1 precedes fr2

    Parameters
    ----------
    fr1 : np.array
        Single vector for shape (n,).
    fr2 : np.array
        Single vector for shape (n,).
    M : np.array
        hypnogram.
    istate : 1, 2, or 3, optional
        Calculate correlation only for sequences of REM (1), 
        Wake (2), or NREM (3). The default is 3.
    win : float, optional
        The cross-correlation ranges from -win to +win seconds. The default is 60.
    tbreak : TYPE, optional
        DESCRIPTION. The default is 10.
    dt : float, optional
        Time bin for firing rates. The default is 2.5.
    pplot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    CC : np.array
        Cross-correlation for each NREM episode that's long enough; trials x time
    t : np.array
        time vector.
    """        
    seq = sleepy.get_sequences(np.where(M == istate)[0], ibreak=int(tbreak/dt)+1)
    seq = [s for s in seq if len(s)*dt > 2*win]

    iwin = int(win / dt)
    CC = []
    for s in seq:
        fr1cut = fr1[s]
        fr2cut = fr2[s]
        
        fr1cut = fr1cut - fr1cut.mean()
        fr2cut = fr2cut - fr2cut.mean()
        
        
        m = np.min([fr1cut.shape[0], fr2cut.shape[0]])
        # Say we correlate x and y;
        # x and y have length m
        # then the correlation vector cc will have length 2*m - 1
        # the center element with lag 0 will be cc[m-1]
        norm = np.nanstd(fr1cut) * np.nanstd(fr2cut)
        # for used normalization, see: https://en.wikipedia.org/wiki/Cross-correlation
        
        #xx = scipy.signal.correlate(dffd[1:m], pow_band[0:m - 1])/ norm
        xx = (1/m) * scipy.signal.correlate(fr1cut, fr2cut)/ norm
        if norm == 0:
            xx[:] = np.nan
        
        
        ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
        ii = [int(i) for i in ii]
        
        ii = np.concatenate((np.arange(m-iwin-1, m), np.arange(m, m+iwin, dtype='int')))
        # note: point ii[iwin] is the "0", so xx[ii[iwin]] corresponds to the 0-lag correlation point
        CC.append(xx[ii])

    CC = np.array(CC)
    t = np.arange(-iwin, iwin+1) * dt

    if pplot:
        plt.figure()
        ax = plt.subplot(111)
        #t = np.arange(-iwin, iwin + 1) * dt
        # note: point t[iwin] is "0"
        c = np.sqrt(CC.shape[0])
        a = np.nanmean(CC, axis=0) - np.nanstd(CC, axis=0)/c
        b = np.nanmean(CC, axis=0) + np.nanstd(CC, axis=0)/c
        plt.fill_between(t,a,b, color='gray', alpha=0.5)
        plt.plot(t, np.nanmean(CC, axis=0), color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Norm. CC')
        plt.xlim([t[0], t[-1]])
        sleepy.box_off(ax)
        plt.show()

    return CC, t



def state_correlation_avg(units, ids1, ids2, M, kcuts=[], istate=3, win=60, tbreak=10, nsmooth=0, 
                          pzscore=True, dt=2.5, pplot=True, self_correlation=False, 
                          config_file='', mouse=''):
    """
    State-dependent cross-correlation. 
    Autocorrelate each pair of units within two sets of units (@ids1 and @ids2). 

    NOTE: Negative time points mean that @ids1 precede @ids2

    Parameters
    ----------
    units : pd.DataFrame
        each column is one unit with the unit ID as label.
    ids1 : list
        IDs of units in set 1.
    ids2 : list
        IDs of units in set 2.
    M : np.array
        hypnogram.
    kcuts : list of tuples, optional
        Cut out the specific areas from the recording. The default is [].
    istate : TYPE, optional
        DESCRIPTION. The default is 3.
    win : TYPE, optional
        DESCRIPTION. The default is 60.
    tbreak : TYPE, optional
        DESCRIPTION. The default is 10.
    nsmooth : float, optional
        Smoothing parameter passed to &sleepy.smooth_data(). The default is 0.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    pplot : bool, optional
        If True, generate figure showing cross-correlation. The default is True.
    self_correlation : bool, optional
        If True, sets ids1 and ids2 are identical. In this case, correlate
        each possible pair only once. 
        The default is False.

    Returns
    -------
    df : pd.DataFrame
        holding the (averaged) cross-correlation for each pair of units,
        with columns ['time', 'cc', 'label', 'id1', 'id2'].
    dfr : pd.DataFrame
        holding the maximum or minimum peak value in the cross-correlation,
        with columns ['time', 'cc', 'label', 'id1', 'id2']

    """

    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)        
    ###########################################################################    



    # ###########################################################################
    # if len(units) == 0:
    #     fine_scale = True
        
    #     tr_path = load_config('mouse_config.txt')[mouse]['TR_PATH']
    #     units = np.load(os.path.join(tr_path,'1k_train.npz')) 
    #     unitIDs = [unit for unit in list(units.keys()) if '_' in unit]
    #     unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    #     dt = dt / NUP
    #     M = upsample_mx(M, NUP)
    #     nhypno = int(np.min((len(M), units[unitIDs[0]].shape[0]/NDOWN)))

    # else:
    #     unitIDs = [unit for unit in units.columns if '_' in unit]
    #     unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    #     nhypno  = np.min((len(M), units.shape[0]))

    
    fr1 = np.array(units[ids1]).T
    fr2 = np.array(units[ids2]).T
        
    fr1_new = np.zeros((len(ids1), nhypno))
    fr2_new = np.zeros((len(ids2), nhypno))

    for i in range(fr1.shape[0]):
        if nsmooth > 0:
            fr1[i,:] = sleepy.smooth_data(fr1[i,:], nsmooth)
        if pzscore:
            fr1_new[i,:] = (fr1[i,tidx]-fr1[i,tidx].mean()) / fr1[i,tidx].std()

    for i in range(fr2.shape[0]):
        if nsmooth > 0:        
            fr2[i,:] = sleepy.smooth_data(fr2[i,:], nsmooth)
        if pzscore:
            fr2_new[i,:] = (fr2[i,tidx]-fr2[i,tidx].mean()) / fr2[i,tidx].std()
    
    fr1 = fr1_new
    fr2 = fr2_new
    
    data = []
    if not self_correlation:
        for i in range(fr1.shape[0]):
            for j in range(fr2.shape[0]):        
                cc, t = state_correlation(fr1[i,:], fr2[j,:], M, 
                                          win=win, istate=istate, tbreak=tbreak, pplot=pplot)
                
                cc = np.nanmean(cc, axis=0)            
                label = ids1[i] + ' x ' + ids2[j]
                m = len(t)
                data += zip(t, cc, [label]*m, [ids1[i]]*m, [ids2[j]]*m)
    else:
        for i in range(fr1.shape[0]-1):
            for j in range(i+1, fr2.shape[0]):        
                cc, t = state_correlation(fr1[i,:], fr2[j,:], M, 
                                          win=win, istate=istate, tbreak=tbreak, pplot=pplot)
                cc = np.nanmean(cc, axis=0)            
                label = ids1[i] + ' x ' + ids2[j]
                m = len(t)
                data += zip(t, cc, [label]*m, [ids1[i]]*m, [ids2[j]]*m)
                    
    df = pd.DataFrame(data=data, columns=['time', 'cc', 'label', 'id1', 'id2'])    
    labels = df['label'].unique().tolist()
    
    data = []
    for l in labels:
        dfs = df[df.label == l]
        imax = np.argmax(np.abs(dfs['cc']))
        tmax = dfs.iloc[imax]['time']
        ccmax = dfs.iloc[imax]['cc']
        id1 = dfs.iloc[imax]['id1']
        id2 = dfs.iloc[imax]['id2']        
        sgn = np.sign(ccmax)        
        data += [[tmax, ccmax, l, id1, id2, sgn]]    
    dfr = pd.DataFrame(data=data, columns=['time', 'cc', 'label', 'id1', 'id2', 'sgn'])
        
    return df, dfr



def state_correlation_avg_fine(ids1, ids2, mouse, kcuts=[], istate=3, win=60, tbreak=10, nsmooth=0, 
                          pzscore=True, dt=2.5, pplot=True, self_correlation=False, 
                          config_file='mouse_config.txt'):
    dt = 2.5  
    NDOWN = 250
    NUP = int(dt / (0.001 * NDOWN))

    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]


    # # flatten out MAs #########################################################
    # if ma_thr>0:
    #     seq = sleepy.get_sequences(np.where(M==2)[0])
    #     for s in seq:
    #         if np.round(len(s)*dt) <= ma_thr:
    #             if ma_rem_exception:
    #                 if (s[0]>1) and (M[s[0] - 1] != 1):
    #                     M[s] = 3
    #             else:
    #                 M[s] = 3
    # ###########################################################################    
    tr_path = load_config(config_file)[mouse]['TR_PATH']
    #units = np.load(os.path.join(tr_path,'1k_train.npz')) 

    if os.path.isfile(os.path.join(tr_path,'1k_train.npz')):
        units = np.load(os.path.join(tr_path,'1k_train.npz')) 
    else:
        units = np.load(os.path.join(tr_path,'lfp_1k_train.npz'))     


    unitIDs = [unit for unit in list(units.keys()) if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    
    dt = dt / NUP
    M = upsample_mx(M, NUP)
    nhypno = int(np.min((len(M), units[unitIDs[0]].shape[0]/NDOWN)))


    M = M[0:nhypno]
    tidx = np.arange(0, nhypno)

    ###########################################################################    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    

    # REWRITE:

    # fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % NDOWN)
    # if not os.path.isfile(fr_file):
    #     for i,unit in enumerate(unitIDs):
    #         tmp = sleepy.downsample_vec(np.array(units[unit]), NDOWN)            
    #         R[i,:] = tmp[tidx]
    #     so.savemat(fr_file, {'R':R, 'ndown':NDOWN})
    # else:
    #     R = so.loadmat(fr_file, squeeze_me=True)['R']
        

    fr1_new = np.zeros((len(ids1), nhypno))
    fr2_new = np.zeros((len(ids2), nhypno))

    print('Starting downsampling, smoothing and z-scoring')

    for i,ID in enumerate(ids1):
        fr1 = np.array(units[ID])
        fr1 = sleepy.downsample_vec(fr1, NDOWN)            
        
        if nsmooth > 0:
            fr1 = sleepy.smooth_data(fr1, nsmooth)
        if pzscore:
            fr1 = (fr1[tidx]-fr1[tidx].mean()) / fr1[tidx].std()
        fr1_new[i,:] = fr1

    for i,ID in enumerate(ids2):
        fr2 = np.array(units[ID])
        fr2 = sleepy.downsample_vec(fr2, NDOWN)            

        if nsmooth > 0:
            fr2 = sleepy.smooth_data(fr2, nsmooth)
        if pzscore:
            fr2 = (fr2[tidx]-fr2[tidx].mean()) / fr2[tidx].std()
        fr2_new[i,:] = fr2
    
    print('done.')
    
    fr1 = fr1_new
    fr2 = fr2_new

    data = []
    for i in range(fr1.shape[0]):
        for j in range(fr2.shape[0]):        
            cc, t = state_correlation(fr1[i,:], fr2[j,:], M, 
                                      win=win, istate=istate, tbreak=tbreak, pplot=pplot, dt=dt)
            
            cc = np.nanmean(cc, axis=0)            
            label = ids1[i] + ' x ' + ids2[j]
            m = len(t)
            data += zip(t, cc, [label]*m, [ids1[i]]*m, [ids2[j]]*m)
            
    df = pd.DataFrame(data=data, columns=['time', 'cc', 'label', 'id1', 'id2'])    
    labels = df['label'].unique().tolist()
    
    data = []
    for l in labels:
        dfs = df[df.label == l]
        imax = np.argmax(np.abs(dfs['cc']))
        tmax = dfs.iloc[imax]['time']
        ccmax = dfs.iloc[imax]['cc']
        id1 = dfs.iloc[imax]['id1']
        id2 = dfs.iloc[imax]['id2']        
        sgn = np.sign(ccmax)        
        data += [[tmax, ccmax, l, id1, id2, sgn]]    
    dfr = pd.DataFrame(data=data, columns=['time', 'cc', 'label', 'id1', 'id2', 'sgn'])

    return df, dfr



def plot_firingrates(units, cell_info, ids, mouse, 
                     config_file, tlegend=60, pzscore=True, 
                     kcuts=[], tstart=0, tend=-1,
                     pind_axes=True, dt=2.5, nsmooth=0, 
                     pnorm_spec=False, box_filt=[], ma_thr=20, ma_rem_exception=True,
                     vm=[], fmax=20, print_unit=False, show_sigma=False):
    """
    Plot firing rates along with EEG spectrogram and hypnogram.

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    cell_info : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    config_file : TYPE
        DESCRIPTION.
    tlegend : TYPE, optional
        DESCRIPTION. The default is 60.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    tstart : TYPE, optional
        DESCRIPTION. The default is 0.
    tend : TYPE, optional
        DESCRIPTION. The default is -1.
    pind_axes : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is False.
    box_filt : TYPE, optional
        DESCRIPTION. The default is [].
    vm : TYPE, optional
        DESCRIPTION. The default is [].
    fmax : TYPE, optional
        DESCRIPTION. The default is 20.
    print_unit : TYPE, optional
        DESCRIPTION. The default is False.
    show_sigma: bool, optional
        If True, also show sigma power below spectrogram

    Returns
    -------
    None.

    """

    ids = list(ids)
    yticks = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    plt.figure(figsize=(10,10))
    sleepy.set_fontarial()

    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]
    if len(M) > units.shape[0]:
        M = M[0:-1]
    
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################

    
    # brain regions corresponding to the unit IDs in @ids:
    regions = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in ids]
    
    tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
    SP = tmp['SP']
    freq = tmp['freq']
    ifreq = np.where(freq <= fmax)[0]

    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        
    # cut out kcuts: ###############
    tidx = kcut_idx(M, units, kcuts)
    M = M[tidx]
    units = units.iloc[tidx,:]
    SP = SP[:,tidx]
    ################################
    
    # set istart and iend
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    istart = int(np.round(tstart/dt))         

    M = M[istart:iend]
    # NOTE units.loc[i:j,:] INCLUDES the j-th element, 
    # that's different from how np.arrays behave
    units = units.iloc[istart:iend,:]
    SP = SP[:,istart:iend]

    nunits = len(ids)
    ntime = units.shape[0]
    fr = np.array(units[ids]).T
    if pzscore:
        for i in range(fr.shape[0]):
            fr[i,:] = (fr[i,:] - fr[i,:].mean()) / fr[i,:].std()
    if nsmooth > 0:
        for i in range(fr.shape[0]):
            fr[i,:] = sleepy.smooth_data(fr[i,:], nsmooth)


    nhypno = np.min((len(M), ntime))
    M = M[0:nhypno]
    
    t = np.arange(0, nhypno)*dt    
    
    axes_brs = plt.axes([0.1, 0.95, 0.8, 0.03])
    
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # plot spectrogram
    # calculate median for choosing right saturation for heatmap
    med = np.median(SP.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.0]    
    axes_spec = plt.axes([0.1, 0.85, 0.8, 0.08], sharex=axes_brs)    
    # axes for colorbar
    axes_cbar = plt.axes([0.9, 0.85, 0.05, 0.08])

    im = axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq,:], vmin=vm[0], vmax=vm[1], cmap='jet')
    sleepy.box_off(axes_spec)
    axes_spec.axes.get_xaxis().set_visible(False)
    axes_spec.spines["bottom"].set_visible(False)
    axes_spec.set_ylabel('Freq. (Hz)')
    
    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, location='right')
    if pnorm_spec:
        #cb.set_label('Norm. power')
        cb.set_label('')
    else:
        cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    axes_cbar.set_alpha(0.0)
    sleepy._despine_axes(axes_cbar)
    
    if not show_sigma:    
        yrange = 0.73    
    else:
        yrange = 0.63
        sigma = [10, 15]
        isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]

        if pnorm_spec:            
            
            sigma_pow = SP[isigma,:].mean(axis=0)        
        else:
            dfreq = freq[2] - freq[1]
            sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
                    
        # show sigmapower
        axes_sig = plt.axes([0.1, 0.75, 0.8, 0.07], sharex=axes_brs)
        
        #pdb.set_trace()
        axes_sig.plot(t, sigma_pow, color='gray')
        axes_sig.set_xlim([t[0], t[-1]])
        axes_sig.set_ylabel('$\mathrm{\sigma}$ power')

        axes_sig.spines["top"].set_visible(False)
        axes_sig.spines["right"].set_visible(False)
        axes_sig.spines["bottom"].set_visible(False)
        axes_sig.axes.get_xaxis().set_visible(False)
        
    if pind_axes:
        palette = sns.color_palette("husl", nunits)
        yborder = (yrange/nunits)*0.3        
        ax = []
        for i in range(nunits):
            tmp = plt.axes([0.1, 0.1+(yrange/nunits)*i, 0.8, yrange/nunits-yborder], sharex=axes_brs)
            
            tmp.axes.get_xaxis().set_visible(False)
            tmp.spines["bottom"].set_visible(False)
            tmp.spines["top"].set_visible(False)
            tmp.spines["right"].set_visible(False)            
            ax.append(tmp)     
       
        for i in range(nunits):
            a = ax[i]   
            perc = np.percentile(fr[i,istart:iend], 99)
            d = yticks - perc
            ii = np.where(d < 0)[0][-1]
            ytick = yticks[ii]       
            #ytick = yticks[np.argmin((np.abs(yticks - perc)))]

            a.set_yticks([0, ytick], ['', ytick])
            a.set_ylim([-1, np.max(fr[i,:])])
            a.plot(t,fr[i,:], color=palette[i])
    

        # axes for time legend
        axes_legend = plt.axes([0.1,0.06,0.8,0.02], sharex=axes_brs)
        plt.plot([0, tlegend], [1, 1], lw=2, color='k')
        plt.ylim([-1, 1])
        axes_legend.text(tlegend/2, -2, '%d s' % tlegend, verticalalignment='bottom', horizontalalignment='center')

        plt.xlim((t[0], t[-1]))
        sleepy._despine_axes(axes_legend)

        axes_lbs = plt.axes([0.9, 0.1, 0.1, yrange])
        for i in range(nunits):
            if print_unit:
                axes_lbs.text(0.05, i+0.25, ids[i] + ' ' + regions[i])
            else:
                axes_lbs.text(0.05, i+0.25, regions[i])
        axes_lbs.set_xlim([0, 1])
        axes_lbs.set_ylim(0, nunits)
        sleepy._despine_axes(axes_lbs)
        
        plt.gcf().text(0.03,0.1+yrange/2, 'Firing rates (spikes/s)', rotation=90, verticalalignment='center')

    else:
        max_list = []
        min_list = []
        frm = fr.mean(axis=0)
        fr = np.vstack((frm, fr))

        for i in range(nunits+1):
            a = fr[i,:]
            max_list.append(a.max())
            min_list.append(a.min())
        
        max_list = np.array(max_list)
        min_list = np.array(min_list)

        ax = plt.axes([0.1, 0.1, 0.8, yrange], sharex=axes_brs)

        axes_lbs = plt.axes([0.9, 0.1, 0.1, yrange], sharey=ax)
        axes_frs = plt.axes([0.01, 0.1, 0.082, yrange], sharey=ax)

        mean_mx = np.zeros((fr.shape[0], 3))
        for j in range(1, 4):
            idx = np.where(M==j)[0]
            mean_mx[:,j-1] = fr[:,idx].mean(axis=1)

        offset = 0
        for i in range(fr.shape[0]):
            f = fr[i,:]
            r = f - min_list[i]
            r = r / (max_list[i] + np.abs(min_list[i]))            
            ax.plot(t, r+offset)
            offset += 1
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim(0, nunits+1)
        #sleepy._despine_axes(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel('Time (s)')
    
        axes_frs.pcolorfast(mean_mx)
        axes_frs.set_xticks([0.5,1.5,2.5])
        axes_frs.set_xticklabels(['R', 'W', 'N'])
        axes_frs.set_xlim([0, 3])
        #axes_frs.set_ylim([0, fr.shape[0]])
        sns.despine()
    
        offset = 0
        ids = ['mean'] + ids
        regions = [' '] + regions
        for i in range(fr.shape[0]):
            axes_lbs.text(0.05, offset, ids[i] + ' ' + regions[i], fontsize=8)
            offset += 1
        axes_lbs.set_xlim([0, 1])
        axes_lbs.set_ylim([0, nunits+1])
        sleepy._despine_axes(axes_lbs)

        

def plot_firingrates_withemg(units, cell_info, ids, mouse, 
                     config_file, tlegend=60, pzscore=True, 
                     kcuts=[], tstart=0, tend=-1,
                     pind_axes=True, dt=2.5, nsmooth=0, 
                     pnorm_spec=False, box_filt=[], ma_thr=20, ma_rem_exception=True,
                     vm=[], fmax=20, print_unit=False, show_sigma=False):
    """
    Plot firing rates along with EEG spectrogram, EMG amplitude, and hypnogram.

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    cell_info : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    config_file : TYPE
        DESCRIPTION.
    tlegend : TYPE, optional
        DESCRIPTION. The default is 60.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    tstart : TYPE, optional
        DESCRIPTION. The default is 0.
    tend : TYPE, optional
        DESCRIPTION. The default is -1.
    pind_axes : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is False.
    box_filt : TYPE, optional
        DESCRIPTION. The default is [].
    vm : TYPE, optional
        DESCRIPTION. The default is [].
    fmax : TYPE, optional
        DESCRIPTION. The default is 20.
    print_unit : TYPE, optional
        DESCRIPTION. The default is False.
    show_sigma: bool, optional
        If True, also show sigma power below spectrogram

    Returns
    -------
    None.

    """
    # range for EMG amplitude:
    r_mu = [5, 50]
    ids = list(ids)
    yticks = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    plt.figure(figsize=(10,10))
    sleepy.set_fontarial()

    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]
    if len(M) > units.shape[0]:
        M = M[0:-1]
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################
    
    # brain regions corresponding to the unit IDs in @ids:
    regions = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in ids]
    
    tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
    SP = tmp['SP']
    freq = tmp['freq']
    ifreq = np.where(freq <= fmax)[0]

    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

    # load EMG
    tmp = so.loadmat(os.path.join(ppath, file, 'msp_%s.mat' % file), squeeze_me=True)
    SPEMG = tmp['mSP']
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0 # back to muV
    # END[load EMG]


    # cut out kcuts: ###############
    tidx = kcut_idx(M, units, kcuts)
    M = M[tidx]
    units = units.iloc[tidx,:]
    SP = SP[:,tidx]
    ################################
    
    # set istart and iend
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    istart = int(np.round(tstart/dt))         

    M = M[istart:iend]
    # NOTE units.loc[i:j,:] INCLUDES the j-th element, 
    # that's different from how np.arrays behave
    units = units.iloc[istart:iend,:]
    SP = SP[:,istart:iend]

    nunits = len(ids)
    ntime = units.shape[0]
    fr = np.array(units[ids]).T
    if pzscore:
        for i in range(fr.shape[0]):
            fr[i,:] = (fr[i,:] - fr[i,:].mean()) / fr[i,:].std()
    if nsmooth > 0:
        for i in range(fr.shape[0]):
            fr[i,:] = sleepy.smooth_data(fr[i,:], nsmooth)


    nhypno = np.min((len(M), ntime))
    M = M[0:nhypno]
    
    t = np.arange(0, nhypno)*dt    
    
    axes_brs = plt.axes([0.1, 0.95, 0.8, 0.03])
    
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # plot spectrogram
    # calculate median for choosing right saturation for heatmap
    med = np.median(SP.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.0]    
    axes_spec = plt.axes([0.1, 0.85, 0.8, 0.08], sharex=axes_brs)    
    # axes for colorbar
    axes_cbar = plt.axes([0.9, 0.85, 0.05, 0.08])

    im = axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq,:], vmin=vm[0], vmax=vm[1], cmap='jet')
    sleepy.box_off(axes_spec)
    axes_spec.axes.get_xaxis().set_visible(False)
    axes_spec.spines["bottom"].set_visible(False)
    axes_spec.set_ylabel('Freq.\n(Hz)')
    
    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, location='right')
    if pnorm_spec:
        #cb.set_label('Norm. power')
        cb.set_label('')
    else:
        cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    axes_cbar.set_alpha(0.0)
    sleepy._despine_axes(axes_cbar)
    
    # Show EMG amplitude
    axes_emg = plt.axes([0.1, 0.75+0.02, 0.8, 0.06], sharex=axes_brs)
    axes_emg.plot(t, p_mu[istart:iend], color='black')    
    sleepy.box_off(axes_emg)
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    axes_emg.set_xticks([])
    #emg_ylim = [0, np.max(p_mu[istart:iend])]
    #axes_emg.set_ylim(emg_ylim)
    axes_emg.set_ylabel('Ampl.\n($\mathrm{\mu}$V)')
    
    if not show_sigma:    
        yrange = 0.7
    else:
        yrange = 0.59
        sigma = [10, 15]
        isigma = np.where((freq>=sigma[0])&(freq<=sigma[1]))[0]

        if pnorm_spec:            
            
            sigma_pow = SP[isigma,:].mean(axis=0)        
        else:
            dfreq = freq[2] - freq[1]
            sigma_pow = SP[isigma, :].sum(axis=0)*dfreq
                    
        # show sigmapower
        axes_sig = plt.axes([0.1, 0.7, 0.8, 0.05], sharex=axes_brs)
        
        #pdb.set_trace()
        axes_sig.plot(t, sigma_pow, color='gray')
        axes_sig.set_xlim([t[0], t[-1]])
        axes_sig.set_ylabel('$\mathrm{\sigma}$ power')

        axes_sig.spines["top"].set_visible(False)
        axes_sig.spines["right"].set_visible(False)
        axes_sig.spines["bottom"].set_visible(False)
        axes_sig.axes.get_xaxis().set_visible(False)
        
    if pind_axes:
        palette = sns.color_palette("husl", nunits)
        yborder = (yrange/nunits)*0.3        
        ax = []
        for i in range(nunits):
            tmp = plt.axes([0.1, 0.1+(yrange/nunits)*i, 0.8, yrange/nunits-yborder], sharex=axes_brs)
            
            tmp.axes.get_xaxis().set_visible(False)
            tmp.spines["bottom"].set_visible(False)
            tmp.spines["top"].set_visible(False)
            tmp.spines["right"].set_visible(False)            
            ax.append(tmp)     
       
        for i in range(nunits):
            a = ax[i]   
            perc = np.percentile(fr[i,istart:iend], 99)
            d = yticks - perc
            ii = np.where(d < 0)[0][-1]
            ytick = yticks[ii]       
            #ytick = yticks[np.argmin((np.abs(yticks - perc)))]

            a.set_yticks([0, ytick], ['', ytick])
            a.set_ylim([-1, np.max(fr[i,:])])
            a.plot(t,fr[i,:], color=palette[i])
    

        # axes for time legend
        axes_legend = plt.axes([0.1,0.06,0.8,0.02], sharex=axes_brs)
        plt.plot([0, tlegend], [1, 1], lw=2, color='k')
        plt.ylim([-1, 1])
        axes_legend.text(tlegend/2, -2, '%d s' % tlegend, verticalalignment='bottom', horizontalalignment='center')

        plt.xlim((t[0], t[-1]))
        sleepy._despine_axes(axes_legend)

        axes_lbs = plt.axes([0.9, 0.1, 0.1, yrange])
        for i in range(nunits):
            if print_unit:
                axes_lbs.text(0.05, i+0.25, ids[i] + ' ' + regions[i])
            else:
                axes_lbs.text(0.05, i+0.25, regions[i])
        axes_lbs.set_xlim([0, 1])
        axes_lbs.set_ylim(0, nunits)
        sleepy._despine_axes(axes_lbs)
        
        plt.gcf().text(0.03,0.1+yrange/2, 'Firing rates (spikes/s)', rotation=90, verticalalignment='center')

    else:
        max_list = []
        min_list = []
        frm = fr.mean(axis=0)
        fr = np.vstack((frm, fr))

        for i in range(nunits+1):
            a = fr[i,:]
            max_list.append(a.max())
            min_list.append(a.min())
        
        max_list = np.array(max_list)
        min_list = np.array(min_list)

        ax = plt.axes([0.1, 0.1, 0.8, yrange], sharex=axes_brs)

        axes_lbs = plt.axes([0.9, 0.1, 0.1, yrange], sharey=ax)
        axes_frs = plt.axes([0.01, 0.1, 0.082, yrange], sharey=ax)

        mean_mx = np.zeros((fr.shape[0], 3))
        for j in range(1, 4):
            idx = np.where(M==j)[0]
            mean_mx[:,j-1] = fr[:,idx].mean(axis=1)

        offset = 0
        for i in range(fr.shape[0]):
            f = fr[i,:]
            r = f - min_list[i]
            r = r / (max_list[i] + np.abs(min_list[i]))            
            ax.plot(t, r+offset)
            offset += 1
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim(0, nunits+1)
        #sleepy._despine_axes(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel('Time (s)')
    
        axes_frs.pcolorfast(mean_mx)
        axes_frs.set_xticks([0.5,1.5,2.5])
        axes_frs.set_xticklabels(['R', 'W', 'N'])
        axes_frs.set_xlim([0, 3])
        #axes_frs.set_ylim([0, fr.shape[0]])
        sns.despine()
    
        offset = 0
        ids = ['mean'] + ids
        regions = [' '] + regions
        for i in range(fr.shape[0]):
            axes_lbs.text(0.05, offset, ids[i] + ' ' + regions[i], fontsize=8)
            offset += 1
        axes_lbs.set_xlim([0, 1])
        axes_lbs.set_ylim([0, nunits+1])
        sleepy._despine_axes(axes_lbs)



def plot_firingrates_dff(units, cell_info, ids, mouse, config_file, kcuts=[], 
                         tstart=0, tend=-1, ma_thr=20, ma_rem_exception=True,
                         pzscore=True, nsmooth=0,
                         pnorm_spec=False, box_filt=[], fmax=20, vm=[], dt=2.5, 
                         tlegend=60):
    dt = 2.5      
    plt.figure(figsize=(10,10))
    sleepy.set_fontarial()

    if len(ids) == 0:        
        ids = list(units.columns)    
        ids = [unit for unit in ids if '_' in unit]

    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]
    if len(M) > units.shape[0]:
        M = M[0:-1]

    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################

    
    # brain regions corresponding to the unit IDs in @ids:
    regions = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in ids]
    
    tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
    SP = tmp['SP']
    freq = tmp['freq']
    ifreq = np.where(freq <= fmax)[0]
    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
    
    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
    
        
    # cut out kcuts: ###############
    tidx = kcut_idx(M, units, kcuts, dt=dt)
    M = M[tidx]
    units = units.iloc[tidx,:]
    SP = SP[:,tidx]
    ################################
    
    # set istart and iend
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    istart = int(np.round(tstart/dt))         
    
    M = M[istart:iend]
    # NOTE units.loc[i:j,:] INCLUDES the j-th element, 
    # that's different from how np.arrays behave

    units = units.iloc[istart:iend,:]
    SP = SP[:,istart:iend]
    
    nunits = len(ids)
    ntime = units.shape[0]
    fr = np.array(units[ids]).T

    if nsmooth > 0:
        for i in range(fr.shape[0]):
            fr[i,:] = sleepy.smooth_data(fr[i,:], nsmooth)
    if pzscore:
        for i in range(fr.shape[0]):
            fr[i,:] = (fr[i,:] - fr[i,:].mean()) / fr[i,:].std()
    
    nhypno = np.min((len(M), ntime))
    M = M[0:nhypno]
    
    
    t = np.arange(0, len(M))*dt        
    # Axes for hypnogram
    xrange = 0.7
    axes_brs = plt.axes([0.1, 0.95, xrange, 0.03])
    
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # plot spectrogram
    # calculate median for choosing right saturation for heatmap
    med = np.median(SP.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.0]    
    # axes for specrogram 
    axes_spec = plt.axes([0.1, 0.85, xrange, 0.08], sharex=axes_brs)    
    # axes for colorbar
    axes_cbar = plt.axes([0.8, 0.85, 0.05, 0.08])

    im = axes_spec.pcolorfast(t, freq[ifreq], SP[ifreq,:], vmin=vm[0], vmax=vm[1], cmap='jet')
    sleepy.box_off(axes_spec)
    axes_spec.axes.get_xaxis().set_visible(False)
    axes_spec.spines["bottom"].set_visible(False)
    axes_spec.set_ylabel('Freq. (Hz)')
    
    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, location='right')
    if pnorm_spec:
        #cb.set_label('Norm. power')
        cb.set_label('')
    else:
        cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    axes_cbar.set_alpha(0.0)
    sleepy._despine_axes(axes_cbar)
    
    t = np.arange(0, len(M))*dt
    yrange = 0.73
    # axes for firing rates
    ax_fr = plt.axes([0.1, 0.1, xrange, yrange], sharex=axes_brs)
    
    
    fmax = fr.max()
    for i,ID in enumerate(ids):

        ax_fr.plot(t, fr[i]+i*fmax, color='k')
        #plt.text(10, i*fmax+fmax/4, str(i), fontsize=14, color=cmap[i,:],bbox=dict(facecolor='w', alpha=0.))

    sleepy._despine_axes(ax_fr)
    plt.ylim([-fmax, fmax*len(ids)])


    ax_tlegend = plt.axes([0.1, 0.05, xrange, 0.05], sharex=axes_brs)
    ax_tlegend.plot([0, tlegend], [0,0], color='k')
    ax_tlegend.text(tlegend/2, -2, '%d s' % tlegend, verticalalignment='bottom', horizontalalignment='center')
    sleepy._despine_axes(ax_tlegend)
    plt.ylim([-1, 1])

    plt.xlim((t[0], t[-1]))

    
    ax_ulegend = plt.axes([0.05, 0.1, 0.05, yrange])
    plt.ylim([-fmax, fmax*len(ids)])
    plt.xlim([-1, 1])
    plt.plot([0, 0], [0, 5], color='black')
    sleepy._despine_axes(ax_ulegend)
    


def plot_firingrates_map(units, cell_info, ids, mouse, config_file, kcuts=[], 
                         tstart=0, tend=-1, ma_thr=20, ma_rem_exception=True,
                         pzscore=True, nsmooth=0, cmap_fr='', yrange_fr=-1,
                         pnorm_spec=False, box_filt=[], fmax=20, r_mu=[10,100],
                         vm=[], cb_ticks=[], dt=2.5, tlegend=300,
                         vm_fr=[], pregion=False):
    """
    See also &plot_firingrates() and &plot_firingrates_map2()
    Plot firing rates of selected units as heatmap.
    
    Note typically units are arranged in a way that a low ID number corresponds
    to a 'deep' unit.
    
    Parameters
    ----------
    units : pd.DataFrame
        Unit DataFrame; wich each column corresponding to one unit.
    cell_info : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    mouse : str
        mouse name.
    config_file : str
        file name of the mouse configuration file.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    tstart: float,
        Start time of the the shown interval (in seconds)
    tend: float
        End point of the shown interval; if -1, how till the end.
    ma_thr: float
        Microarousal threshold
    ma_rem_expection:
        If a wake period follows REM, it stays wake, even if it
        is shorter than the microarousal threshold ($ma_thr)
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    pnorm_spec : bool, optional
        If True, normalize EEG spectrogram. 
    box_filt : tuple or two element list, optional
        Filter EEG spectrogram using box filder . If [], no filtering is applied.
    fmax : TYPE, optional
        DESCRIPTION. The default is 20.
    vm : tuple
        lower and upper range color color range for EEG spectrogram colormap.
        If vm==[], the colorange is set automatically
    cb_ticks: list
        Ticks on the colorbar. If you set cb_ticks = vm, then only the upper
        and lower range of the colorbar are set as ticks. 
        If empty, set ticks automatically.
    pregion : bool
        If True, color code different brain regions using color bar on the right

    Returns
    -------
    None.

    """
    dt = 2.5          
    plt.figure(figsize=(10,10))
    sleepy.set_fontarial()

    if len(ids) == 0:        
        ids = list(units.columns)    
        ids = [unit for unit in ids if '_' in unit]
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]
    if len(M) > units.shape[0]:
        M = M[0:-1]
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3

    # brain regions corresponding to the unit IDs in @ids:
    if pregion:
        regions = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in ids]
    
    tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
    SP = tmp['SP']
    freq = tmp['freq']
    ifreq = np.where(freq <= fmax)[0]
    
    # load EMG
    tmp = so.loadmat(os.path.join(ppath, file, 'msp_%s.mat' % file), squeeze_me=True)
    SPEMG = tmp['mSP']
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0 # back to muV
    # END[load EMG]

    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
    
    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        
    # cut out kcuts: ###############
    tidx = kcut_idx(M, units, kcuts, dt=dt)
    M = M[tidx]
    units = units.iloc[tidx,:]
    SP = SP[:,tidx]
    ################################
    
    # set istart and iend
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    istart = int(np.round(tstart/dt))         
    
    M = M[istart:iend]
    # NOTE units.loc[i:j,:] INCLUDES the j-th element, 
    # that's different from how np.arrays behave

    units = units.iloc[istart:iend,:]
    SP = SP[:,istart:iend]
    
    nunits = len(ids)
    ntime = units.shape[0]
    #units are organized in columns like: 0_good, 1_good, ... n_good    
    fr = np.array(units[ids]).T
    # fr is organized in rows:
    # 0_good
    # 1_good
    # ...
    # n_good

    if nsmooth > 0:
        for i in range(fr.shape[0]):
            fr[i,:] = sleepy.smooth_data(fr[i,:], nsmooth)
    if pzscore:
        for i in range(fr.shape[0]):
            fr[i,:] = (fr[i,:] - fr[i,:].mean()) / fr[i,:].std()
    
    
    nhypno = np.min((len(M), ntime))
    M = M[0:nhypno]        
    t = np.arange(0, len(M))*dt        
    # Axes for hypnogram
    xrange = 0.7
    axes_brs = plt.axes([0.1, 0.95, xrange, 0.03])
    
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # plot spectrogram
    # calculate median for choosing right saturation for heatmap
    med = np.median(SP.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.0]    
    # axes for spectrogram 
    axes_spec = plt.axes([0.1, 0.85, xrange, 0.08], sharex=axes_brs)    
    # axes for colorbar
    axes_cbar = plt.axes([0.83, 0.85, 0.015, 0.08])

    im = axes_spec.pcolormesh(t, freq[ifreq], SP[ifreq,:], vmin=vm[0], vmax=vm[1], cmap='jet')
    sleepy.box_off(axes_spec)
    axes_spec.axes.get_xaxis().set_visible(False)
    axes_spec.spines["bottom"].set_visible(False)
    axes_spec.set_ylabel('Freq.\n(Hz)')
    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, cax=axes_cbar, pad=0.0, aspect=10.0, location='right')
    if pnorm_spec:
        cb.set_label('Norm. power')
    else:
        cb.set_label('Power ($\mathrm{\mu}$V$^2$/Hz)')
    if len(cb_ticks) > 0:
        cb.set_ticks(cb_ticks)        
    axes_cbar.set_alpha(0.0)
    #sleepy._despine_axes(axes_cbar)
        
    # Show EMG amplitude
    # Axes for EMG
    axes_emg = plt.axes([0.1, 0.79, xrange, 0.05], sharex=axes_brs)
    axes_emg.plot(t, p_mu[istart:iend], color='black')    
    sleepy.box_off(axes_emg)
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    axes_emg.set_xticks([])
    axes_emg.set_ylabel('EMG\n($\mathrm{\mu}$V)')
    # END[Show EMG amplitude]

    # axes for firing rates
    yrange_fr_total = 0.73

    if yrange_fr < 0:    
        yrange = 0.68+0.05
    else:
        yrange = yrange_fr
    ax_fr = plt.axes([0.1, 0.05+yrange_fr_total-yrange, xrange, yrange], sharex=axes_brs)
    ax_cbfr = plt.axes([0.83, 0.1+yrange*0.08+yrange_fr_total-yrange, 0.015, yrange*0.5])

    if len(vm_fr) == 0:
        vm_fr[0] = np.percentile(fr, 1)
        vm_fr[1] = np.percentile(fr, 99)

    y = np.arange(0, nunits)
    t = np.arange(0, len(M))*dt
    if cmap_fr == '':
        cmap_fr = 'magma'
    im = ax_fr.pcolorfast(t, y, fr, vmin=vm_fr[0], vmax=vm_fr[1], cmap=cmap_fr)
    sleepy.box_off(ax_fr)
    #ax_fr.set_xlabel('Time (s)')
    ax_fr.set_ylabel('Unit no.')    
    cb = plt.colorbar(im, cax=ax_cbfr, pad=0.0, aspect=10.0, location='right')
    cb.set_label('FR (z-scored)')
    #sleepy._despine_axes(ax_cbfr)

    # Axes for time legend
    axes_time = plt.axes([0.1, 0.02+yrange_fr_total-yrange, xrange, 0.02], sharex=axes_brs)
    axes_time.set_xlim([t[0], t[-1]])
    plt.plot([0,tlegend], [0,0], 'k')
    plt.text(0, -2, '%s s' % str(tlegend))
    plt.ylim([-1, 1])
    sleepy._despine_axes(axes_time)
    
    reg_code = []
    if pregion:
        # build region map
        regions = list(cell_info[cell_info.ID.isin(ids)].brain_region.unique())
        #regions.sort()
        clrs = sns.color_palette('Set2', len(regions))
        reg2int = {r:i for r,i in zip(regions, range(len(regions)))}
        
        cmap = plt.matplotlib.colors.ListedColormap(clrs)
            
        reg_code = np.zeros((nunits,))
        for i,ID in enumerate(ids):
            region = cell_info[cell_info.ID == ID]['brain_region'].item()        
            reg_code[i] = reg2int[region]
        
        # Axes for brain region colorcode 
        ax_rg = plt.axes([0.82, 0.05, 0.02, yrange])
        #ax_fr = plt.axes([0.1, 0.05, xrange, yrange], sharex=axes_brs)

        A = np.zeros([nunits,1])
        A[:,0] = reg_code
        ax_rg.pcolorfast(A, cmap=cmap)
        sleepy._despine_axes(ax_rg)
            
        # Add axes for brain region legends
        ax_lb = plt.axes([0.85, 0.1, 0.1, yrange*0.25])
        for k in reg2int:
            i = reg2int[k]
            ax_lb.plot([0.5, 1], [i,i], color=clrs[i], lw=2)
            ax_lb.text(1.5, i, k, va='center')
            
        ax_lb.set_ylim([-1, len(regions)])
        ax_lb.set_xlim([0, 2])        
        sleepy._despine_axes(ax_lb)
                        
    return reg_code


        
def plot_firingrates_map2(cell_info, ids, mouse, config_file, kcuts=[], 
                         tstart=0, tend=-1, ma_thr=10, ma_rem_exception=True,
                         pzscore=True, nsmooth=0,
                         pnorm_spec=False, box_filt=[], fmax=20, vm=[], dt=2.5, 
                         vm_fr=[]):
    """
    Plot Neuropixels firing rates at heatmap using 100ms resolution

    Parameters
    ----------
    cell_info : pd.DataFrame
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    config_file : TYPE
        DESCRIPTION.
    kcuts : TYPE, optional
        DESCRIPTION. The default is [].
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is False.
    box_filt : TYPE, optional
        DESCRIPTION. The default is [].
    fmax : float, optional
        Maximal frequency in EEG spectrogram. The default is 20.
    vm : tuple, optional
        Min and Max value of color saturation of EEG spectrogram. If empty,
        determine saturation automatically.
    
        
    Returns
    -------
    None.

    """
    dt = 2.5  
    NDOWN = 100
    dt_tr = 0.001 * NDOWN
    NUP = int(dt / (0.001 * NDOWN))

    
    plt.figure(figsize=(10,10))
    sleepy.set_fontarial()

    if len(ids) == 0:        
        ids = list(cell_info.ID)    
        ids = [unit for unit in ids if '_' in unit]

    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]

    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################    
    tr_path = load_config(config_file)[mouse]['TR_PATH']
    try:
        units = np.load(os.path.join(tr_path,'lfp_1k_train.npz')) 
    except:
        units = np.load(os.path.join(tr_path,'1k_train.npz')) 
        

    unitIDs = [unit for unit in list(units.keys()) if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']

    # Selects IDs of units
    if len(ids) == 0:
        ids = unitIDs

    
    # brain regions corresponding to the unit IDs in @ids:
    regions = [cell_info[cell_info.ID == i].brain_region.iloc[0] for i in ids]
    
    tmp = so.loadmat(os.path.join(ppath, file, 'sp_%s.mat'%file), squeeze_me=True)
    SP = tmp['SP']
    freq = tmp['freq']
    ifreq = np.where(freq <= fmax)[0]
    
    if len(box_filt) > 0:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
    
    if pnorm_spec:
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            
    
    # KCUT ####################################################################
    # get the indices (in brainstate time) that we're going to completely discard:
    nhypno = np.min((SP.shape[1], M.shape[0]))
    M = M[0:nhypno]
    
    tidx = np.arange(0, nhypno)
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)
    ###########################################################################    
    
    # set istart and iend
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    istart = int(np.round(tstart/dt))         
    
    M = M[istart:iend]
    SP = SP[:,istart:iend]
    
    tbs = np.arange(0, len(M)) * dt
    
    # Process firing rates ####################################################   
    nsample = int(len(units[unitIDs[0]])/NDOWN)
    R = np.zeros((len(units), nsample))
    fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % NDOWN)
    if not os.path.isfile(fr_file):
        print('downsampling spike trains...')
        processed_ids = []
        for i, unit in enumerate(unitIDs):
            tmp = sleepy.downsample_vec(np.array(units[unit]), NDOWN)            
            R[i,:] = tmp
            processed_ids.append(unit)
        so.savemat(fr_file, {'R':R, 'ndown':NDOWN, 'ID':np.array(processed_ids)})
        print('saving spike trains...')
    else:
        tmp = so.loadmat(fr_file, squeeze_me=True)
        R = tmp['R']
        processed_ids = list(tmp['ID'])
    ###########################################################################
    unit2idx = {ID.strip():i for ID,i in zip(processed_ids, list(range(len(processed_ids))))}

    istart_tr = istart * NUP
    iend_tr   = iend   * NUP 
    tidx_tr = np.arange(tidx[0] * NUP, tidx[-1] * NUP)
    nunits = len(ids)
    fr = np.zeros((len(ids), len(tidx_tr)))
    print('smoothing and z-scoring spike trains...')
    for i,ID in enumerate(ids):
        j = unit2idx[ID]
        r = R[j,tidx_tr]
        if nsmooth>0:
            r = sleepy.smooth_data(r, nsmooth)
        if pzscore:
            r = (r - r.mean()) / r.std()
        fr[i,:] = r
            
    
    fr = fr[:,istart_tr:iend_tr]
    t_tr = np.arange(0, fr.shape[1])*dt_tr
    
    # Axes for hypnogram
    xrange = 0.7
    axes_brs = plt.axes([0.1, 0.95, xrange, 0.03])
    
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(tbs, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # plot spectrogram
    # calculate median for choosing right saturation for heatmap
    med = np.median(SP.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.0]    
    # axes for specrogram 
    axes_spec = plt.axes([0.1, 0.85, xrange, 0.08], sharex=axes_brs)    
    # axes for colorbar
    axes_cbar = plt.axes([0.8, 0.85, 0.05, 0.08])

    im = axes_spec.pcolorfast(tbs, freq[ifreq], SP[ifreq,:], vmin=vm[0], vmax=vm[1], cmap='jet')
    sleepy.box_off(axes_spec)
    axes_spec.axes.get_xaxis().set_visible(False)
    axes_spec.spines["bottom"].set_visible(False)
    axes_spec.set_ylabel('Freq. (Hz)')
    
    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, location='right')
    if pnorm_spec:
        #cb.set_label('Norm. power')
        cb.set_label('')
    else:
        cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    axes_cbar.set_alpha(0.0)
    sleepy._despine_axes(axes_cbar)
    
    # axes for firing rates    
    yrange = 0.73    
    ax_fr = plt.axes([0.1, 0.1, xrange, yrange], sharex=axes_brs)
    ax_cbfr = plt.axes([0.85, 0.1+yrange*0.5, 0.05, yrange*0.5])

    if len(vm_fr) == 0:
        vm_fr[0] = np.percentile(fr, 1)
        vm_fr[1] = np.percentile(fr, 99)

    y = np.arange(0, nunits)
    im = ax_fr.pcolorfast(t_tr, y, fr, vmin=vm_fr[0], vmax=vm_fr[1], cmap='magma')
    sleepy.box_off(ax_fr)
    ax_fr.set_xlabel('Time (s)')
    ax_fr.set_ylabel('Unit no.')
    
    cb = plt.colorbar(im, ax=ax_cbfr, pad=0.0, aspect=10.0, location='right')
    sleepy._despine_axes(ax_cbfr)

    
    # build region map
    regions = list(cell_info[cell_info.ID.isin(ids)].brain_region.unique())
    #regions.sort()
    clrs = sns.color_palette('Set2', len(regions))

    reg2int = {r:i for r,i in zip(regions, range(len(regions)))}
    
    cmap = plt.matplotlib.colors.ListedColormap(clrs)


    reg_code = np.zeros((nunits,))
    for i,ID in enumerate(ids):
        region = cell_info[cell_info.ID == ID]['brain_region'].item()        
        reg_code[i] = reg2int[region]

    # Axes for brain region colorcode 
    ax_rg = plt.axes([0.82, 0.1, 0.02, yrange])
    A = np.zeros([nunits,1])
    A[:,0] = reg_code
    ax_rg.pcolorfast(A, cmap=cmap)
    sleepy._despine_axes(ax_rg)
    
    # Add axes for brain region legends
    ax_lb = plt.axes([0.85, 0.1, 0.1, yrange*0.25])
    for k in reg2int:
        i = reg2int[k]
        ax_lb.plot([0.5, 1], [i,i], color=clrs[i], lw=2)
        ax_lb.text(1.5, i, k, va='center')
        
    ax_lb.set_ylim([-1, len(regions)])
    ax_lb.set_xlim([0, 2])
    
    sleepy._despine_axes(ax_lb)
        
    print(reg2int)
    return reg_code



def downsample_fr(mouse, config_file, ndown, ids=[]):
    """
    Downsample 1ms firing rates by factor $ndown, and
    return as np.array, with each row corresponding to a unit

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    tr_path : TYPE
        DESCRIPTION.
    ndown : TYPE
        DESCRIPTION.
    ids : TYPE
        List of unit IDs. If empty, process all units in $units.


    Returns
    -------
    R : TYPE
        DESCRIPTION.
    processsed_ids: list
        list of unit IDs in np.array R (row by row)

    """
    
    tr_path = load_config(config_file)[mouse]['TR_PATH']
    try:
        units = np.load(os.path.join(tr_path,'lfp_1k_train.npz')) 
    except:
        units = np.load(os.path.join(tr_path,'1k_train.npz')) 
        
    # Selects IDs of units
    if len(ids) == 0:
        ids = [unit for unit in list(units.keys()) if '_' in unit]
        ids = [unit for unit in ids if re.split('_', unit)[1] == 'good']
        unitIDs = ids
    else:
        unitIDs = ids
                
    nsample = int(len(units[unitIDs[0]])/ndown)
    R = np.zeros((len(units), nsample))
    fr_file = os.path.join(tr_path, 'fr_fine_ndown%d.mat' % ndown)
    if not os.path.isfile(fr_file):
        print('downsampling spike trains...')
        processed_ids = []
        for i, unit in enumerate(unitIDs):
            tmp = sleepy.downsample_vec(np.array(units[unit]), ndown)            
            R[i,:] = tmp
            processed_ids.append(unit)
        so.savemat(fr_file, {'R':R, 'ndown':ndown, 'ID':np.array(processed_ids)})
        print('saving spike trains...')
        print('done.')
    else:
        tmp = so.loadmat(fr_file, squeeze_me=True)
        R = tmp['R']
        processed_ids = list(tmp['ID'])
        processed_ids = [p.strip() for p in processed_ids]

    return R, processed_ids
        


def plot_avg_firingrates(units, cell_info, ids, M, pzscore=True, 
                         pind_axes=True, dt=2.5, nsmooth=0, ma_thr=10, 
                         ma_rem_exception=False,
                         ma_mode=False, ax='', point_color='black', pplot=True):
    """
    
    Calculate and plot for each unit (ID) in DataFrame units the average
    firing rate for each state.

    Parameters
    ----------
    units : pd.DataFrame
        Each column is a unit; each column name is the ID of a unit.
    cell_info : TYPE
        DESCRIPTION.
    ids : list of str
        IDs (columns names in $units) of the units to be analyzed.
    M : np.array
        hypnogram.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    pind_axes : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    ma_thr : float, optional
        Microarousal threshold. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is False.
    ma_mode : bool
        If True, also plot average activity during MAs.
    ax : TYPE, optional
        DESCRIPTION. The default is ''.
    point_color : TYPE, optional
        DESCRIPTION. The default is 'black'.


    Returns
    -------
    df : pd.DataFrame
        with columns: fr, state, ID.
        i.e. each row contains for each neuron ID and state the
        average firing rate

    """
    nhypno = np.min((len(M), units.shape[0]))
    M = M[0:nhypno]
    
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    if not ma_mode:
                        M[s] = 3
                    else:
                        M[s] = 4

    nunits = len(ids)
    R = np.zeros((nunits, nhypno))
    for i,u in enumerate(ids):
        tmp = units[u]        
        tmp = sleepy.smooth_data(tmp, nsmooth)
        if pzscore:
            tmp = (tmp - tmp.mean()) / tmp.std()        
        R[i,:] = tmp

    states = [1,2,3]
    if ma_mode:
        states = [1,2,3,4]

    state_map = {1:'REM', 2:'Wake', 3:'NREM', 4:'MA'}
    data = []
    for s in states:
        idx = np.where(M == s)[0]
        tmp = R[:,idx].mean(axis=1)

        data += zip(ids, tmp, [state_map[s]]*nunits)

    df = pd.DataFrame(data=data, columns=['ID', 'fr', 'state'])
    
    if pplot:
        if ax == '':
            plt.figure()
            ax = plt.axes([0.2, 0.15, 0.45, 0.75])
        custom_palette = ['lightgray']
        g = sns.lineplot(data=df, x='state', y='fr', hue='ID', palette=custom_palette, ax=ax)
        g.legend_.remove()
        
        sns.pointplot(data=df, y='fr', x='state', color=point_color, ax=ax)
        sns.despine()
        plt.xlabel('')
        plt.ylabel('FR (spikes/s)')
    
    return df
    

    
def plot_hypnogram(M, dt=2.5, axes_brs=''):

    t = np.arange(0, M.shape[0])*dt
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)



def fr_infraslow(units, cell_info, mouse, ids=[], pzscore=True, dt=2.5, nsmooth=0, ma_thr=10, 
                 ma_rem_exception=False, ma_mode=True, kcuts=[], state=3, peeg2=False,
                 band=[10,15], min_dur=120, win=100, 
                 pnorm=False, spec_norm=True, spec_filt=False, box=[1,4], 
                 ma_thr_exception=True,
                 config_file='mouse_config.txt'):
    """
    Calculate FFT of firing rates for consolidated bouts of NREM sleep and compare with
    FFT of sigma power (= infraslow rhythm)

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    cell_info : TYPE
        DESCRIPTION.
    mouse : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is False.
    ma_mode : TYPE, optional
        DESCRIPTION. The default is True.
    kcut : TYPE, optional
        DESCRIPTION. The default is [].
    state : TYPE, optional
        DESCRIPTION. The default is 3.
    peeg2 : TYPE, optional
        DESCRIPTION. The default is False.
    band : TYPE, optional
        DESCRIPTION. The default is [10,15].
    min_dur : TYPE, optional
        DESCRIPTION. The default is 120.
    win : TYPE, optional
        DESCRIPTION. The default is 100.
    pnorm : TYPE, optional
        DESCRIPTION. The default is False.
    spec_norm : TYPE, optional
        DESCRIPTION. The default is True.
    spec_filt : TYPE, optional
        DESCRIPTION. The default is False.
    box : TYPE, optional
        DESCRIPTION. The default is [1,4].

    Returns
    -------
    df : pd.DataFrame
        with columns ['mouse', 'freq', 'pow', 'ID'].

    """        
    sdt = 2.5
    min_dur = np.max([win*2.5, min_dur])
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    M = sleepy.load_stateidx(ppath, file)[0]
        
    # flatten out MAs #########################################################
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*sdt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3
    ###########################################################################
    
    nhypno = np.min((units.shape[0], M.shape[0]))
    tidx = np.arange(0, nhypno)    

    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)        
    ###########################################################################    

    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']
    nsample = len(tidx)   # number of time points
    if ids == []:
        ids = unitIDs
    nvar    = len(ids)     # number of units    
    R = np.zeros((nvar, nsample))    
        
    for i,unit in enumerate(ids):        
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]
    
    # flatten out MAs
    if ma_thr>0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round(len(s)*dt) <= ma_thr:
                if ma_rem_exception:
                    if (s[0]>1) and (M[s[0] - 1] != 1):
                        M[s] = 3
                else:
                    M[s] = 3

    seq = sleepy.get_sequences(np.where(M==state)[0], np.round(ma_thr/dt)+1)
    seq = [list(range(s[0], s[-1]+1)) for s in seq]
    
    # load frequency band
    P = so.loadmat(os.path.join(ppath, file,  'sp_' + file + '.mat'))
    if not peeg2:
        SP = np.squeeze(P['SP'])[:,tidx]
    else:
        SP = np.squeeze(P['SP2'])[:, tidx]
    freq = np.squeeze(P['freq'])
    ifreq = np.where((freq>=band[0]) & (freq<=band[1]))[0]
    if spec_filt:
        filt = np.ones(box)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if spec_norm:
        sp_mean = SP[:, :].mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        pow_band = SP[ifreq,:].mean(axis=0)
    else:
        pow_band = SP[ifreq, :].sum(axis=0) * (freq[1]-freq[0])
        nidx = np.where(M==3)[0]
        pow_band = pow_band / pow_band[nidx].mean()


    seq = [s for s in seq if len(s)*dt >= min_dur]   
    Spec = []
    for s in seq:
        y,f = sleepy.power_spectrum(pow_band[s], win, dt)
        Spec.append(y)
        
    # Transform %Spec to ndarray
    SpecVec = np.array(Spec).mean(axis=0)
    data = []

    if pnorm==True:
        SpecVec = SpecVec / SpecVec.mean()
    data += zip([mouse]*len(f), f, SpecVec, ['spec']*len(f))    
    
    Units = {unit:[] for unit in ids}
    for i,unit in enumerate(ids):
        r = R[i,:]
        
        for s in seq:            
            y,f = sleepy.power_spectrum(r[s], win, dt)
            Units[unit].append(y)
    
    UnitMx = np.zeros((len(Units), len(f)))
    for i,unit in enumerate(Units):
        UnitMx[i,:]  = np.array(Units[unit]).mean(axis=0)
        if pnorm==True:
            UnitMx[i,:]  = UnitMx[i,:] / UnitMx[i,:].mean()            
        data += zip([mouse]*len(f), f, UnitMx[i,:], [unit]*len(f))
        
    df = pd.DataFrame(data=data, columns=['mouse', 'freq', 'pow', 'ID'])

    # todo: add pearson correlation
    idx = np.where(M==3)[0]
    data = []
    for i,unit in enumerate(ids):
        r,p = scipy.stats.pearsonr(units[unit][idx], pow_band[idx])
        data += [[unit, r, p]]

    df_stats = pd.DataFrame(data=data, columns=['ID', 'r2', 'p'])
        
    return df, df_stats



def pc_infraslow(PC, M, mouse, kcuts=(), dt=2.5, nsmooth=0, ma_thr=20, 
                 ma_rem_exception=False, state=3, peeg2=False,
                 band=[10,15], min_dur=120, win=120, config_file='mouse_config.txt',
                 pnorm=False, spec_norm=True, spec_filt=False, box=[1,4], pplot=True):
    """
    
    NOTE on kcuts: We assume that for PC the kcut intervals are removed.
    If kcut intervals are provided then kcut will be applied to both 
    the sigma power and brain state (M). 


    Parameters
    ----------
    PC : np.array
        Each row corresponds to a PC.
    mouse : str
        mouse name.
    kcuts : list of tuples or lists with two elements.
        Discard the time interval ranging from kcuts[i][0] to kcuts[i][1] seconds.
    dt : float, optional
        Time binning of PCs and hypnogram. The default is 2.5.
    nsmooth : float, optional
        Smooth firing rates with Gaussain kernel with standard deviation $nsmooth. The default is 0.
    ma_thr : float, optional
        Wake sequences <= $ma_thr s are interpreted as NREM (3). The default is 10.
    ma_rem_exception : bool, optional
        If True, then the MA rule does not apply for wake episodes directly following REM. 
        The default is False.
    state : int, optional
        Calculate PSD only for stae $state. 1 - NREM, 2 - Wake, 3 - NREM
    peeg2 : bool, optional
        If True, calculate sigma power PSD using EEG2. The default is False.
    band : tuple or list, optional
        Define frequency range for sigma power (or other band). The default is [10,15].
    min_dur : float, optional
        Minimum duration of NREM bouts used for infraslow calculation. The default is 120.
    win : float, optional
        Time window for PSD calculation. The default is 120.
    pnorm : bool, optional
        If True, normalize PSDs of PCs and sigma power (by dividing by mean power).
    spec_norm : bool, optional
        If True, normalize spectrogram. The default is True.
    spec_filt : bool, optional
        If True, run box filter over EEG spectrogram. The default is False.
    box : bool, optional
        Specifies dimension of box filter, if $spec_filt == True. The default is [1,4].
    pplot : bool optional
        If True, plot figure. The default is True.

    Returns
    -------
    df : pd.DataFrame 
         with columns=['mouse', 'freq', 'pow', 'typ']
        'freq' is the frequency vlaue
        'pow' the power for the given frequency
        'mouse' is the given mouse name
        'typ' specifies the PC ('PC1', 'PC2', ...) or sigma power 'Spec' 


    """
    
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, file = os.path.split(path)
    min_dur = np.max([win*2.5, min_dur])

    # load frequency band
    P = so.loadmat(os.path.join(ppath, file,  'sp_' + file + '.mat'))
    if not peeg2:
        SP = np.squeeze(P['SP'])
    else:
        SP = np.squeeze(P['SP2'])
    freq = np.squeeze(P['freq'])
    ifreq = np.where((freq>=band[0]) & (freq<=band[1]))[0]
    if spec_filt:
        filt = np.ones(box)
        filt = np.divide(filt, filt.sum())
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

    if spec_norm:
        sp_mean = SP[:, :].mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        pow_band = SP[ifreq,:].mean(axis=0)
    else:
        pow_band = SP[ifreq, :].sum(axis=0) * (freq[1]-freq[0])
        nidx = np.where(M==3)[0]
        pow_band = pow_band / pow_band[nidx].mean()

    # cut out kcuts: ###############
    tidx = kcut_idx(M, PC, kcuts)
    M = M[tidx]
    # PC is already kcut! 
    #PC = PC[:,tidx]
    SP = SP[:,tidx]
    pow_band = pow_band[tidx]            
    if not(len(M) == len(pow_band) == PC.shape[1]):
        print('Something went wrong with KCUT')
        print('returning')
        return
    ################################

    seq = sleepy.get_sequences(np.where(M==state)[0], int(np.round(ma_thr/dt)+1))
    seq = [list(range(s[0], s[-1]+1)) for s in seq]


    seq = [s for s in seq if len(s)*dt >= min_dur]   
    Spec = []
    ndim = PC.shape[0]
    Act = []
    for s in seq:
        y,f = sleepy.power_spectrum(pow_band[s], win, dt)
        Spec.append(y)
    
    Act = {i:[] for i in range(ndim)}
    for s in seq:
        for i in range(ndim):        
            y,f = sleepy.power_spectrum(PC[i,s], win, dt)
            Act[i].append(y)
        
        
    # Transform @Spec to np.array
    Spec = np.array(Spec).mean(axis=0)
    
    # Transofrm @Act to np.array
    for i in range(ndim):
        Act[i] = np.array(Act[i]).mean(axis=0)
    
    # normalize sigma power spectrum
    data = []
    if pnorm:
        Spec = Spec / Spec.mean()
    data += zip([mouse]*len(f), f, Spec, ['Spec']*len(f))

    # normalize PC spectrum
    if pnorm:
        for i in range(ndim):
            Act[i] = Act[i] / Act[i].mean()
            
    for i in range(ndim):
        data += zip([mouse]*len(f), f, Act[i], ['PC%d' % (i+1)]*len(f))
        
    df = pd.DataFrame(data=data, columns=['mouse', 'freq', 'pow', 'typ'])
        
    if pplot:
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        pc_categories = ['PC%d' % (i+1) for i in range(ndim)]
        pc_palette = sns.color_palette("husl", ndim)
        for i,pc_category in enumerate(pc_categories):
            sns.lineplot(data=df[df['typ'] == pc_category], x='freq', y='pow', ax=ax1, label=pc_category, color=pc_palette[i], lw=2)
        
        #sns.lineplot(data=df[df['typ'].isin(pc_categories)], x='freq', y='pow', hue='typ', ax=ax1)
        g = sns.lineplot(data=df[df['typ'] == 'Spec'], x='freq', y='pow', ax=ax2, color='gray', label='Spec', lw=2, linestyle='--')
        
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
        ax1.set_ylabel('Norm. power')
        ax2.set_ylabel('Norm. power', color='gray')
        
        ax2.tick_params(axis='y', colors='gray')
        
        g.legend().remove()
        
        ax1.set_xlabel('Freq. (Hz)')
        ax2.set_xlabel('Freq. (Hz)')
        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2)
    
    return df
    


def bandpass_corr_state(mouse, band, ids=[], mouse_config='mouse_config.txt', 
                        sr=0, sdt=2.5, fft_win=2.5, perc_overlap=0.8, pnorm_spec=True,
                        win=120, state=3, ma_thr=20, 
                        cc_max_win=0, sign='no',
                        rm_flag=False):
    """
    Correlate band in EEG spectrogram with unit activity.
    The function recalculates the EEG spectrogram using overlapping sliding windows to 
    attain higher temporal resolution. The overlap is controlled by $perc_overlap.

    The firing rates are downsampled using the same windowing as for the spectrogram calculation 
    (see &downsample_overlap). The function uses the 1kHz spike train to calculate
    the downsampled firing rate vector. The location of the 1kHz spike trains is
    specified in the file $mouse_config (using the flag 'TR_PATH:').
    
    Correlates spike train with EEG power band


    Parameters
    ----------
    mouse : string
        identifier of mouse.
    band : 2 element list or tuple
        lower and upper range for the EEG frequency band that is correlted with
        the firing rates.
    ids : list
        list of unit IDs to be correlated.
    mouse_config : TYPE, optional
        DESCRIPTION. The default is 'mouse_config.txt'.
    sr : TYPE, optional
        DESCRIPTION. The default is 0.
    sdt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    fft_win : TYPE, optional
        DESCRIPTION. The default is 2.5.
    perc_overlap : float, optional
        Value between 0 and 1; controls overlap between of two successive
        Hanning windows used for EEG spectrogram calculation. 
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is True.
    win : TYPE, optional
        DESCRIPTION. The default is 120.
    state : TYPE, optional
        DESCRIPTION. The default is 3.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 20.
    rm_flag : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    a = load_config(mouse_config)[mouse]
    ppath, name = os.path.split(a['SL_PATH'])
    tr_path = a['TR_PATH']
    #tr_units = np.load(os.path.join(tr_path,'1k_train.npz')) 
    if os.path.isfile(os.path.join(tr_path,'1k_train.npz')):
        tr_units = np.load(os.path.join(tr_path,'1k_train.npz')) 
    else:
        tr_units = np.load(os.path.join(tr_path,'lfp_1k_train.npz'))     

    
    # calculate and load powerspectrum
    if sr==0:
        sr = sleepy.get_snr(ppath, name)

    nwin = int(np.round(sr * fft_win))
    if nwin % 2 == 1:
        nwin += 1
    noverlap = int(nwin*perc_overlap)
    
    sp_file = os.path.join(ppath, name, 'sp_fine_%s.mat' % name)
    if rm_flag and os.path.isfile(sp_file):
        os.remove(sp_file)
        
    if not os.path.isfile(sp_file):    
        EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
        fband, tband, spec_band = scipy.signal.spectrogram(EEG, nperseg=nwin, noverlap=noverlap, fs=sr)
        so.savemat(sp_file, {'fband':fband, 'tband':tband, 'spec_band':spec_band})

    else:
        tmp = so.loadmat(sp_file, squeeze_me=True)
        tband = tmp['tband']
        fband = tmp['fband']
        spec_band = tmp['spec_band']
        
    if pnorm_spec:
        sp_mean = spec_band.mean(axis=1)
        spec_band = np.divide(spec_band, np.tile(sp_mean, (spec_band.shape[1], 1)).T)
    
    ifreq = np.where((fband >= band[0]) & (fband <= band[1]))[0]
    dt = tband[1] - tband[0]
    iwin = int(win / dt)

    if not pnorm_spec:
        pow_band = spec_band[ifreq, :].sum(axis=0) * (fband[1]-fband[0])
    else:
        pow_band = spec_band[ifreq, :].mean(axis=0)
    pow_band -= pow_band.mean()
    
    M = sleepy.load_stateidx(ppath, name)[0]
    seq = sleepy.get_sequences(np.where(M==state)[0], np.round(ma_thr/sdt)+1)
    seq = [list(range(s[0], s[-1]+1)) for s in seq]
    seq = [s for s in seq if len(s)*2.5 > 2*win]
    
        
    unitIDs = [unit for unit in list(tr_units.keys()) if '_' in unit]
    unitIDs = [unit for unit in unitIDs if re.split('_', unit)[1] == 'good']
    
    if len(ids) == 0:
        ids = unitIDs

    t = np.arange(-iwin, iwin+1) * dt
    n = len(t)
    data = []
    frd = []
    for unit in ids:
        tr = tr_units[unit]*1000
        trd = downsample_overlap(tr, nwin, noverlap)
        frd.append(trd)

        CC = []        
        for s in seq:
            ta = s[0]*sdt
            tb = (s[-1]+1)*sdt
                
            idxa = np.argmin(np.abs(tband-ta))
            idxb = np.argmin(np.abs(tband-tb))
            
            pow_band_cut = pow_band[idxa:idxb+1]
            trd_cut = trd[idxa:idxb+1]
        
            pow_band_cut -= pow_band_cut.mean()
            trd_cut -= trd_cut.mean()
        
            m = np.min([pow_band_cut.shape[0], trd_cut.shape[0]])
            # Say we correlate x and y;
            # x and y have length m
            # then the correlation vector cc will have length 2*m - 1
            # the center element with lag 0 will be cc[m-1]
            norm = np.nanstd(trd_cut) * np.nanstd(pow_band_cut)
            # for used normalization, see: https://en.wikipedia.org/wiki/Cross-correlation
            
            #xx = scipy.signal.correlate(dffd[1:m], pow_band[0:m - 1])/ norm
            xx = (1/m) * scipy.signal.correlate(trd_cut, pow_band_cut)/ norm
            ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
            ii = [int(i) for i in ii]

            ii = np.concatenate((np.arange(m-iwin-1, m), np.arange(m, m+iwin, dtype='int')))
            # note: point ii[iwin] is the "0", so xx[ii[iwin]] corresponds to the 0-lag correlation point
            CC.append(xx[ii])
            
        CC = np.array(CC)
        data += zip([unit]*n, t, np.nanmean(CC, axis=0))
    
    df = pd.DataFrame(data=data, columns=['ID', 't', 'cc'])
    FRd = np.array(frd)

    data = []
    if cc_max_win == 0:
        cc_max_win = 120
    for unit in df.ID.unique():
        dfs = df[(df.ID==unit) & (df.t>=-cc_max_win) & (df.t<=cc_max_win)]
        t = np.array(dfs.t)
        cc = np.array(dfs['cc'])
        
        if sign == 'no':
            i = np.argmax(np.abs(cc))
        elif sign == '-':
            i = np.argmin(cc)
        else:
            i = np.argmax(cc)
        tmax = t[i]
        mmax = cc[i]
        
        data += [[unit, tmax, mmax]]
        
    dfmax = pd.DataFrame(data=data, columns=['ID', 'tmax', 'max'])

    return df, dfmax, pow_band, FRd



def is_cycle(mouse, units, band, ids=[], mouse_config='mouse_config.txt', 
             ma_thr=10, dt=2.5, nstates=21, nsections=5,
             filt_speeg = True, box_filt=[1,4], pnorm_spec=True, wfreq=[0.01, 0.03], nrem_thr=120,
             nsmooth=0, kcuts=[], pzscore=True, 
             min_irem_dur=0, wake_perc=1):
    """
    For each consolidated NREM episode determine the infraslow oscillation using Hilbert transform
    and then plot the average firing rate throughout one infraslow cycle.


    Parameters
    ----------
    mouse : TYPE
        DESCRIPTION.
    units : TYPE
        DESCRIPTION.
    band : TYPE
        DESCRIPTION.
    ids : TYPE, optional
        DESCRIPTION. The default is [].
    mouse_config : TYPE, optional
        DESCRIPTION. The default is 'mouse_config.txt'.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    dt : TYPE, optional
        DESCRIPTION. The default is 2.5.
    nstates : TYPE, optional
        DESCRIPTION. The default is 21.
    filt_speeg : TYPE, optional
        DESCRIPTION. The default is True.
    box_filt : TYPE, optional
        DESCRIPTION. The default is [1,4].
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is True.
    wfreq : TYPE, optional
        DESCRIPTION. The default is [0.01, 0.03].
    nrem_thr : TYPE, optional
        DESCRIPTION. The default is 120.
    nsmooth : TYPE, optional
        DESCRIPTION. The default is 0.
    kcut : TYPE, optional
        DESCRIPTION. The default is [].
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    dfm : TYPE
        DESCRIPTION.

    """    
    from scipy.signal import hilbert
    
    if mouse_config == '':
        mouse_config = 'mouse_config.txt'
    
    path = load_config(mouse_config)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)
    M = sleepy.load_stateidx(ppath, name)[0]
        
    nhypno = np.min((units.shape[0], M.shape[0]))
    tidx = np.arange(0, nhypno)    
    # if len(kcut) > 0:
    #     kidx = np.arange(int(kcut[0]/dt), int(kcut[-1]/dt))
    #     tidx = np.setdiff1d(tidx, kidx)
    #     nhypno = len(tidx)
    M = M[0:nhypno]
    
    # NEW 07/01/22:
    # get the indices (in brainstate time) that we're going to completely discard:
    if len(kcuts) > 0:
        kidx = []
        for kcut in kcuts:
            a = int(kcut[0]/dt)
            b = int(kcut[-1]/dt)
            if b > len(M):
                b = len(M)
            kidx += list(np.arange(a, b))
        
        tidx = np.setdiff1d(tidx, kidx)
        M = M[tidx]
        nhypno = len(tidx)        
    ###########################################################################    

    seq = sleepy.get_sequences(np.where(M==2)[0])
    for s in seq:
        if np.round(len(s)*dt) <= ma_thr:
            M[s] = 3
    seq = sleepy.get_sequences(np.where(M>=3)[0])

    P = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'), squeeze_me=True)
    SPEEG = P['SP']
    freq = P['freq']
    dfreq = freq[1]-freq[0]
    isigma = np.where((freq>=band[0])&(freq<=band[1]))[0]
    SPEEG = SPEEG[:,0:nhypno]    
    SPEEG_orig = SPEEG.copy()

    if filt_speeg:
        filt = np.ones(box_filt)
        filt = np.divide(filt, filt.sum())
        SPEEG = scipy.signal.convolve2d(SPEEG, filt, boundary='symm', mode='same')

    if pnorm_spec:
        sp_mean = SPEEG.mean(axis=1)
        SPEEG = np.divide(SPEEG, np.tile(sp_mean, (SPEEG.shape[1], 1)).T)
        sigma_pow = SPEEG[isigma,:].mean(axis=0)
        
        SPEEG_orig = np.divide(SPEEG_orig, np.tile(sp_mean, (SPEEG.shape[1], 1)).T)
        sigma_pow_orig = SPEEG_orig[isigma,:].mean(axis=0)        
    else:
        sigma_pow = SPEEG[isigma, :].sum(axis=0)*dfreq

    # load unit data: #########################################################
    unitIDs = [unit for unit in list(units.columns) if '_' in unit]
    unitIDs = [unit for unit in units.columns if re.split('_', unit)[1] == 'good']
    if ids == []:
        ids = unitIDs
    R = np.zeros((len(ids), len(tidx)))    
        
    for i,unit in enumerate(ids):        
        tmp = sleepy.smooth_data(np.array(units[unit]),nsmooth)
        if pzscore:
            R[i,:] = (tmp[tidx] - tmp[tidx].mean()) / tmp[tidx].std()
        else:
            R[i,:] = tmp[tidx]
    ###########################################################################

    sr_is = 1/dt
    w1 = wfreq[0] / (sr_is/2.0)
    w2 = wfreq[1] / (sr_is/2.0)
    sigma_pow_filt = sleepy.my_bpfilter(sigma_pow, w1, w2)
    total_res = hilbert(sigma_pow_filt)

    data = []
    phase_bins = np.linspace(-np.pi, np.pi, nstates)
    for s in seq:
        if len(s)*dt >= nrem_thr:
            res = total_res[s]
            instantaneous_phase = np.angle(res)
            
            # get minima in phase
            lidx  = np.where(instantaneous_phase[0:-2] > instantaneous_phase[1:-1])[0]
            ridx  = np.where(instantaneous_phase[1:-1] <= instantaneous_phase[2:])[0]
            thidx = np.where(instantaneous_phase[1:-1]<-1)[0]
            sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
            
            sidx_corr = []
            for (a,b) in zip(sidx[0:-1], sidx[1:]):
                ii = np.where(instantaneous_phase[a:b]>1)[0]
                if len(ii) > 0:
                    sidx_corr.append(b)
            
            if len(sidx) >= 2:
                a = sidx[0]
                b = sidx[1]
                ii = np.where(instantaneous_phase[a:b]>1)[0]
                if len(ii) > 0:
                    sidx_corr = [a] + sidx_corr

            sidx = np.array(sidx_corr)+s[0]

            if len(sidx) >= 2:

                for (p,q) in zip(sidx[0:-1], sidx[1:]):
                    perc, isection = get_cycle_perc(M, p, nsections, min_irem_dur=min_irem_dur, wake_perc=wake_perc)
                    if np.isnan(perc):
                        continue
                    
                    for i,unit in enumerate(ids):
                        fr = R[i,:]                
                        fr_cut = fr[p:q]
                        sigma_cut = sigma_pow_orig[p:q]
                        fr_morph = time_morph(fr_cut, nstates)
                        sigma_morph = time_morph(sigma_cut, nstates)
    
                        #data += zip([unit]*nstates, range(nstates), fr_morph, phase_bins, sigma_morph, [str(isection)]*nstates)
                        # 08/26/24 changing type of last element to int
                        data += zip([unit]*nstates, range(nstates), fr_morph, phase_bins, sigma_morph, [isection]*nstates)
    
    df = pd.DataFrame(data=data, columns=['ID', 'section', 'fr', 'phase', 'sigma', 'irem'])
    
    dfm = df.groupby(['ID', 'section']).mean().reset_index()
    
    df_section = df[~df.irem.isna()].groupby(['ID', 'section', 'phase', 'irem']).mean().reset_index()
    
    return dfm, df_section, df


    
def get_cycle_perc(M, si, nsections, min_irem_dur=0, wake_perc=1, dt=2.5):
    
    seq = sleepy.get_sequences(np.where(M==1)[0])

    found = False
    for pp,qq in zip(seq[0:-1], seq[1:]):        
        if pp[-1] <= si < qq[0]:
            found = True
            break
    
    # It's the last REM period in the recording:
    # so there's no inter-REM period
    if found == False:
        #print('not found - cond 1')
        return np.nan, np.nan
            
    #idur = qq[0] - pp[-1]
    idx = np.arange(pp[-1]+1, qq[0])
    
    if len(idx)*dt <= min_irem_dur:
        #print('not found - cond 2')
        return np.nan, np.nan
                
    mcut = M[idx]
    nrem_idx = np.where(mcut == 3)[0]
    wake_idx = np.where(mcut == 2)[0]
    
    if len(wake_idx) / len(idx) > wake_perc:
        #print('not found - cond 3')
        return np.nan, np.nan
    
    x = si - pp[-1]
    
    perc = len(np.where(nrem_idx <= x)[0]) / len(nrem_idx)
    idur = len(idx)
    
    perc = x/idur
    
    quantiles = np.arange(0, 101, 100/nsections) / 100
    
    i = 0
    for p,q in zip(quantiles[0:-1], quantiles[1:]):
        if p <= perc < q:
            break
        i += 1
    
    return perc, i
    


def laser_triggered_train(ppath, name, train, pre, post, nbin=1, offs=0, iters=1, istate=-1, sf=0, laser_end=-1, pplot=True):
    """
    plot each laser stimulation trial in a raster plot, plot laser triggered firing rate and
    bar plot showing firing rate before, during, and after laser stimulation
    :param ppath: folder with sleep recording
    :param name: name of sleep recording
    :param train: 
    :param pre: time before laser onset [s]
    :param post: time after laser onset [s]
    :param nbin: downsample firing rate by a factor of $nbin
    :param offs: int, the first shown laser trial
    :param iters: int, show every $iters laser trial
    :param istate: int, only plot laser trials, where the onset falls on brain state $istate
                   istate=-1 - consider all states, 
                   istate=1 - consider only REM trials
                   istate=2 - consider only Wake trials 
                   istate=3 - consider only NREM trials
    :param sf: something factor for firing rates
    :param laser_end: if > -1, then use only the first $laser_end seconds during laser interval
                for fr_lsr
    :return: 
        fr_pre: firing rates before laser 
        fr_lsr: firing rates during laser 
        fr:     firing rates for complete time window including before, during, 
                and after laser
        t:      time vector
         
    """
    import matplotlib.patches as patches

    # load brain state
    M = sleepy.load_stateidx(ppath, name)[0]

    sr = sleepy.get_snr(ppath, name)
    dt = 1.0 / sr
    pre = int(np.round(pre/dt))
    post = int(np.round(post/dt))
    laser = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(laser)
    
    # only collect laser trials starting during brain state istate
    tmps = []
    tmpe = []
    if istate > -1:
        nbin_state = int(np.round(sr) * 2.5)
        for (i,j) in zip(idxs, idxe):
            if M[int(i/nbin_state)] == istate:
                tmps.append(i)
                tmpe.append(j)                
        idxs = np.array(tmps)
        idxe = np.array(tmpe)

    laser_dur = np.mean(idxe[offs::iters] - idxs[offs::iters] + 1)*dt
    
    len_train = train.shape[0]
    raster = []
    fr_pre = []
    fr_lsr = []
    fr_post = []
    for (i,j) in zip(idxs[offs::iters], idxe[offs::iters]):
        if (i - pre >= 0) and (i + post < len_train):

            raster.append(train[i - pre:i + post + 1])
            fr_pre.append(train[i-pre:i].mean())
            if laser_end == -1:
                fr_lsr.append(train[i:j+1].mean())
            else:
                k = int(laser_end/dt)
                fr_lsr.append(train[i:i+k].mean())
                
            fr_post.append(train[j+1:j+post+1].mean())

    fr_pre = np.array(fr_pre) * sr
    fr_lsr = np.array(fr_lsr) * sr
    fr_post = np.array(fr_post) * sr

    # time x trials
    raster = np.array(raster).T
    # downsampling
    raster = downsample_mx(raster, nbin).T * sr
    dt = nbin*1.0/sr
    t = np.arange(0, raster.shape[1]) * dt - pre*(1.0/sr)
    fr = raster.mean(axis=0)
     
    # NEW 06/05/24 ############################################################
    if sf > 0:
        fr2 = np.concatenate((np.flip(fr), fr, np.flip(fr)))
        n = fr.shape[0]
        
        fr2 = sleepy.smooth_data(fr2, sf)
        fr = fr2[n:2*n]
    ###########################################################################        
    
    if pplot:
        sleepy.set_fontarial()
        plt.figure(figsize=(3.5,5))
        ax = plt.axes([0.2, 0.4, 0.75, 0.25])
        
        ax.plot(t, fr, color='black')
        max_fr = np.max(raster.mean(axis=0))
        max_fr = np.max(fr)
        ylim_fr = max_fr + 0.05*max_fr
        plt.ylim([0, ylim_fr])
        ax.add_patch(patches.Rectangle((0, 0), laser_dur, ylim_fr, 
                                       facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
        plt.xlim((t[0], t[-1]))
        plt.xlabel('Time (s)')
        plt.ylabel('FR (spikes/s)')
        sleepy.box_off(ax)

        ax = plt.axes([0.2, 0.7, 0.75, 0.2])
        R = raster.copy()
        R[np.where(raster>0)] = 1

        for i in range(R.shape[0]):
            r = R[i,:]
            idx = np.where(r == 1)[0]
            plt.plot(t[idx], np.ones((len(idx),))*i, '.', ms=1, color='k')
        ax.set_xlim((t[0], t[-1]))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylabel('Trial no.')
        ax.add_patch(patches.Rectangle((0, 0), laser_dur, R.shape[0], 
                                       facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))

    return fr_pre, fr_lsr, fr, t
    


def upsample_mx(x, nbin):
    """
    if x is a vector:
        upsample the given vector $x by duplicating each element $nbin times
    if x is a 2d array:
        upsample each matrix by duplication each row $nbin times        
    """
    if nbin == 1:
        return x
    
    nelem = x.shape[0]
    if x.ndim == 1:        
        y = np.zeros((nelem*nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    else:
        y = np.zeros((nelem*nbin,x.shape[1]))
        for k in range(nbin):
            y[k::nbin,:] = x

    return y



def downsample_mx(X, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the columns in x by replacing nbin consecutive rows
    bin by their mean
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down * nbin, :]
    X_down = np.zeros((n_down, X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin):
        idx = list(range(i, int(n_down * nbin), int(nbin)))
        X_down += X[idx, :]

    return X_down / nbin



def downsample_vec(x, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive 
    bin by their mean 
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]

    return x_down / nbin



def downsample_overlap(x, nwin, noverlap):
    """
    Say,
    len(x)=10
    nwin=5
    nolverap=3
    1 2 3 4 5 6 7 8 9 10

    1 2 3 4 5
        3 4 5 6 7
            5 6 7 8 9

    :param x:
    :param nwin:
    :param noverlap:
    :return:
    """
    nsubwin = nwin-noverlap
    n_down = int(np.floor((x.shape[0]-noverlap)/nsubwin))
    x_down = np.zeros((n_down,))
    j = 0
    for i in range(0, x.shape[0]-nwin+1, nsubwin):
        x_down[j] = x[i:i+nwin].mean()
        j += 1

    return x_down



def time_morph(X, nstates):
    """
    upsample vector or matrix X to nstates states
    :param X, vector or matrix; if matrix, the rows are upsampled.
    :param nstates, number of elements or rows of returned vector or matrix

    I want to upsample m by a factor of x such that
    x*m % nstates == 0,
    a simple soluation is to set x = nstates
    then nstates * m / nstates = m.
    so upsampling X by a factor of nstates and then downsampling by a factor
    of m is a simple solution...
    """
    m = X.shape[0]
    A = upsample_mx(X, nstates)
    # now we have m * nstates rows
    if X.ndim == 1:
        Y = downsample_vec(A, int((m * nstates) / nstates))
    else:
        Y = downsample_mx(A, int((m * nstates) / nstates))
    # now we have m rows as requested
    return Y



def xcorr_frs(unit1, unit2, window, ndown, sr=1000, pplot=False):     
    dt = 1/sr
    dt_dn = dt * ndown
    
    iwin = int(window / dt_dn)

    unit1 = unit1 * 1000
    unit2 = unit2 * 1000
    
    unit1 = sleepy.downsample_vec(unit1, ndown)
    unit2 = sleepy.downsample_vec(unit2, ndown)
        
    m = np.min([unit1.shape[0], unit2.shape[0]])
    norm = np.nanstd(unit1) * np.nanstd(unit2)
    xx = (1/m) * scipy.signal.correlate(unit1, unit2)/ norm

    ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
    ii = [int(i) for i in ii]

    t = np.arange(-iwin, iwin+1) * dt_dn

    if pplot:
        plt.figure()
        plt.plot(t, xx[ii])

    return xx[ii], t, unit1, unit2



def cc_jitter(unit1, unit2, window, sr, plot=True, plt_win=0.5, version=1):
    """
    Interpretation

    Say b follows a, e.g.,
    a = [1,0,1,0]
    b = [0,1,0,1]
        
    cross-correlation:
    [1, 0, 2, 0, 1, 0, 0]
              |
             t=0

    then there's a positive peak at a negative time point.
    In other words, negative time points mean
    a precedes b

    Parameters
    ----------
    unit1 : np.array
        spike train of unit 1.
    unit2 : np.array
        spike train of unit 2.
    window : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    plt_win : TYPE, optional
        DESCRIPTION. The default is 0.5.
    version : int
        1 - no normalization for length of spike train
        2 - normalize for length of spike train

    Returns
    -------
    np.array
        time vector.
    np.array
        Cross-correlogram.

    """    
    # number of data points per window:
    iwin = window * sr
    nsplit = int(np.floor(len(unit1) / iwin))
    
    un1=np.array_split(unit1,nsplit)
    new_un1=[]
    #un2=np.array_split(unit2,nsplit)
    #new_un2=[]
    
    for count,value in enumerate(un1):
        new_un1.append(np.random.permutation(value))
        #new_un2.append(np.random.permutation(un2[count]))
    fin1=np.concatenate(new_un1)
    #fin2=np.concatenate(new_un2)
    #fin2 = unit2.copy()
    fin2 = unit2

    # 01/02/24 - NOTE: added 'n' to normalization!     
    # not if unit1 has len n, then the correlation output has len 2*n-1
    #n = len(unit1) / 1000
    
    if version == 1:
        n = 1
         
        norm = n * np.sqrt(fin1.mean() * fin2.mean()) * sr
        CC = scipy.signal.correlate(fin1, fin2) / norm
        
        norm = n * np.sqrt(unit1.mean() * unit2.mean()) * sr
        CCorig = scipy.signal.correlate(unit1, unit2) / norm
    
        corrCC = CCorig-CC
    else:
        n = len(unit1)
         
        norm = n * np.sqrt(fin1.mean() * fin2.mean()) * sr
        CC = scipy.signal.correlate(fin1, fin2) / norm
        
        norm = n * np.sqrt(unit1.mean() * unit2.mean()) * sr
        CCorig = scipy.signal.correlate(unit1, unit2) / norm
    
        corrCC = CCorig-CC
        
    
    # not if unit1 has len n, then the correlation output has len 2*n-1
    n = len(unit1)
    # the center point (= 0 s) is element (n-1)
    t = np.arange(-n+1, n) * (1/sr)
        
    idx = np.where((t>=-plt_win) & (t<=plt_win))[0]
    
    if plot:
        plt.figure()
        ax = plt.subplot(111)

        plt.plot(t[idx], corrCC[idx], 'k')
        plt.xlabel('Lag (ms)')
        
        ylim = ax.get_ylim()
        plt.plot([0,0], ylim, 'k--')

        sns.despine()
            
    return corrCC[idx], t[idx]
   
            
           
def cc_jitter2(unit1, unit2, window, sr, plot=True, plt_win=0.5):
    """
    Interpretation

    Say b follows a, e.g.,
    a = [1,0,1,0]
    b = [0,1,0,1]
    
    
    cross-correlation:
    [1, 0, 2, 0, 1, 0, 0]
              |
             t=0

    then there's a positive peak at a negative time point.
    In other words, negative time points mean
    a precedes b

    Parameters
    ----------
    unit1 : TYPE
        DESCRIPTION.
    unit2 : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    plt_win : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    calc_win = int(2*sr)
    ncalc_win = int(np.floor(len(unit1)/calc_win))
    
    # number of data points per window:
    iwin = window * sr
    nsplit = int(np.floor(len(unit1) / iwin))
    
    un1=np.array_split(unit1,nsplit)
    new_un1=[]
    un2=np.array_split(unit2,nsplit)
    new_un2=[]
    
    for count,value in enumerate(un1):
        new_un1.append(np.random.permutation(value))
        new_un2.append(np.random.permutation(un2[count]))
    fin1=np.concatenate(new_un1)
    #fin2=np.concatenate(new_un2)
    fin2 = unit2.copy()

    #jnorm = np.sqrt(fin1.mean() * fin2.mean()) * 1000
    #norm = np.sqrt(unit1.mean() * unit2.mean()) * 1000
    
    corrCC = []     
    for i in range(ncalc_win-1):
        a = i*calc_win
        b = (i+1)*calc_win
        u1 = unit1[a:b]
        u2 = unit2[a:b]

        ju1 = fin1[a:b]
        ju2 = fin2[a:b]

        norm = np.sqrt(u1.mean() * u2.mean()) * 1000
        CC = scipy.signal.correlate(u1, u2)/norm

        
        jnorm = np.sqrt(ju1.mean() * ju2.mean()) * 1000
        jCC = scipy.signal.correlate(ju1, ju2)/jnorm
        
        aa = CC - jCC

        corrCC.append(aa)    

    corrCC = np.nanmean(np.array(corrCC), axis=0)

    # not if unit1 has len n, then the correlation output has len 2*n-1
    n = len(u1)
    # the center point (= 0 s) is element (n-1)
    t = np.arange(-n+1, n) * (1/sr)

    #pdb.set_trace()

        
    idx = np.where((t>=-plt_win) & (t<=plt_win))[0]
    
    if plot:
        plt.figure()
        
        
        #plt.plot(t[idx], CC[idx])
        #plt.plot(t[idx], CC1[idx])
        plt.plot(t[idx], corrCC[idx])
        
        #plt.plot(corrCC)
        
        
    
    # if plot==True:
    #     n = len(unit1)
    #     delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    #     delay = delay_arr[np.argmax(corrCC)]
    #     plt.figure()
    #     plt.plot(delay_arr, corrCC)
    #     plt.title(f'unit {unit1.name} is {delay} behind unit {unit2.name}'+ '\n'+ 'Lag: '  + str(np.round(delay, 3)) + ' s')
    #     plt.xlabel('Lag')
    #     plt.ylabel('Correlation coeff')
    #     plt.show()
    
    #return corrCC[idx], t[idx] 
           



def cc_jitter_irem(unit1, unit2, window, sr, M, perc=3, plot=True, plt_win=0.5, version=2):
    """
    Jitter-corrected cross-correlation for different intervals of the inter-REM interval

    Parameters
    ----------
    unit1 : TYPE
        DESCRIPTION.
    unit2 : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    perc : int, optional
        Number of intervals (percentiles) that the inter-REM interval is divided into.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    plt_win : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    corr_dict : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    """

    sdt = 2.5
    seq = sleepy.get_sequences(np.where(M==1)[0])
    tvec = np.arange(0, len(unit1)) * 1/sr
    
    corr_dict = {p:0 for p in range(1, perc+1)}

    for p in range(1, perc+1):
        perc_idx = np.array([], dtype='int')

        for si,sj in zip(seq[0:-1], seq[1:]):
            irem_idx = np.arange(si[-1]+1, sj[0], dtype='int')
            irem_start = (si[-1]+1) * sdt
                
            irem_dur = len(irem_idx) * sdt            
            perc_dur = irem_dur / perc
            
            perc_start = irem_start + (p-1) * perc_dur
            perc_end   = irem_start + p * perc_dur

            idx = np.where((tvec >= perc_start) & (tvec < perc_end))[0]            
            perc_idx = np.concatenate((perc_idx, idx))            
                        
        cc, t = cc_jitter(unit1[perc_idx], unit2[perc_idx], window, sr, plot=False, plt_win=0.1, version=version)
        corr_dict[p] = cc
                
    return corr_dict, t



def cc_jitter_refr(unit1, unit2, window, sr, M, nrem_only=True, ma_thr=10, plot=True, plt_win=0.5, version=2):
    """
    Jitter-corrected cross-correlation for different intervals of the inter-REM interval

    Parameters
    ----------
    unit1 : TYPE
        DESCRIPTION.
    unit2 : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    perc : int, optional
        Number of intervals (percentiles) that the inter-REM interval is divided into.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    plt_win : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    corr_dict : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    """
    NBIN = 2500
    corr_dict = {'refr':[], 'perm':[]}

    refr = add_refr(M, ma_thr)[1]

    refr_idx = np.where(refr==1)[0]
    perm_idx = np.where(refr==2)[0]
    nrem_idx = np.where(M==3)[0]

    if nrem_only:
        refr_idx = np.intersect1d(refr_idx, nrem_idx)
        perm_idx = np.intersect1d(perm_idx, nrem_idx)
        

    seq = sleepy.get_sequences(refr_idx)
    idx = []
    for s in seq:
        idx += list(range(s[0]*NBIN, s[-1]*NBIN))
        

    refr_idx = idx
        
    seq = sleepy.get_sequences(perm_idx)
    idx = []
    for s in seq:
        idx += list(range(s[0]*NBIN, s[-1]*NBIN))
    perm_idx = idx


    cc, t = cc_jitter(unit1[refr_idx], unit2[refr_idx], window, sr, plot=False, plt_win=0.1, version=version)
    corr_dict['refr'] = cc

    cc, t = cc_jitter(unit1[perm_idx], unit2[perm_idx], window, sr, plot=False, plt_win=0.1, version=version)
    corr_dict['perm'] = cc

    return corr_dict, t



def pca(data, dims=2):
    """
    @data is a 2D matrix.
    each row is an observation (a data point).
    each column is a variable (or dimension)
    
    The goal is to reduce the number of dimensions
    
    :return projection of data onto eigenvectors, eigenvalues, eigenvectors
    
    """
    from scipy import linalg as LA
    
    m, n = data.shape
    # mean center the data, i.e. calculate the average "row" and subtract it
    # from all other rows
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    # Note:  each column represents a variable, while the rows contain observations.
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R, check_finite=True)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    # return projection of data onto eigenvectors, eigenvalues and eigenvectors  
    return np.dot(evecs.T, data.T).T, evals, evecs
    


def pca_svd(Y, ndim=2):
    """
    Each row is a sample (or data point or observation).
    Each column is a variable or dimension.
    
    We want to reduce the number of dimensions (or columns)

    Y = samples x dimensions
    
    
    For example for multiple neurons (dimensions) recorded over time (each time point is a sample),
    The matrix arrangement should be
    
    
    Y = time x neurons
    
    
    See also, 
    https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    """
    nsample = Y.shape[0]
    
    Y = Y / np.sqrt(nsample-1)
    
    Y = Y - Y.mean(axis=0)
    U,S,Vh = scipy.linalg.svd(Y)
    
    PC = np.dot(U,np.diag(S))[:,0:ndim]
    # Alternative:
    # PC = np.dot(Vh,Y.T)[0:ndim,:].T
    
    
    var = S**2
    var_total = np.sum(S**2)

    # Calculate the cumulative variance explained, i.e. how much
    # of the variance is captured by the first i principal components. 
    p = []
    for i in range(1,len(var)+1):
        s = np.sum(var[0:i])
        p.append(s/var_total)

    # c = 1 / np.sqrt(nsample-1)
    return PC, 1 * Vh[0:ndim,:].T, p



def pca_bootstrap(Y, ndim=2, nboot=100):
    import numpy.random as rand

    
    nvar = Y.shape[1]
    
    pc_dict = {i : np.zeros((nboot, Y.shape[0])) for i in range(ndim)}
    
    for b in range(nboot):
        iselect = rand.randint(0, nvar, (nvar,))        
        Y2 = Y[:, iselect].copy()
        
        
        PC = pca_svd(Y2)[0]
        
        for i in range(0, ndim):
            pc_dict[i][b,:] = PC[:,i]
    
    
    return pc_dict



def smooth_causal(data, tau):

    p = np.inf
    L = 10
    while p > 0.0001:
        L += 10
        p = _efilt(L, tau)

    filt = _efilt(np.arange(0, L), tau)
    filt = filt / np.sum(filt)
    z = np.zeros(len(filt))
    filt = np.concatenate((np.flipud(filt), z))

    filt = np.flipud(filt)

    y = scipy.signal.convolve(data, filt, mode='same')

    return y

def _efilt(x, tau):
    y = (1 / tau) * np.exp(-x / tau)
    return y



def calculate_lfp_spectrum(ppath, name, LFP, fres=0.5):
    """
    calculate EEG and EMG spectrogram used for sleep stage detection.
    Function assumes that data vectors EEG.mat and EMG.mat exist in recording
    folder ppath/name; these are used to calculate the powerspectrum
    fres   -   resolution of frequency axis
    
    all data saved in "true" mat files
    :return  EEG Spectrogram, EMG Spectrogram, frequency axis, time axis
    """
    
    SR = sleepy.get_snr(ppath, name) * 2.5
    print(SR)
    swin = round(SR)*5
    fft_win = round(swin/5) # approximate number of data points per second
    if (fres == 1.0) or (fres == 1):
        fft_win = int(fft_win)
    elif fres == 0.5:
        fft_win = 2*int(fft_win)
    else:
        print("Resolution %f not allowed; please use either 1 or 0.5" % fres)
    
    # Calculate EEG spectrogram
    Pxx, f, t = sleepy.spectral_density(LFP, int(swin), int(fft_win), 1/SR)

    spfile = os.path.join(ppath, name, 'lp_' + name + '.mat')    
    so.savemat(spfile, {'LP':Pxx, 'freq':f, 'dt':t[1]-t[0],'t':t})
    
    print(r'Saved LFP Spectrum to file %s.' % spfile)
    
    return Pxx, f, t



def corr_eeg(mouse, config_file='mouse_config.txt'):
    """
    Calculate EEG/EMG artifact and subtract it from EEG/EMG for
    correction

    The artifact looks the following way:
    Every 1000ms there's a sharp wave going down, 500ms later there's a sharp
    wave going up.
    So if we know when the first down wave happens, we know
    when every up and down wave will come

    Parameters
    ----------
    mouse : str
        mouse name.
    config_file : str, optional
        file name of mouse configuration file. 


    """
    path = load_config(config_file)[mouse]['SL_PATH']
    ppath, name = os.path.split(path)

    
    file_eeg_orig = os.path.join(ppath, name, 'EEG_orig.mat')
    file_emg_orig = os.path.join(ppath, name, 'EMG_orig.mat')
    

    if os.path.isfile(file_eeg_orig):
        eeg = so.loadmat(file_eeg_orig, squeeze_me=True)['EEG']
    else:
        try:
            eeg = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
        except:
            eeg = np.array(h5py.File(os.path.join(ppath, name, 'EEG.mat'),'r').get('EEG'))        
            eeg = np.squeeze(eeg)


    if os.path.isfile(file_emg_orig):
        emg = so.loadmat(file_emg_orig, squeeze_me=True)['EMG']            
    else:
        try: 
            emg = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
        except:
            emg = np.array(h5py.File(os.path.join(ppath, name, 'EMG.mat'),'r').get('EMG'))        
            emg = np.squeeze(emg)

    # save original EEG/EMG
    eeg_orig = eeg.copy()
    emg_orig = emg.copy()
    
    #emg = sleepy.my_notchfilter(emg)
    emgl = sleepy.my_lpfilter(emg, 0.2)
    emgl = emgl[0:10*60*1000]
    
    emg_cut = emg[0:10*60*1000]
    eeg_cut = eeg[0:10*60*1000]
    
    # Autocorrelation of low-pass filtered EMG to detect
    # the interval in which the EMG artifact is repeated
    a = emgl-emgl.mean()
    m = len(a)
    norm = np.nanstd(a) * np.nanstd(a)
    xx = (1/m) * scipy.signal.correlate(a, a, 'same') / norm    
    plt.figure()
    plt.plot(xx)
    plt.xlabel('Index lag')
    
    # get indices of peaks and troughts (valleys)    
    peak_idx = scipy.signal.find_peaks(xx, prominence=0.1, distance=900)[0]
    valley_idx = scipy.signal.find_peaks(-xx, prominence=0.1, distance=900)[0]
    
    # get distance between two consecutive valleys
    valley_dist = int(np.round(np.mean(np.diff(valley_idx))))
    hvalley_dist = int(valley_dist/2)
    peak_dist = int(np.round(np.mean(np.diff(peak_idx))))
    
    print(valley_dist)
    print(peak_dist)

    # Determine the first valley (offset of the first artifact going down)
    x = []
    y = []
    for i in np.arange(0, valley_dist):
        tmp = np.sum(emg_cut[i::valley_dist])
        y.append(tmp)
        x.append(i)

    plt.figure()
    plt.plot(x, y)       
    plt.plot(np.diff(y))
    
    # the first valley is when the first downwards jump happens
    # if we now ifirst_valley, we know every index where a down or
    # up wave will happen: The next up wave comes 500ms later, the
    # next down wave comes 1 s later.
    ifirst_valley = np.argmin(np.diff(y))+1    
    plt.plot(ifirst_valley, y[ifirst_valley], 'r*', label='Offset of first valley')
    
    iwin = int(valley_dist/4)
    
    if ifirst_valley < iwin:
        ifirst_valley += valley_dist
        
    wave = []
    wave2=[]
    for i in np.arange(ifirst_valley, len(emg_cut), valley_dist):
        tmp = emg_cut[i-iwin:i+iwin]
        wave.append(tmp)
        
        tmp = eeg_cut[i-iwin:i+iwin]
        wave2.append(tmp)
        
    wave_down_emg = np.array(wave[1:-1]).mean(axis=0)
    wave_down_eeg = np.array(wave2[1:-1]).mean(axis=0)


    wave, wave2 = [], []
    for i in np.arange(ifirst_valley+hvalley_dist, len(emg_cut)-hvalley_dist, valley_dist):
        tmp = emg_cut[i-iwin:i+iwin]
        wave.append(tmp)
        
        tmp = eeg_cut[i-iwin:i+iwin]
        wave2.append(tmp)
        
    wave_up_emg = np.array(wave).mean(axis=0)
    wave_up_eeg = np.array(wave2).mean(axis=0)

    
    plt.figure()
    plt.plot(wave_down_emg, label='EMG down')
    plt.plot(wave_up_emg, label='EMG up')
    
    plt.plot(wave_down_eeg, label='EEG down')
    plt.plot(wave_up_eeg, label='EEG up')
    plt.legend()
    
    # correct EEG/EMG signals
    for i in np.arange(ifirst_valley, len(emg)-valley_dist, valley_dist):
        eeg[i-iwin:i+iwin] -= wave_down_eeg
        emg[i-iwin:i+iwin] -= wave_down_emg
        
    for i in np.arange(ifirst_valley+hvalley_dist, len(emg)-valley_dist, valley_dist):
        eeg[i-iwin:i+iwin] -= wave_up_eeg
        emg[i-iwin:i+iwin] -= wave_up_emg
    
    
    plt.figure()
    plt.subplot(211)
    plt.plot(eeg_orig[0:len(eeg_cut)], label='Orig. EEG')
    plt.plot(eeg[0:len(eeg_cut)], label='Corrected EEG')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(emg_orig[0:len(eeg_cut)], label='Orig. EMG')
    plt.plot(emg[0:len(eeg_cut)], label='Corrected EMG')

    
    # save the original and corrected EEG/EMG signals
    so.savemat(os.path.join(ppath, name, 'EEG_orig.mat'), {'EEG':eeg_orig})
    so.savemat(os.path.join(ppath, name, 'EMG_orig.mat'), {'EMG':emg_orig})
    
    so.savemat(os.path.join(ppath, name, 'EEG.mat'), {'EEG':eeg})
    so.savemat(os.path.join(ppath, name, 'EMG.mat'), {'EMG':emg})
    
    # re-calculate EEG and EMG spectrograms    
    inp = input('Overwrite EEG.mat and EMG.mat as well as the sp_*.mat and msp_* files? (yes|no)\n')    
    if inp == 'yes':
        # save the original and corrected EEG/EMG signals
        so.savemat(os.path.join(ppath, name, 'EEG_orig.mat'), {'EEG':eeg_orig})
        so.savemat(os.path.join(ppath, name, 'EMG_orig.mat'), {'EMG':emg_orig})
        
        so.savemat(os.path.join(ppath, name, 'EEG.mat'), {'EEG':eeg})
        so.savemat(os.path.join(ppath, name, 'EMG.mat'), {'EMG':emg})
                
        sleepy.calculate_spectrum(ppath, name)
        
    return ifirst_valley


