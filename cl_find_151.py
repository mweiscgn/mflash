#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import numpy as np
from collections import OrderedDict

import libmf.micflash as mflash
import libmf.tools as tools

import cl_fofg_142 as fof
import cl_fellwalk_086 as fellwalk
import cl_dendrofind as ddg

from py27hash.hash import hash27

__author__ = "Michael Weis"
__version__ = "0.1.3.0"


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================
"""
def GetChemNet(hdfdata):
    si = hdfdata.read_raw('sim info')
    cflags = (si['cflags'][0]).decode()
    chemnet = 0
    for flag in flags:
        if 'DCHEMISTRYNETWORK' in flag.upper():
            chemnet = int(flag.split('=')[1])
    assert chemnet != 0
    return chemnet
    
chvars = {5:var_ch5,15:var_ch15} 
   

def OpenPlotfile(fn, **kwargs):
    print(fn)
    hdfdata = plotfile(fn, **kwargs)
    chemnet = GetChemNet(hdfdata)
    var_ch = chvars[chemnet]
    hdfdata.learn(mflash.var_mhd)
    hdfdata.learn(mflash.var_grid)
    hdfdata.learn(var_ch)
    return hdfdata
 """
# ==== CONSTANTS ===============================================================
from constants import *

# ==== BASE SETTINGS ===========================================================
var_ch = mflash.var_ch5 # XXX TRIPLE-CHECK ! XXX
plotfile_pattern = '*hdf5_*'
clist_basename = 'cl_l870_'

# ==== CELL SELECTION CRITERIA =================================================
cell_criteria = OrderedDict()

cell_criteria['COCx']   = {'abund_co':[1e-4, np.inf], }
cell_criteria['COCex8'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf],}

cell_criteria['COCxNL99']   = {'abund_co':[7e-5, np.inf], }
cell_criteria['COCex8NL99'] = {'abund_co':[7e-5, np.inf], 'cdto':[8., np.inf],}

cell_criteria['COCex'] = {'abund_co':[1e-4, np.inf], 'cdto':[7., np.inf],}
cell_criteria['COCex5'] = {'abund_co':[1e-4, np.inf], 'cdto':[5., np.inf],}
cell_criteria['COCex6'] = {'abund_co':[1e-4, np.inf], 'cdto':[6., np.inf],}
cell_criteria['COCxco']   = {'abund_co':[1.3975e-4, np.inf], }
cell_criteria['COCxp'] = {'abund_co':[1e-4, np.inf], 'p_sum':[-np.inf, 0.],}
cell_criteria['COCxphd'] = {'abund_co':[1e-4, np.inf], 'p_hdyn':[-np.inf, 0.],}
cell_criteria['CCE19x'] = {'abund_co':[1e-4, np.inf], 'dens':[1e-19, np.inf], }
cell_criteria['CC2E19x'] = {'abund_co':[1e-4, np.inf], 'dens':[2e-19, np.inf], }
cell_criteria['CC5E19x'] = {'abund_co':[1e-4, np.inf], 'dens':[5e-19, np.inf], }
cell_criteria['test1']   = {'abund_co':[1e-4, np.inf], }
cell_criteria['HIIregb']   = {'ihp':[.6, np.inf], }
cell_criteria['CCE20x'] = {'abund_co':[1e-4, np.inf], 'dens':[1e-20, np.inf], }
cell_criteria['COCxgg'] = {'abund_co':[1e-4, np.inf], 'absgrad_gpot':[5e-8, np.inf],}
cell_criteria['COCxggs2'] = {'abund_co':[1e-4, np.inf], 'aggps':[5e-8, np.inf],}
cell_criteria['COCxggs8l'] = {'abund_co':[1e-4, np.inf], 'gcritl':[1., np.inf],}
cell_criteria['COCxggs8q'] = {'abund_co':[1e-4, np.inf], 'gcritsq':[1., np.inf],}
cell_criteria['COCxggt2'] = {'abund_co':[1e-4, np.inf], 'absgrad_gpot':[5e-8, np.inf],}
cell_criteria['all'] = {'dens':[0., np.inf],}

#cell_criteria['COCex8-reg1'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf], 'reg1':[.5, np.inf]}
#cell_criteria['COCex8-reg2'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf], 'reg2':[.5, np.inf]}

# ==== ALGORITHM SETTINGS ======================================================
select_criteria_combos = OrderedDict()

select_criteria_combos['CCV7ax0_proj']  = {'axis':0,
      'dims':2, 'MaxJump':4, 'MinDipStd':.3, 'MinPts':2, 'periodic':True,
      'cl2d_intvar':'numdens', 'cl2d_threshold':8.*1.87e+21,
      'cldepth_var':'dens_co', 'cldepth_threshold':1e-21, 'MinPts3D':100, }
select_criteria_combos['CCE20g_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['CCE20x']}
select_criteria_combos['COCg_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['COCx']}
select_criteria_combos['COCg0_fw2']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['COCx']}
select_criteria_combos['COMax_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':{'abund_co':[1.375e-4, np.inf], }}
select_criteria_combos['COCg0_fw2s']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCx'}
select_criteria_combos['COCg0_fw2fs']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCx'}
select_criteria_combos['COCxgg_fw2fs']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCxgg'}
select_criteria_combos['COCxgg_stitch'] = {'base_criterion':'COCxgg', 'outer_criterion':'COCx', }
select_criteria_combos['COCxgg2_stitch'] = {'base_criterion':'COCxgg', 'outer_criterion':'COCx', }

select_criteria_combos['COCx'] = {'criteria':cell_criteria['COCx'], 'MinPts':100}
select_criteria_combos['COCex8'] = {'criteria':cell_criteria['COCex8'], 'MinPts':100}
select_criteria_combos['COCxs'] = {'criteria':cell_criteria['COCx'], 'MinPts':5}
select_criteria_combos['COCex8s'] = {'criteria':cell_criteria['COCex8'], 'MinPts':5}
select_criteria_combos['COCxgg'] = {'criteria':cell_criteria['COCxgg'], 'MinPts':100}
select_criteria_combos['COC_iso'] = {'criteria':cell_criteria['COCx'],
    'var':'gpot', 'direction':'up', 'dperc':1.0, 'MinPts':30}    
select_criteria_combos['COCex8p30:isocut'] = {'criteria':cell_criteria['all'],
    'var':'gpot', 'direction':'up', 'dperc':0.1}
select_criteria_combos['COCex8p30:dendroleaf'] = {'criteria':cell_criteria['all'],
    'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}

select_criteria_combos['COCxp30'] = {'criteria':cell_criteria['COCx'], 'MinPts':30}
select_criteria_combos['COCex8p30'] = {'criteria':cell_criteria['COCex8'], 'MinPts':30}

select_criteria_combos['COCxp30NL99'] = {'criteria':cell_criteria['COCxNL99'], 'MinPts':30}
select_criteria_combos['COCex8p30NL99'] = {'criteria':cell_criteria['COCex8NL99'], 'MinPts':30}

#select_criteria_combos['COCex8p30-reg1'] = {'criteria':cell_criteria['COCex8-reg1'], 'MinPts':30}
#select_criteria_combos['COCex8p30-reg2'] = {'criteria':cell_criteria['COCex8-reg2'], 'MinPts':30}


require_gridvars = ['COCxgg', 'COCxgg_stitch', 'COCxgg2_stitch', 'COCxggs2',
        'COCxggs8l', 'COCxggs8q', 'COCxggt2', 'COCxgg_fw2fs']



#===============================================================================
# ==== CRITERIA INTERFACE ======================================================
#===============================================================================

def GetCriterionLimits(criterion):
    #XXX THIS FUNCTION SHOULD NOT EXIST! REFACTOR CRITERION FORMAT!
    critdict = select_criteria_combos[criterion]
    critlim = dict()
    if '_fw' in criterion:
        if 'preselect' in critdict:
            critkey = critdict['preselect']
            critlim.update(GetCriterionLimits(critkey))
        if 'criteria' in critdict:
            critlim.update(critdict['criteria'])
    else:
        critlim.update(critdict)
    return critlim
    
    
#===============================================================================
# ==== ADDITIONAL GRIDVARS =====================================================
#===============================================================================

def load_gridvars(leafblocks, octree):
    import mf_fields as clfields
    
    octree = mflash.pm3dgrid(leafblocks)
    fields = clfields.get_fields(leafblocks, octree)
    dx_gpot, dy_gpot, dz_gpot = fields['g_gpot']
    
    absgrad_gpot = np.sqrt(dx_gpot**2 +dy_gpot**2 +dz_gpot**2)
    leafblocks.cache('absgrad_gpot', absgrad_gpot)
    
    # XXX MW 01.03.2022: Hack: Stitch hole in criterion
    dens = leafblocks['dens']
#    gmask = absgrad_gpot > 5e-8
#    if np.any(gmask):
#        densthr = np.percentile(dens[gmask], 50)
#        stitchmask = dens > densthr
#        aggps = np.copy(absgrad_gpot)
#        aggps[stitchmask] = np.min(absgrad_gpot[gmask])
#    leafblocks.cache('aggps', absgrad_gpot)

    # XXX MW 03.03.2022: Hack didn't work. Try continuous relaxation strategy
    from matplotlib.colors import LogNorm, Normalize
    gcrit_a = np.clip(Normalize(0., 5e-8)(absgrad_gpot), 0., np.inf)
    #gcrit_b = np.clip(LogNorm(1e-18, 1e-16)(dens), 0., np.inf)
    gcrit_b = np.clip(LogNorm(1e-19, 1e-16)(dens), 0., np.inf)
    gcritsq = np.sqrt(gcrit_a**2 +gcrit_b**2)
    gcritl = gcrit_a +gcrit_b
    leafblocks.cache('gcritl', gcritl)
    leafblocks.cache('gcritsq', gcritsq)
    return None


#===============================================================================
# ==== LOAD/SAVE DATA MANAGEMENT ===============================================
#===============================================================================

def get_criterion_fingerprint(criterion):
    crit_repr = str(select_criteria_combos[criterion])
    crit_hex = format(hash27(crit_repr), 'x')
    crit_fp = criterion +'_' +crit_hex
    return crit_fp
    
def get_clist_path(hdfdata, criterion):
    crit_fp  = get_criterion_fingerprint(criterion)
    hdf_fp  = tools.get_plotfile_fingerprint(hdfdata)
    prefix = clist_basename +crit_fp +'_' +hdf_fp +'_'
    filepath = tools.get_datafile_path(hdfdata, prefix)
    return filepath
    
def determine_clist(hdfdata, octree, criterion):
    ''' Compile a cluster list according to the selected criterion '''
    select_criteria = select_criteria_combos[criterion]
    print(f'Criterion: {criterion}')
    print(f'Requires gridvars: {criterion in require_gridvars}')
    if criterion in require_gridvars:
        load_gridvars(hdfdata, octree)
    if '_fw2fs' in criterion:
        print('-> Using Fellwalker algorithm on fused N-o-N preseletion')
        pre_criterion = select_criteria['preselect']
        pre_clist = get_clist(hdfdata, octree, pre_criterion)
        clist = fellwalk.fellwalk_clist_fused(pre_clist, hdfdata, octree, **select_criteria)        
    elif '_fw2s' in criterion:
        print('-> Using Fellwalker algorithm on N-o-N preseletion')
        pre_criterion = select_criteria['preselect']
        pre_clist = get_clist(hdfdata, octree, pre_criterion)
        clist = fellwalk.fellwalk_clist(pre_clist, hdfdata, octree, **select_criteria)
    elif '_proj' in criterion:
        import cl_project_007 as proj ### TODO: PORT
        print('-> Using 2D-Fellwalker deprojection algorithm')
        clist = proj.findobj(hdfdata, octree, **select_criteria)
    elif '_fw' in criterion:
        print('-> Using Fellwalker algorithm')
        clist = fellwalk.fellwalk(hdfdata, octree, **select_criteria)
    elif '_adv' in criterion:
        print('Using 1-connected neighbors of neighbors sampling algorithm')
        clist = fof.findcluster_adv(hdfdata, octree, **select_criteria)
    elif '_stitch' in criterion:
        print('-> Using stitched neighbors of neighbors sampling algorithm')
        base_criterion_key = select_criteria['base_criterion']
        base_criteria = select_criteria_combos[base_criterion_key]
        outer_criterion_key = select_criteria['outer_criterion']
        outer_criteria = select_criteria_combos[outer_criterion_key]
        pre_clist = fof.findcluster_direct(hdfdata, octree, base_criteria)
        clist = fof.StitchCluster(hdfdata, octree, pre_clist, outer_criteria)
    elif '_iso' in criterion:
        print('-> Using isosurface maximization algorithm')
        clist = fof.findcluster_isos(hdfdata, octree, **select_criteria)
        print(f'#DEBUG# len clist: {len(clist)}')
    elif ':isocut' in criterion:
        print('-> Using 2-stage isosurface cutout algorithm (stage 1)')
        l1crit  = criterion.split(':')[0]
        l1clist = get_clist(hdfdata, octree, l1crit)
        print('-> Using 2-stage isosurface cutout algorithm (stage 2)')
        clist   = fof.padcluster_isos(hdfdata, octree, l1clist, **select_criteria)
    elif ':dendroleaf' in criterion:
        print('-> Using 2-stage dendrogram leaf algorithm (stage 1)')
        l1crit  = criterion.split(':')[0]
        l1clist = get_clist(hdfdata, octree, l1crit)
        print('-> Using 2-stage dendrogram leaf algorithm (stage 2)')
        clist   = ddg.isolate_leafs(hdfdata, octree, l1clist, **select_criteria)
    else:
        print('-> Using neighbors of neighbors sampling algorithm')
        clist = fof.findcluster_direct(hdfdata, octree, **select_criteria)
        print(f'#DEBUG# len clist: {len(clist)}')
        
    return clist

def prepare_clist(hdfdata, octree, criterion):
    ''' Compile and save a cluster list according to criterion if not existing yet '''
    filepath = get_clist_path(hdfdata, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        print(f'-> Cluster list already exists: {filepath}')
        sys.stdout.flush()
    else:
        clist = determine_clist(hdfdata, octree, criterion)
        tools.save_data(clist, filepath, force=False)
    return filepath

def get_clist(hdfdata, octree, criterion):
    ''' Obtain a cluster list according to criterion; compile if necessary '''
    filepath = get_clist_path(hdfdata, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading cluster list ({criterion})')
        sys.stdout.flush()
        try:
            clist = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load cluster list !!')
            force = True
        else:
            return clist
    clist = determine_clist(hdfdata, octree, criterion)
    print(f'-> Saving cluster list ({criterion})')
    sys.stdout.flush()
    try:
        tools.save_data(clist, filepath, force=force)
    except:
        print('!! Warning: Could not save cluster data !!')
    return clist


#===============================================================================
# ==== DATA ACQUISITION LOOP ===================================================
#===============================================================================

def make_clists(dirpath, dirfiles, criterion):
    for filename in dirfiles:
        plotfile = os.path.join(dirpath, filename)
        print(f'>> Processing {plotfile}')
        sys.stdout.flush()
        # Open plotfile data
        hdfdata = mflash.plotfile(plotfile, memorize=False)
        hdfdata.learn(mflash.var_mhd)
        hdfdata.learn(mflash.var_grid)
        hdfdata.learn(var_ch)
        # Get octree from plotfile
        octree = mflash.pm3dgrid(hdfdata)
        # Prepare cluster list if not yet done
        prepare_clist(hdfdata, octree, criterion)
        # Close plotfile
        hdfdata.close()
    return 0


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def main(simpath, criterion):
    dirdict = tools.scanpath(simpath, plotfile_pattern)
    for dirpath in dirdict:
        print(f'-> Walking {dirpath}')
        sys.stdout.flush()
        dirfiles = dirdict[dirpath]
        make_clists(dirpath, dirfiles, criterion)
    return 0
        
if __name__ == '__main__':
    simpath = sys.argv[1]
    criterion = sys.argv[2]
    main(simpath, criterion)
    print('FINISHED!')
    
