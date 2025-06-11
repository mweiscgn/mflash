# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools

from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from var_settings_v3 import GetVarSettings, intvar

__author__ = "Michael Weis"
__version__ = "0.1.0.0"

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# Constants:
from constants import *
version_str = 'varscatter010c'
debug = False

# Flash settings:
var_ch = mflash.var_ch5
plotfile_pattern = '*hdf5_plt_cnt_????'

# Figure settings:
fs = mplot.set_fig_preset('display')
mplot.fig_outdir = f'sim/{version_str}-{mplot.fig_target}'

# Histogram parameters:
histo_res = 201


#===============================================================================
# ==== HELPER ==================================================================
#===============================================================================

def LoadPlotfile(dirpath, filename, verbose=False):
    plotfile = os.path.join(dirpath, filename)
    if verbose > 0:
        print(f'\n>> Processing {plotfile}')
    assert os.path.isfile(plotfile)
    # Open plotfile data
    leafdata = mflash.plotfile(plotfile, memorize=False)
    leafdata.learn(mflash.var_mhd)
    leafdata.learn(mflash.var_grid)
    leafdata.learn(var_ch)
    # Get octree from plotfile
    octree = mflash.pm3dgrid(leafdata)
    return leafdata, octree

def LoadVar(hdfdata, key_in, octree=None):
    var = key_in.split('::')[0]
    if var[:5] == 'fftc_':
        import mf_fftconv as fftc
        if octree is None:
            octree = mflash.pm3dgrid(hdfdata)
        datakey = var[5:]
        var_fftc = fftc.get_fftc(hdfdata, octree, datakey)
        hdfdata.cache(var, var_fftc)
        return True
    elif var == 'lEq': # Local kinetic / magnetic energy ratio
        LoadVar(hdfdata, 'fftc_velp', octree)
        LoadVar(hdfdata, 'fftc_magp', octree)
        lEq = hdfdata['fftc_velp'] / hdfdata['fftc_magp']
        hdfdata.cache('lEq', lEq)

    elif var in ['lvdi', 'lvdi_sq', 'lcdi']:
        import localdisp
        if octree is None:
            octree = mflash.pm3dgrid(hdfdata)
        lvdi = localdisp.get_lvdi(hdfdata, octree)
        hdfdata.cache('lvdi', lvdi)
        hdfdata.cache('lvdi_sq', lvdi**2)
        if var in ['lcdi',]:
            c_s = hdfdata['c_s']
            hdfdata.cache('lcdi', lvdi/c_s)
        return True
    elif var in ['xcoord', 'ycoord', 'zcoord', 'coords']:
        octree = mflash.pm3dgrid(hdfdata)
        coords = octree.coords()
        hdfdata.cache('xcoord', coords[:,0,:,:,:])
        hdfdata.cache('ycoord', coords[:,1,:,:,:])
        hdfdata.cache('zcoord', coords[:,2,:,:,:])
        return True
    return False


#===============================================================================
# ==== Scatter Plot Code  ======================================================
#===============================================================================

def _VSBinData(leafdata, varkey):
    units, norm, cmap, label, ulabel, vartitle, weighvar = GetVarSettings(varkey)
    data = leafdata[varkey] / units
    varname = varkey.split('::')[0]
    res = int(norm.vmax-norm.vmin) if varname in intvar else histo_res
    partition = norm.inverse(np.linspace(0, 1, res+1))
    bins = np.searchsorted(partition, data.ravel())
    return bins, res

def _varlabel(symbol, ulabel):
    if ulabel:
        return f'{symbol} [{ulabel}]'
    else:
        return symbol
    
def _barlabel(x_var, y_var, w_var, normalize=True, **kwargs):
    # Get settings for the given axis variables
    x_units, x_norm, x_cmap, x_label, x_ulabel, x_vartitle, x_weighvar = GetVarSettings(x_var)
    y_units, y_norm, y_cmap, y_label, y_ulabel, y_vartitle, y_weighvar = GetVarSettings(y_var)
    w_units, w_norm, w_cmap, w_label, w_ulabel, w_vartitle, w_weighvar = GetVarSettings(w_var)
    if normalize is True:
        x_islog = np.isclose(x_norm(np.sqrt(x_norm.vmin*x_norm.vmax)), .5)
        y_islog = np.isclose(y_norm(np.sqrt(y_norm.vmin*y_norm.vmax)), .5)
        x_islin = np.isclose(x_norm(.5*(x_norm.vmin+x_norm.vmax)), .5)
        y_islin = np.isclose(y_norm(.5*(y_norm.vmin+y_norm.vmax)), .5)
        if x_islog: 
            x_bardenom_label = '$\mathrm{d}\,\log$(%s)'%x_label
            x_bardenom_ulabel = ''
        else:
            x_bardenom_label = '$\mathrm{d}\,$%s'%x_label
            x_bardenom_ulabel = x_ulabel
        if y_islog: 
            y_bardenom_label = '$\mathrm{d}\,\log$(%s)'%y_label
            y_bardenom_ulabel = ''
        else:
            y_bardenom_label = '$\mathrm{d}\,$%s'%y_label
            y_bardenom_ulabel = y_ulabel
        #
        barnom_label = '$\mathrm{d}^{2}\,$%s'%w_label
        barnom_ulabel = w_ulabel
        #
        bar_symlabel = f'{barnom_label} / ({x_bardenom_label}$\,${y_bardenom_label})'
        #
        if not (x_bardenom_ulabel or y_bardenom_ulabel or barnom_ulabel):
            barlabel = bar_symlabel
        else:
            if not barnom_ulabel: barnom_ulabel = '1'
            if not (x_bardenom_ulabel or y_bardenom_ulabel):
                bar_ulabel = f'{barnom_ulabel} '
            else:
                bar_ulabel = f'{barnom_ulabel}$\,/\,$({x_bardenom_ulabel}$\,${y_bardenom_ulabel})'
            barlabel = f'{bar_symlabel} [{bar_ulabel}]'
    elif normalize is False:
        barlabel = f'Incidence ({w_vartitle})'
    else:
        barlabel = f'{w_vartitle} / pixel'
    return barlabel


def axVarScatter(ax, leafdata, x_var, y_var, weighting='vol', weights=None, normalize=True, wspan=1e6,
                 debug=False, show_cbar=True, cmap=None):
    # Get settings for the given axis variables
    x_units, x_norm, x_cmap, x_label, x_ulabel, x_vartitle, x_weighvar = GetVarSettings(x_var)
    y_units, y_norm, y_cmap, y_label, y_ulabel, y_vartitle, y_weighvar = GetVarSettings(y_var)
    w_units, w_norm, w_cmap, w_label, w_ulabel, w_vartitle, w_weighvar = GetVarSettings(weighting)
    # Bin data
    x_bins, x_res = _VSBinData(leafdata, x_var)
    y_bins, y_res = _VSBinData(leafdata, y_var)
    # Determine the weighting scheme for the 2d-heatmap
    w_weights = leafdata[weighting] / w_units
    if weights is None:
        weights = np.ones_like(w_weights)     
    hm_weights = (w_weights*weights).ravel()
    # Normalize heatmap weights
    if normalize is True:
        x_islog = np.isclose(x_norm(np.sqrt(x_norm.vmin*x_norm.vmax)), .5)
        y_islog = np.isclose(y_norm(np.sqrt(y_norm.vmin*y_norm.vmax)), .5)
        x_islin = np.isclose(x_norm(.5*(x_norm.vmin+x_norm.vmax)), .5)
        y_islin = np.isclose(y_norm(.5*(y_norm.vmin+y_norm.vmax)), .5)
        print(f'DEBUG: NORMALIZE:')
        print(f'    x_islog:{x_islog}, y_islog:{y_islog}, x_islin:{x_islin}, y_islin:{y_islin}')
        if x_islog: 
            x_uspan = np.log10(x_norm.vmax) - np.log10(x_norm.vmin)
            x_bardenom_label = '$\mathrm{d}\,\log(%s)$'%x_label
        elif x_islin:
            x_uspan = x_norm.vmax - x_norm.vmin
            x_bardenom_label = '$\mathrm{d}\,%s$'%x_label
        else:
            x_uspan = None
        if y_islog:
            y_uspan =  np.log10(y_norm.vmax) - np.log(y_norm.vmin)
            y_bardenom_label = '$\mathrm{d}\,\log(%s)$'%y_label
        elif y_islin:
            y_uspan = y_norm.vmax - y_norm.vmin
            y_bardenom_label = '$\mathrm{d}\,{%s}$'%y_label
        else:
            y_uspan = None
        if (x_uspan is None) or (y_uspan is None):
            wmax = np.sum(hm_weights)
        else:
            x_pixelsize_axunits = x_uspan / x_res
            y_pixelsize_axunits = y_uspan / y_res
            pxarea = x_pixelsize_axunits * y_pixelsize_axunits
            hm_weights /= pxarea
            wmax = np.sum(hm_weights) * np.cbrt(wspan)
    elif normalize is False:
        hm_weights /= hm_weights.sum()
        n_pixel = x_res*y_res
        assert n_pixel > 0
        hm_weights *= n_pixel
        wmax = .5*n_pixel
    else:
        wmax = np.sum(hm_weights)
    #
    if wmax <= 0.:
        wmax_old, wmax = wmax, 1.
        print(f'WARNING! [axVarScatter]: wmax={wmax_old}<=0. Reset to wmax={wmax}')
        print(f'         This might will yield an empty plot.')
    hm_norm = LogNorm(wmax/wspan, wmax, clip=True)
    # Compute weighted 2d-heatmap for the xy-binned data
    hm_loads = mplot.bincount2d(x_bins, y_bins, hm_weights, x_res+2, y_res+2)
    hm_data = np.clip(hm_loads[1:-1,1:-1], hm_norm.vmin, hm_norm.vmax)
    # Display the heatmap
    extent = [x_norm.vmin, x_norm.vmax, y_norm.vmin, y_norm.vmax]
    # Display the heatmap
    if cmap is None: cmap = w_cmap
    im = ax.imshow(hm_data.T, origin="lower", interpolation='none',
        extent=extent, aspect='auto', norm=hm_norm, cmap=cmap)
    # Format the axis ticks/labels to show them scaled to the correct norm
    # Note: While the bin-bounds are not necessarily spaced linearly,
    #       the bins itself are always supposed to be displayed equally sized.
    #       For all norms except Normalize, matplotlib has to be tricked to allow this.    
    # Format x-axis
    ax.set(xlim=extent[0:2], xlabel=_varlabel(x_label, x_ulabel))
    xticks, xtickpos = mplot.get_ticks(x_norm)
    ax.set_xticks(xtickpos)
    ax.set_xticklabels(xticks)
    # Format y-axis
    ax.set(ylim=extent[2:4], ylabel=_varlabel(y_label, y_ulabel))
    yticks, ytickpos = mplot.get_ticks(y_norm)
    ax.set_yticks(ytickpos)
    ax.set_yticklabels(yticks)
    # Add colorbar
    barlabel = _barlabel(x_var, y_var, weighting, normalize=normalize)
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    else:
        cbar = None
    #
    if debug:
        print('# Debug [axVarScatter]:')
        print(f'x-norm    : [{x_norm.vmin},{x_norm.vmax}]')
        print(f'y-norm    : [{y_norm.vmin},{y_norm.vmax}]')
        print(f'hm-norm   : [{hm_norm.vmin},{hm_norm.vmax}]')
        print(f'hm-val.   : [{np.nanmin(hm_data)},{np.nanmax(hm_data)}]')
        print(f'hm-weights: [{np.nanmin(hm_weights)},{np.nanmax(hm_weights)}]')
        print(f'w-weights : [{np.nanmin(w_weights)},{np.nanmax(w_weights)}]')
        print(f'weights   : [{np.nanmin(weights)},{np.nanmax(weights)}]')
        print()

    return im, cbar


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def Main(dirpath, dirfile, x_var, y_var, w_var, show=True):
    leafdata, octree = LoadPlotfile(dirpath, dirfile, verbose=debug)

    for var in [x_var,y_var,w_var]:
        LoadVar(leafdata, var)

    fig, ax = plt.subplots(1, 1)
    im1, cb1 = axVarScatter(ax, leafdata, x_var, y_var,
                            weighting=w_var, weights=None, normalize=True,
                            wspan=3.16e9, debug=debug, show_cbar=True)
    mplot.ax_title(ax, dirfile)
    if show:
        fig.set_size_inches(14, 12, True)
        fig.tight_layout()
        plt.show()
    else:
        filespec = f'scatter-{y_var}_vs_{x_var}-{w_var}'
        figname = f'{filespec}-{dirfile}'
        subpath = os.path.basename(dirpath)
        mplot.savefig(fig, figname, subpath)


#===============================================================================
# ==== LOBBY ===================================================================    
#===============================================================================

def getarg(i, default):
    try:
        return sys.argv[i]
    except IndexError:
        return default

if __name__ == '__main__':
    path = sys.argv[1]
    x_var = getarg(2, 'dens')
    y_var = getarg(3, 'pres')
    w_var = getarg(4, 'mass')

    if os.path.isfile(path):
        dirpath, dirfile = os.path.split(path)
        Main(dirpath, dirfile, x_var, y_var, w_var, show=True)
    elif os.path.isdir(path):
        dirdict = tools.scanpath(path, plotfile_pattern)
        for simdir,dirfiles in dirdict.items():
            for dirfile in dirfiles:
                Main(simdir, dirfile, x_var, y_var, w_var, show=False)
    else:
        raise RuntimeError(f'Path "{path}" not understood.')

    print('FINISHED!')
