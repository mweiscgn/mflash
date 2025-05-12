import sys, os
import numpy as np

import cl_fofg_142 as fof
import libmf.tools as tools

from copy import deepcopy
from astrodendro import Dendrogram


#===============================================================================
# ==== INDEXING HELPER =========================================================
#===============================================================================

def IndexAmrToUniform(octree, bbox_extent):
    #print('Determining uniform grid indicies...')
    #sys.stdout.flush()
    # Get center coordinates of uniform grid around inside given extent
    grating = [octree.grating(d, extent=bbox_extent[d], align=True) for d in range(3)]
    coords_raw = np.meshgrid(*grating, indexing='ij')
    # Unwind extent window meshgrid to main period of domain
    Xmg, Ymg, Zmg = BG_UnwindPeriodic(octree, coords_raw)
    # Find cell indicies for given uniformly spaced coordinate grid
    bl, (k, j, i) = octree.findcell(Xmg, Ymg, Zmg)
    print('Uniform grid shape:', bl.shape)
    return (bl,k,j,i)

def MaskUniformBoxToAmr(amrshape, Iuniform):
    mask = np.zeros(amrshape, dtype=bool)
    mask[Iuniform] = True
    return mask

def MaskSubToAmr(amrshape, Iuniform, submask):
    mask = np.zeros(amrshape, dtype=bool)
    unfmask = mask[Iuniform]
    unfmask[submask] = True
    mask[Iuniform] = unfmask
    return mask

def RadiusExtent(center, radius):
    x_extent = (center-2.*radius)[0], (center+2.*radius)[0]
    y_extent = (center-2.*radius)[1], (center+2.*radius)[1]
    z_extent = (center-2.*radius)[2], (center+2.*radius)[2]
    return x_extent, y_extent, z_extent


#===============================================================================
# ==== OBJECT BULK TRANSFORMATIONS =============================================
#===============================================================================

def BG_UnwindPeriodic(octree, coords_raw):
    x_raw, y_raw, z_raw = coords_raw
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_unw = tools.modclip(x_raw, xmin, xmax)
    y_unw = tools.modclip(y_raw, ymin, ymax)
    z_unw = tools.modclip(z_raw, zmin, zmax)
    return x_unw, y_unw, z_unw

def BG_RelCoords(octree, center):
    ''' Determine radial vector from center to cells,
    while rotating the coordinate system window to center the center '''
    coords = octree.coords()
    x_raw, y_raw, z_raw = [coords[:,i,:,:,:] for i in range(3)]
    x_ctr, y_ctr, z_ctr = center
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
    x_rel = tools.modclip(x_raw -x_ctr, -.5*xsize, +.5*xsize)
    y_rel = tools.modclip(y_raw -y_ctr, -.5*ysize, +.5*ysize)
    z_rel = tools.modclip(z_raw -z_ctr, -.5*zsize, +.5*zsize)
    return x_rel, y_rel, z_rel

def ObjCenter(hdfdata, octree, objwhere):
    # Get object cell raw positions
    coords = octree.coords()
    posx_raw = coords[:,0,:,:,:][objwhere]
    posy_raw = coords[:,1,:,:,:][objwhere]
    posz_raw = coords[:,2,:,:,:][objwhere]
    # Rotate the coordinate system window to center one of the object
    # grid points (i.e. select the first one) according to per. bounds.
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
    posx = tools.modclip(posx_raw, posx_raw[0]-.5*xsize, posx_raw[0]+.5*xsize)
    posy = tools.modclip(posy_raw, posy_raw[0]-.5*ysize, posy_raw[0]+.5*ysize)
    posz = tools.modclip(posz_raw, posz_raw[0]-.5*zsize, posz_raw[0]+.5*zsize)
    cellpos = (posx,posy,posz)
    # Calculate center of mass using rotated grid window coordinates
    mass = hdfdata['mass'][objwhere]
    center = [np.average(cellpos[i], weights=mass) for i in range(3)]
    center_unw = BG_UnwindPeriodic(octree, center)
    return np.array(center_unw)

def ObjRadius(hdfdata, octree, objwhere, objcenter):
    relcoords = BG_RelCoords(octree, objcenter)
    x, y, z = [relcoords[i][objwhere] for i in range(3)] 
    rsq = x**2 + y**2 + z**2
    radius = np.sqrt(np.max(rsq))
    return radius


# ==============================================================================
# ==== DENDROGRAM SUBSTRUCTURE DETECTION =======================================
# ==============================================================================

formulae = {
    'dens_log': ('dens', np.log10),
    'l_jeans_recp': (1.0, 'l_jeans', '/'),
    'l_jeans_recp_log': (1.0, 'l_jeans', '/', np.log10)
}

def Dendroleafs(leafdata, octree, mask, l1clist, var='dens_log', **ddgargs):
    leafobjs = list()
    if not l1clist:
        return leafobjs
    data_amr = leafdata[var]
    baseshape = data_amr.shape
    for iobj, objl in enumerate(l1clist):
        print(f'...Processing L1-object {iobj+1}/{len(l1clist)}...')
        objmask = fof.GetMask(baseshape, objl)
        objpos = ObjCenter(leafdata, octree, objmask)
        objradius = ObjRadius(leafdata, octree, objmask, objpos)
        objextent = RadiusExtent(objpos, objradius)
        Iunf = IndexAmrToUniform(octree, objextent)
        data_unf = np.copy(data_amr[Iunf])
        objmask *= mask
        ambientmask_unf = np.logical_not(objmask[Iunf])
        data_unf[ambientmask_unf] = np.min(data_amr)
        dendro = Dendrogram.compute(data_unf, **ddgargs)
        print(f'Saving {len(dendro.leaves)} dendrogram leafs...')
        for il,leaf in enumerate(dendro.leaves):
            leafmask_unf = leaf.get_mask()
            leafmask_amr = MaskSubToAmr(baseshape, Iunf, leafmask_unf)
            leafwhere = np.where(leafmask_amr)
            clist = fof.lzip(*leafwhere)
            leafobjs.append(clist)
        # Clean up dendrogram data
        for key in dendro._structures_dict:
            dendro._structures_dict[key]._tree_index = None
            dendro._structures_dict[key] = None
        dendro._structures_dict.clear()
        dendro.index_map = None
        dendro.data = None
        dendro = None
    print(f'!  Detected {len(leafobjs)} leaf objects in {len(l1clist)} L1-objects')
    return deepcopy(leafobjs)


def isolate_leafs(leafdata, octree, l1clist, **param):
    leafdata.learn(formulae)
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria')
    mask = fof.mask_cells(leafdata, cell_criteria)
    #print(f'#DEBUG# Selected {np.sum(mask)} cells')
    #print('-> Detecting Substructure Dendrogram Leafs...')
    #print('Parameters:')
    #print(param)
    sys.stdout.flush()
    object_list = Dendroleafs(leafdata, octree, mask, l1clist, **param)
    return object_list
