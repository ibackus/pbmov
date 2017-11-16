# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:54:19 2015

@author: ibackus
"""
# External imports
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import get_cmap
import numpy as np
import pynbody as pb
SimArray = pb.array.SimArray

# Internal imports
import cubehelix
import pbmov_utils

# setup colormaps
ch=cubehelix.cmap()
cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)  #mostly reds
cx3 = cubehelix.cmap(reverse=False, start=0.3, rot=-0.5)# mostly blues
cx_default = ch

def renderFrame(frameSettings, params, sim, pos0=None, preview=False):
    """
    Renders a frame of a movie and returns it as an RGB array.  Optionally,
    if preview is True, it will plot the image instead of returning it.
    
    NOTE: when not previewing, the sim['pos'] will be altered by the view space
    transformation.
    
    Parameters
    ----------
    
    frameSettings : dict
        A dictionary containing settings/parameters pertinent only to the
        current frame of the movie (see movieSettings.getFrame)
    params : dict
        A dictionary containing settings for the whole movie
    sim : snapshot
        pynbody snapshot to render.  IF a family-level quantity is being plotted,
        e.g. 'rho', then just that family should be given.  ie:
            snapshot = pynbody.load('filename')
            sim = snapshot.gas
    pos0 : array
        Initial particle positions.  useful if the particle positions in 
        sim were altered by a previously rendered frame
    preview : bool
        If True, the image is plotted and nothing is returned
        If False (default) the image is rendered and returned without plotting
        
    Returns
    -------
    
    im : array
        If preview=False, an RGB array of the image is returned.  Otherwise,
        None is returned
    """
    # Load settings for this frame
    cam = frameSettings['cam']
    target = frameSettings['target']
    camera_rot = frameSettings['camOrient']
    vmax = frameSettings['vmax']
    vmin = frameSettings['vmin']
    simRot = frameSettings['simRot']
    simRotAxis = np.zeros(3)
    simRotAxis[0:2] = simRot[0:2]
    simRotAngle = simRot[2]
    
    # Load movie-wide (all frames) settings
    cmap = params['cmap']
    kwargs = params['pbkwargs']
    res = params['res']
    if kwargs is None:
        
        kwargs = {}
        
    # Get positions
    if preview:
        
        original_pos = sim['pos'].copy()
        
    if pos0 is None:
        
        pos0 = sim['pos']
        
    # Default target
    if target is None:
        
        target = np.zeros(3)
        
    # Get frame width (physical size)
    width = pbmov_utils.frame_width(cam, target)
    d = np.sqrt( ((cam - target)**2).sum())
    # Perform view space transformation on position
    pos = pbmov_utils.vsm_transform(pos0, cam, target, camera_rot*(np.pi/180))
    # perform an additional rotation of the simulation
    R = pbmov_utils.rotation_matrix(simRotAxis, simRotAngle*(np.pi/180), 
                                    fourD=False)
    pos = np.dot(R, pos.T).T
    sim['pos'] = pos
    
    if preview:
        
        # Plot
        pb.plot.sph.image(sim.g, width=width, z_camera=d, noplot=False, \
        resolution=res, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs);
        
        # Revert simulation position back to original
        sim['pos'] = original_pos
        color_im = None
        
    else:
        
        # Render image in pynbody
        im = pb.plot.sph.image(sim.g, width=width, z_camera=d, noplot=True, \
        resolution=res, **kwargs)
        # Now make a color, RGB image
        log = kwargs.get('log', True)
        color_im = rgbify(im, vmin, vmax, cmap, log)
        return color_im
        
def rgbify(im, vmin=None, vmax=None, cmap=cx_default, log=True):
    """
    Converts an image made by pynbody.plot.sph.image to an RGB array of ints,
    dtype uint8.  A logarithmic image will be produced
    
    **ARGUMENTS**
    
    im : SimArray or numpy array
        2D image array.  
    vmin, vmax : float or SimArray
        Normalization limits.  Values outside these bounds will be mapped to
        0, 1.
        
    **RETURNS**
    
    color_im_array : numpy array, dtype uint8
        Logarithmic, normalized RGB image of integers
    """
    if isinstance(cmap, str):
        
        cmap = get_cmap(cmap)
    
    im = im.copy()
    im[im==0] = np.nan    
    
    # Set up the min/max values
    if vmin is not None:
        
        # Filter min values
        im[im < vmin] = vmin
        
    else:
        
        vmin = np.nanmin(im)
        
    if vmax is not None:
        
        # Filter max values
        im[im > vmax] = vmax
        
    else:
        
        vmax = np.nanmax(im)
    
    # Assume all nans were zeros originally and replace them
    im[np.isnan(im)] = vmin
    
    # Run a log normalization
    if log:
        a = LogNorm(vmin=vmin, vmax=vmax)
    else:
        a = Normalize(vmin=vmin, vmax=vmax)
    try:
        im = a(im)
    except ValueError:
        print vmin, vmax
        raise
    
    # Apply color map
    color_im = cmap(im)
    
    # Convert to 0-255 integer values
    color_im_int = (255*color_im[:,:,0:-1] + 0.5).astype(np.uint8)
    
    return color_im_int