import logging

from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np

import pynbody as pb
SimArray = pb.array.SimArray
import pynbody.plot.sph as sph

import cubehelix #Jim Davenport's color schemes, better for science than default color schemes
import ffmpeg_writer
import pbmov_utils

# setup colormaps
ch=cubehelix.cmap()
cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)  #mostly reds
cx3 = cubehelix.cmap(reverse=False, start=0.3, rot=-0.5)# mostly blues
cx_default = ch

def render_movie(sim, cameras, targets, nt, vmin=None, vmax=None, camera_rot=0.0,\
 res=500, cmap=cx_default, fps=25, savename='movie.mp4', preview=None, nskip=0, **kwargs):
    """
    Renders a movie.
    
    **STATIC ARGUMENTS**
    sim : snapshot
        A pynbody snapshot
    nt : int
        Numer of frames
    res : int
        Frame resolution.  All frames are square (ie, resolution = res x res)
    cmap : matplotlib colormap or str
        Colormap to use
    fps : int
        Frames per second
    savename : str
        Filename to save movie to
    preview : None or int
        If int, then frame number preview is plotted in a window instead of
        rendering the whole movie
        If None (DEFAULT), movie is rendered
    nskip : int
        (buggy still) Number of frames to skip.  Can be used to render every
        nth frame
    **kwargs
        Additional keyword arguments to pass to the pynbody renderer
        (see pynbody.plot.sph.image)
    
    **ANIMATEABLE ARGUMENTS**
    NOTE: All the animateable arguments can be passed as static if wanted.
    e.g., if ONE camera position is passed, then the camera will be treated
    as static.
    IF a value is to be animated, it should be provided for every frame
    
    cameras : array
        Camera x,y,z position(s).  shape (nt, 3) if animated
    targets : array
        Target x,y,z position(s).  shape (nt, 3) if animated
    vmin, vmax : float or array
        min/max values for color map
    camera_rot : float or array
        Rotation of the camera (in radians) around the axis connecting camera
        and the target.
    """
    # Make multiples of animated values if needed.  This ensures the values
    # are all defined at every frame
    cameras     = repeat_val(cameras, nt, 1)
    targets     = repeat_val(targets, nt, 1)
    vmin        = repeat_val(vmin, nt, 0)
    vmax        = repeat_val(vmax, nt, 0)
    camera_rot  = repeat_val(camera_rot, nt, 0)
    
    if preview is not None:
        
        # Just preview a frame and end
        i = preview
        
        render_frame(sim, cameras[i], targets[i], vmin=vmin[i], vmax=vmax[i], \
        camera_rot=camera_rot[i], res=500, cmap=cmap, preview=True, **kwargs)
        
        return
        
    # we'll be making a video.  Initialize a video_writer object
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(savename, (res,res), fps)
    # Copy initial positions
    pos0 =  sim['pos'].copy()
    
    # setup frames to render
    irange = range(nt)
    if nskip != 0:
        
        irange = irange[0::nskip]
        fps = max(int(fps/float(nskip)+0.5), 1)
    
    # Loop through frames and render
    for i in irange:
        
        print '\n{} of {}\n'.format(i+1, nt)
        # render frame
        color_im = render_frame(sim, cameras[i], targets[i], pos0=pos0,\
        vmin=vmin[i], vmax=vmax[i], camera_rot=camera_rot[i], res=res, cmap=cmap,\
        revert_sim_pos=False, **kwargs)
        # write to video
        video_writer.write_frame(color_im)
        
    # Finalize
    video_writer.close()
    sim['pos'] = pos0
            
def render_frame(sim, camera, target=None, pos0=None, vmin=None, vmax=None, camera_rot=0.0, \
res=500, cmap=cx_default, preview=False, revert_sim_pos=True, **kwargs):
    """
    Renders a frame of a movie and returns it as an RGB array.  Optionally,
    if preview is True, it will plot the image instead of returning it.
    
    **ARGUMENTS**
    
    sim : snapshot
        pynbody snapshot to render.  IF a family-level quantity is being plotted,
        e.g. 'rho', then just that family should be given.  ie:
            snapshot = pynbody.load('filename')
            sim = snapshot.gas
    camera : array
        x,y,z position of the camera
    target : array or NONE
        x,y,z position of target.  If None, the origin is assumed
    pos0 : array
        Initial particle positions.  useful if the particle positions in 
        sim were altered by a previously rendered frame
    vmin, vmax : float
        min/max values for color map
    camera_rot : float
        rotation of the camera around the axis connecting the camera and
        the target (in radians)
    res : int
        Resolution (in pixels) for the square image.
    cmap : str or matplotlib cmap
        Colormap
    preview : bool
        If True, the image is plotted and nothing is returned
        If False (default) the image is rendered and returned without plotting
    revert_sim_pos : bool
        If True (default) the particle positions in sim are reverted to their
        original values.  Requires copying the particle positions
        If False, the particle positions in the simulation will be changed
    **kwargs
        Additional keyword arguments to pass to pynbody.plot.sph.image
        
    **RETURNS**
    
    im : array
        If preview=False, an RGB array of the image is returned.  Otherwise,
        nothing is returned
    """
    if revert_sim_pos or preview:
        
        original_pos = sim['pos'].copy()
        
    if pos0 is None:
        
        pos0 = sim['pos']
        
    if target is None:
        
        target = np.zeros(3)
        
    width = pbmov_utils.frame_width(camera, target)
    d = np.sqrt( ((camera - target)**2).sum())
    pos = pbmov_utils.vsm_transform(pos0, camera, target, camera_rot)
    sim['pos'] = pos
    
    if preview:
        
        # Plot
        pb.plot.sph.image(sim, width=width, z_camera=d, noplot=False, \
        resolution=res, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs);
        
        # Revert simulation position back to original
        sim['pos'] = original_pos
        color_im = None
        
    else:
        
        # Render image in pynbody
        im = pb.plot.sph.image(sim, width=width, z_camera=d, noplot=True, \
        resolution=res, **kwargs)
        # Now make a color, RGB image
        color_im = rgbify(im, vmin, vmax, cmap)
        return color_im
    
def repeat_val(x, nt, single_val_dim=0):
    """
    Repeats x for every frame if x is constant.  If there are multiple values
    of x, x is returned (unchanged)
    
    **ARGUMENTS**
    
    x : array or number
        array/number to check
    nt : int
        number of time steps
    single_val_dim : int
        Number of dimensions a single value would have.  0 for a float, 1
        for a 1D array, 2 for a 2D array, etc...
    
    **RETURNS**
    
    x : array
        array which is just repeated values of x if x is constant OR x (unchanged)
        if x has multiple values
    """
    # Check that values has more more dimensions than a single value would
    ndim = np.ndim(x)
    if ndim - single_val_dim > 1:
        
        raise ValueError, 'x has too many dimensions. At most it can have single_val_dim + 1'
        
    if np.ndim(x) > single_val_dim:
        
        # Check that there is more than 1 value
        nx = len(x)
        if nx > 1:
            # There are multiple values.  Make sure there are nt of them
            if nx != nt:
                
                raise ValueError, 'x has multiple entries, but not nt of them'
                
        else:
            
            # x has only a single value but an extra dimension.
            x = x[0]
            
    else:
         # There's only one entry
        nx = 1
    
    if nx == 1:
        # if there's only one entry
        # Copy x, nt times
        x_list = []
        for i in range(nt):
            
            x_list.append(x)
            
        # make an array
        x = np.array(x_list)
    
    return x
    
def rgbify(im, vmin=None, vmax=None, cmap=cx_default):
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
    a = LogNorm(vmin=vmin, vmax=vmax)
    im = a(im)
    
    # Apply color map
    color_im = cmap(im)
    
    # Convert to 0-255 integer values
    color_im_int = (255*color_im[:,:,0:-1] + 0.5).astype(np.uint8)
    
    return color_im_int
    
