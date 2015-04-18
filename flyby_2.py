import time
from matplotlib.colors import LogNorm
import numpy as np
import logging

import pynbody as pb
SimArray = pb.array.SimArray
import pynbody.plot.sph as sph

import cubehelix #Jim Davenport's color schemes, better for science than default color schemes
import isaac
import ffmpeg_writer

# setup colormaps
ch=cubehelix.cmap()
cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)  #mostly reds
cx3 = cubehelix.cmap(reverse=False, start=0.3, rot=-0.5)# mostly blues
cx_default = cubehelix.cmap(reverse=False, start=0., rot=0.5)  #mostly reds

    
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
    color_im = cx4(im)
    
    # Convert to 0-255 integer values
    color_im_int = (255*color_im[:,:,0:-1] + 0.5).astype(np.uint8)
    
    return color_im_int
    
def rotate_sim(sim,angles):
    """
    Rotates a snapshot by angles, where angles defines the amount in degrees
    to rotate around the (x,y,z) axes.
    
    The order of rotations is y,z,x
    
    **ARGUMENTS**
    
    sim : TipsySnapshot
        Simulation to rotate
    angles : array, tuple, list
        Angles to rotate by around the (x,y,z) axes
        
    **RETURNS**
    
    Nothing.  Transformation is done in place
    """
    
    sim.rotate_y(angles[1])
    sim.rotate_z(angles[2])
    sim.rotate_x(angles[0])
    
    return

def rotate_movie(sim, vmin, vmax, width, rotations, cmap=cx_default, \
rot_frames=50, res=500, fps=10, codec="libx264", preset="slow",quality=18, \
**kwargs):
    """
    Renders a movie of rotations around a simulation.  by default, renders
    'rho'
    Frames are rendered using pynbody.plot.sph.image
    
    **ARGUMENTS**
    
    sim : Tipsy Snapshot subsnap
        A sub-snap to be rendered, ie the dark matter or the gas from a halo
    vmin, vmax : float or SimArray
        Color scale normalization limits
    width : 
        Width of the simulation to render, ie '62 au' or '300 kpc'        
    rotations : list or numpy array
        A set of rotations around the (x,y,z) axes to perform.  To rotate
        180 deg around x then 360 around y,z do:
        rotations = [[180,0,0], [0,360,360]]
    rot_frames : int
        Number of frames per rotation to render        
    res : int
        Resolution in pixels.
    fps : int
        Frames per second of the movie
    codec : str
        movie codec to use
    preset : str
        ffmpeg rendering setting.  Options are: ultrafast,superfast,
        veryfast, faster, fast, medium (default), slow, slower, veryslow,
        placebo.
    quality : int
        quality on a logarithmic scale (for libx624 and mpeg4 codecs)
        0 = lossless.  (see ffmpeg documentation)
    
    **kwargs
        Additional kwargs to get passed to pynbody.plot.sph.image
    """
    
    if isinstance(rotations, list):
        
        rotations = np.array(rotations, dtype=float)
        
    n_rot = rotations.shape[0] # number of rotations
    rot_per_frame = rotations / rot_frames # How much to rotation per frame
    
    # Toggle verbosity off
    if hasattr(pb, 'logger'):
        # As of v0.30, pynbody uses python's logging to handle verbosity
        logger_level = pb.logger.getEffectiveLevel()
        pb.logger.setLevel(logging.ERROR)
        
    else:
        
        verbosity = pb.config['verbose']
        pb.config['verbose'] = False
    
    ims = []
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(save_path, (res,res), fps, codec, preset=preset, quality=quality)
    
    for i, rot in enumerate(rot_per_frame):
        
        for j in range(rot_frames):
            
            print i, j
            
            rotate_sim(sim, rot)
            im = sph.image(sim, width=zoom, resolution=res, noplot=True, **kwargs)
            color_im = rgbify(im, vmin=vmin, vmax=vmax)
            video_writer.write_frame(color_im)
            
            ims.append(color_im)
    
    video_writer.close()
    im = np.array(ims, dtype=np.uint8)  
    
    # Toggle verbosity back
    if hasattr(pb, 'logger'):
        # As of v0.30, pynbody uses python's logging to handle verbosity
        pb.logger.setLevel(logger_level)
        
    else:
        
        pb.config['verbose'] = verbosity
        
    return
    
    
#set the sim path to the simulation you want to see
sim_path = 'wengen_test_final.std'
param_path = 'wengen_test_final.param'
save_path = 'tester.mp4'

# Load the simulation
sim = pb.load(sim_path, paramname=param_path)
isaac.snapshot_defaults(sim)


#The colorbar min/max need to be set:
vmin=1e-9
vmax=1e0

zoom= SimArray(62.0, 'au')#the camera distance in kpc for each sim

# Video settings
res = 500
fps = 10
codec = "libx264"
preset = "slow"
quality = 18

# Set the rotations to do
rotations = [[0, 0, 360]]
rot_frames = 50  #the number of frames to rotate for each index in rotations_x/y/z


