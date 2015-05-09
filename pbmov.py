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
    Renders a movie
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

def camera_pass3(sim, cam0, target, vmin=None, vmax=None, camera_rot=0.0,nt=50,\
 b=1, res=500, fps=25, savename='movie.mp4'):
    """
    A simple implementation of having a camera pass by a target.  Not fully
    functional or robust...just proof of concept
    """
    target_r = np.sqrt((target**2).sum())
    cam_middle = target - b*(target/target_r)
    cameras = np.zeros([nt, 3])
    t = np.linspace(0,1,nt)
    
    for i in range(3):
        
        cameras[:,i] = (cam_middle[i] - cam0[i])*(2*t-1)**3 + cam_middle[i]
        
    camera_rots = np.linspace(0, camera_rot, nt)
    
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(savename, (res,res), fps)
    pos0 = sim['pos'].copy()
    
    for i in range(nt):
        
        print '\n{} of {}\n'.format(i+1, nt)
        width = pbmov_utils.frame_width(cameras[i], target)
        pos = pbmov_utils.vsm_transform(pos0, cameras[i], target, camera_rots[i])
        sim['pos'] = pos
        d = np.sqrt( ((cameras[i] - target)**2).sum())
        im = pb.plot.sph.image(sim, width=width, z_camera=d, noplot=True, resolution=res)
        color_im = rgbify(im, vmin, vmax)
        video_writer.write_frame(color_im)
        
    video_writer.close()
    sim['pos'] = pos0

def camera_pass2(sim, cameras, target, vmin=None, vmax=None, camera_rot=0.0,\
 res=500, fps=25, savename='movie.mp4', preview=None, nskip=0):
    """
    A simple implementation of having a camera pass by a target.  Not fully
    functional or robust...just proof of concept
    """
    nt = len(cameras)
    
    camera_rots = np.linspace(0, camera_rot, nt)
    pos0 = sim['pos'].copy()
    
    irange = range(nt)
    
    if nskip != 0:
        
        irange = irange[0::nskip]
        fps = max(int(fps/float(nskip)+0.5), 1)
    
    if preview is None:
        
        video_writer = ffmpeg_writer.FFMPEG_VideoWriter(savename, (res,res), fps)
        
    else:
        
        irange = range(preview, preview+1)
    
    for i in irange:
        
        print '\n{} of {}\n'.format(i+1, nt)
        width = pbmov_utils.frame_width(cameras[i], target)
        pos = pbmov_utils.vsm_transform(pos0, cameras[i], target, camera_rots[i])
        sim['pos'] = pos
        d = np.sqrt( ((cameras[i] - target)**2).sum())
        im = pb.plot.sph.image(sim, width=width, z_camera=d, noplot=True, resolution=res)
        color_im = rgbify(im, vmin, vmax)
        
        if preview is None:
            
            video_writer.write_frame(color_im)
            
        else:
            
            plt.imshow(color_im)
        
    if preview is None:
        
        video_writer.close()
        
    sim['pos'] = pos0
    
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

def rotation_movie(sim, vmin, vmax, width, rotations, savename='movie.mp4', \
cmap=cx_default, rot_frames=50, res=500, fps=10, codec="libx264", preset="slow",\
quality=18, **kwargs):
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
    cmap : colormap or str
        Colormap to use
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
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(savename, (res,res), fps, codec, preset=preset, quality=quality)
    
    for i, rot in enumerate(rot_per_frame):
        
        for j in range(rot_frames):
            
            print i, j
            
            rotate_sim(sim, rot)
            im = sph.image(sim, width=width, resolution=res, noplot=True, **kwargs)
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
        
    return im