# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 07:02:50 2015

@author: ibackus
"""
import numpy as np
import logging

from diskpy.utils import pbverbosity

import pynbody as pb
SimArray = pb.array.SimArray

def pbload(filename, paramname=None):
    """
    Loads a snapshot using pynbody.  Can load a single species by appending
    ::gas, ::star, or ::dm to the filename
    
    Parameters
    ----------
    
    filename : str
        Filename to load
    paramname : str
        (optional) .param file to use
    
    Returns
    -------
    
    sim : snapshot
        A pynbody snapshot
    """
    if '::' in filename:
        
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
        
    else:
        
        sim = pb.load(filename, paramname=paramname)
        
    return sim

def _interp_onestep(x,y,zero_slope=None):
    """
    Creates a spline interpolator for one step.  The interpolations are:
    
        - linear if zero_slope is None OR [False, False]
        - quadratic if one of zero_slope is True
        - cubic if both of zero_slope are True
        
    **ARGUMENTS**
    
    x : array like
        2 points
    y : array like
        for 2 points.  Can be a 2D array
        
    **RETURNS**
    
    spline : function
        A vectorized function returning the interpolated value of y.
    """
    
    dx = float(x[1] - x[0])
    dy = y[1] - y[0]
    
    y0shape = np.shape(y[0])
    
    if len(y0shape) == 0:
        
        ydim = 1
        
    else:
        
        ydim = y0shape[0]
    
    if (zero_slope is None) or np.all(y[1]==y[0]) or (np.all(~zero_slope)):
        # Slope is not fixed to be zero.  Do a linear interpolation
        a = dy/dx
        b = y[0]
        
        def f(xin):
            
            output = (xin - x[0])*a + b
            return output
            
    elif np.all(zero_slope):
        # Both slopes must be zero.  Do a cubic interpolation
        a = 3/dx**2
        b = 2/dx**3
        
        def f(xin):
            
            x1 = xin - x[0]
            return (a*x1**2 - b*x1**3)*dy + y[0]
            
    elif zero_slope[0]:
        # Just the left slope is zero
        a = dy/(dx**2)
        b = y[0]
        
        def f(xin):
            
            return a*(xin - x[0])**2 + b
            
    elif zero_slope[1]:
        # Just the right slope is zero
        a = dy/(dx**2)
        b = y[1]
        
        def f(xin):
            
            return -a*(xin - x[1])**2 + b
            
    else:
        
        raise ValueError, 'Could not understand zero_slope'
        
    def g(xin):
        """
        A Single step interpolator function.  Evaluating at x gives the
        interpolated value.
        
        Returns an array of shape (len(xin), y_dimension)
        """
        
        if not hasattr(xin, '__iter__'):
                
            xin = [xin]
        
        npts = len(xin)
        output = np.zeros([npts, ydim])
        
        for i in range(npts):
            
            output[i] = f(xin[i])
            
        return output
        
    return g

def interpolate(x, y, zero_slope=None):
    """
    Generates an interpolator for y evaluated at x.  zero_slope can be used
    to make y change slowly around point x.  The interpolation type used between
    x[i] and x[i+1] is as follows:
    
        - linear if zero_slope[i, i+1] = [False, False] (or if zero_slope=None)
        - quadratic if one of zero_slope[i, i+1] is True
        - cubic if zero_slope[i, i+1] = [True, True]
        
    Parameters
    ----------
    
    x : array like
        1D, length num_points
    y : array like
        1D, length num_points OR 2D, shape(num_points, y_dimension)
        
    Returns
    -------
    
        Interpolator function to interpolate y between the x data points.
    
    """     
    npts = len(x)
    nspl = npts - 1
    if npts < 2:
        
        raise ValueError, 'Need at least 2 points to interpolate'
        
    yshape = list(y.shape)
    
    # set up the zero_slopes
    if zero_slope is None:
        # Assume none are zeros
        zero_slope = np.zeros(npts, dtype=bool)
    elif zero_slope.dtype != np.bool:
        # Assmume these are indices
        temp = np.zeros(npts, dtype=bool)
        temp[zero_slope] = True
        zero_slope = temp
        
    spl_list = []
    
    for i in range(nspl):
        
        x1 = x[i:i+2]
        y1 = y[i:i+2]
        zero_slope1 = zero_slope[i:i+2]
        
        spl = _interp_onestep(x1,y1,zero_slope1)
        spl_list.append(spl)
        
    def spline(xpts):
        """
        Interpolator function.  Returns y interpolated at x.  x can be 
        array like or a number.
        """
        
        if np.ndim(xpts) == 0:
            xpts = np.array([xpts])
        
        ind = np.digitize(xpts, x) - 1
        ind[ind >= nspl] = nspl - 1
        ind[ind < 0] = 0
        
        nx = len(xpts)
        yshape[0] = nx
        y_out = np.zeros(yshape)
        
        for i in range(nspl):
            
            mask = (ind==i)
            
            if np.any(mask):
                y_out[mask] = spl_list[i](xpts[mask])
            
        return y_out
        
    return spline
    
def interpKeyframes(keyframe, nt=None):
    """
    Generates an interpolator to interpolate keyframes (for a single key)
    See pbmov.keyframes
    
    Parameters
    ----------
    
    keyframe : dict
        keyframe for a single key.  keyframes are stored as dictionaries.
        keyframe[i] is [value, zero_slope] at frame i
    nt : int
        (optional) Number of time steps.  If None, taken to be the last
        defined frame number
        
    Returns
    -------
    
    interp : interpolator (see pbmov_utils.interpolate)
        Interpolator, accessed by:
        >>> val = interp(frameNumber)
    """
    # Check that keyframes are properly formatted
    for frameNum in keyframe.keys():
        
        if not isinstance(frameNum, int):
            
            raise ValueError('Poorly formatted keyframe  '
            'frame: {0}.  Frame must be int'.format(frameNum))
            
    frameNums = keyframe.keys()
    frameNums.sort()
    nKeyframes = len(frameNums)
    
    if nt is None:
        
        nt = frameNums[-1] + 1
        
    # Set-up endpoints
    keyframe[0] = keyframe[frameNums[0]]
    frameNums = keyframe.keys()
    frameNums.sort()
    keyframe[nt-1] = keyframe[frameNums[-1]]
    frameNums = keyframe.keys()
    frameNums.sort()
    nKeyframes = len(frameNums)
    
    # Generate interpolator
    x = np.array(frameNums)
    y = []
    zero_slope = []
    
    for i in frameNums:
        
        a = keyframe[i]
        y.append(a[0])
        zero_slope.append(a[1])
        
    y = np.array(y)
    zero_slope = np.array(zero_slope)
    interp = interpolate(x, y, zero_slope)
    
    return interp

def perpendicular_vector(v):
    """
    Returns a vector perpendicular to v.  v can be of arbitrary length
    
    **ARGUMENTS**
    
    v : array
        1D array (ie a vector)
    
    **RETURNS**
    
    b : array
        1D array length(v).  A vector perpendicular to v
    """
    
    # Cast as an array for safety
    v = np.asarray(v, dtype=float)
    
    if np.all(v == 0):
        
        raise ValueError,'Zero vector as input...cannot make perpendicular vector'
                
    n_elements = len(v)
    b = np.zeros(n_elements)
    
    # Check for case where an element of v=0
    for i in range(n_elements):
        
        if v[i] == 0:
            
            b[i] = 1
            
            return b
            
    # Otherwise, just let the first 2 elements of b be non-zero and solve
    # the condition that v.dot.b = 0
    c = v[0]/v[1]
    b[0] = np.sqrt(1 / (1 + c**2))
    b[1] = -c * b[0]
    return b


def frame_width(camera, target, fov = 30):
    """
    Get frame width from camera, target positions and field of view
    
    **ARGUMENTS**
    
    camera : array
        Camera position.  A 1D array
    target : array
        Target position.  A 1D array
    fov : float
        Field of view (degrees) of the camera
        
    **RETURNS**
    
    width : float (or SimArray)
        width of the frame (used for pynbody.plot.sph.image())
    """
    camera = np.asarray(camera)
    target = np.asarray(target)
    
    # distance from camera to target
    d = np.sqrt(((camera - target)**2).sum(-1))
    # field of view in radians
    fov_rad = fov * np.pi/180
    # Frame width
    width = 2*d*np.tan(fov_rad/2)
    
    return width
    
def vsm_transform(pos, camera, target=None, camera_rot = 0):
    """
    Transform pos from world-space to view-space for a given target, camera
    configuration.
    
    **ARGUMENTS**
    
    pos : array or SimArray
        An Nx3 2D array of particle positions.
    camera : array or SimArray
        A 1D array of (x,y,z) for the camera position
    target : array or SimArray
        A 1D array of (x,y,z) for the target position.  If None, target
        is assumed to be the origin.
    camera_rot : float
        Amount by which to rotate the camera (radians) around the axis between
        the camera and the target.  The rotation is done after the camera is
        pointing at the target.
        
    **RETURNS**
    
    vs_pos : array
        View space position of the particles
    """
    # Go into 4D (homogenous) space
    pos1 = add_column(pos)
    # Generate the view-space transformation matrix
    vsm = view_space_matrix(camera, target, camera_rot)
    # Calculate view-space coordinates
    pos2 = np.dot(vsm, pos1.T).T
    # Go back to normal 3D space
    vs_pos = np.delete(pos2, -1, -1)
    
    return vs_pos

def add_column(x, fill_value=1):
    """
    Adds a column of fill_value (by default ones) to the last dimension of x.
    i.e., adds a bunch of ones along the last dimension.  For a 2D matrix this
    is equivalent to a column of fill_value.
    
    Useful for converting positions to 4D space for transformations.
    
    To delete the column, just do:
        numpy.delete(x, -1, -1)
    """
    shape = list(x.shape)
    shape[-1] = 1
    values = np.ones(shape)
    if fill_value != 1:
        
        values *= fill_value
        
    x2 = np.append(x, values, axis=-1)
    
    return x2
    
def view_space_matrix(camera, target=None, camera_rot=0):
    """
    Generates the view-space transformation matrix for a camera pointing at a
    target.
    
    **ARGUMENTS**
    
    camera : array or SimArray
        A 1D array of (x,y,z) for the camera position
    target : array or SimArray
        A 1D array of (x,y,z) for the target position.  If None, target
        is assumed to be the origin.
    camera_rot : float
        Amount by which to rotate the camera (radians) around the axis between
        the camera and the target.  The rotation is done after the camera is
        pointing at the target.
        
    **RETURNS**
    
    vsm : array
        View-space transformation matrix
    """
    camera = np.asarray(camera)
    
    if target is None:
        
        target=np.zeros(3)
        
    else:
        
        target = np.asarray(target)
        
    # Unit vector pointing from camera to target
    cam_dir = normalize(target - camera)
    # Get the rotation matrix to place the camera along the z-axis
    axis,angle = rotation_pars([0,0,-1], cam_dir)
    R1 = rotation_matrix(axis, angle)
    # Translation matrix to place target at origin
    T = translation_matrix(-target)
    
    # Generate matrix for final camera rotation around axis between camera
    # and target
    if camera_rot==0:
        
        R2 = np.identity(4)
        
    else:
        
        R2 = rotation_matrix([0,0,1], -camera_rot)
    
    # Put all the transformations together and return
    return np.dot(np.dot(R2, R1), T)
    
def normalize(x, axis=-1, return_mag = False):
    """
    Normalize a vector or an array of vectors
    
    NaNs will be returned for a zero vector
    
    **ARGUMENTS**
    
    x : numpy array
        Array to normalize.  Either 1D array or a 2D array of vectors.  If
        2D, shape MxN for M vectors of length N
    axis : int
        If x is 2D, axis to normalize along.  Default is -1
    return_mag : bool
        Default = False.  Optionally return the magnitude(s) of x as well.
        
    **RETURNS**
    
    x_normalized : array
        Normalized vector(s)
    x_mag (optional) : array or float
        magnitude(s) of input vector(s)
    """
    x_mag = np.sqrt((x**2).sum(axis))
    
    if np.ndim(x) > 1:
        
        x = x/x_mag[:,None]
        
    else:
        
        x = x/x_mag
        
    if return_mag:
        
        return x, x_mag
        
    else:
        
        return x
    
def rotation_pars(v1, v2):
    """
    Calculates the rotation axis and angle of rotation required to rotate a
    vector from the direction of v1 to the direction of v2
    
    **ARGUMENTS**
    
    v1, v2 : numpy arrays
        Vectors defining initial and final directions for a rotation from
        the direction of v1 to v2.
        Either both are vectors in 3D or Nx3 arrays where N is the number of
        rotations to consider
        
    **RETURNS**
    
    axis : array
        Nx3 array of rotation axes where N is the number of rotations performed
        For all parallel v1,v2's a vector perpendicular to v1 is used as the axis
    theta : array
        Rotation angles
    """
    # Cast as arrays
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    
    if v1.shape != v2.shape:
        
        raise ValueError, "Array dimensions don't match"
        
    ndim = np.ndim(v1)
    
    if ndim > 1:
        
        n_vec = len(v1)
        
    else:
        
        n_vec = 1
        
    # Normalize input vectors along the last axis
    v1 = normalize(v1)
    v2 = normalize(v2)
    # Calculate rotation axis
    axis = np.cross(v1, v2)
    # Normalize axis
    axis = normalize(axis)
    # Calculate rotation angle
    theta = np.arccos((v1*v2).sum(-1))
    
    # Handle parallel vectors.  By default, rotation around v1 of 0 degrees
    mask = (theta % np.pi == 0)
    
    if ndim > 1:
        
        for i in np.arange(n_vec)[mask]:
            
            axis[i] = perpendicular_vector(v1[i])
        
    elif mask:
        
        axis = perpendicular_vector(v1)
        
    
    return axis,theta
    
def translation_matrix(direction):
    """
    Generate a translation matrix in N+1 dimensions (where N=len(direction))
    
    **ARGUMENTS**
    
    direction : array
        1D array specifying the translation vector in N-dim space
    
    **RETURNS**
    
    T : array
        N-dim + 1 square matrix specifying the translation
    """
    ndim = len(direction)
    mat = np.identity(ndim+1)
    mat[:ndim, ndim] = direction[:ndim]
    
    return mat
    
def rotation_matrix(axis, theta, fourD=True):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians, using the Euler-Rodrigues formula
    
    **ARGUMENTS**
    
    axis : list, array
        Axis (a vector) in 3D to rotate around
    theta : float
        Angle to rotate by (radians)
    fourD : bool
        Return a 4D matrix (default=True)
        
    **RETURNS**
    
    rot_mat : numpy array
        Rotation matrix
    """
    # Special case of no rotation:
    if theta==0:
        
        if fourD:
            
            mat = np.identity(4)
            
        else:
            
            mat = np.identity(3)
            
        return mat
        
    # re-cast as array (lists are acceptable)
    axis = np.asarray(axis, dtype=float)
    # Normalize the rotation axis (make a unit vector)
    axis = normalize(axis)
    # Calculate the Euler parameters
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    
    # Generate rotation matrix
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    
    if fourD:
        
        rot_mat = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0], \
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0], \
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],\
        [0, 0, 0, 1] ])
        
    else:
        
        rot_mat = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], \
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], \
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
        
    return rot_mat
    