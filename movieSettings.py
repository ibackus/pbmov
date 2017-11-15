# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:47:17 2015

@author: ibackus
"""
# External imports
import os
import numpy as np
import cPickle as pickle
from warnings import warn

# Internal imports
import pbmov_utils

# Setup defaults file
_dir = os.path.dirname(os.path.abspath(__file__))
fDefaults = os.path.join(_dir, '.keyframerc.py')
if not os.path.exists(fDefaults):
    
    fDefaults = os.path.join(_dir, 'defaults.py')

class movieSettings():
    """
    Defines the movieSettings class, used by pbmov for handling keyframes and
    storing movie settings.
    
    Examples
    --------
    
    >>> from movieSettings import movieSettings
    >>> settings = movieSettings()
    
    Change a movie-wide setting
    
    >>> settings.params['nt'] = 50
    
    Add a keyframe
    
    >>> settings.addKeyframes('cam', [0,1,0], frames=49, zero_slope=True)
    
    Delete a keyframe
    
    >>> settings.delKeyframes('cam', 0)
    
    Make sure frame-by-frame data is up-to-date
    
    >>> settings.makeFrames()
    
    Add frames without changing the total movie-time:
    
    >>> settings.timeStretch(200, adjustFPS=True)
    
    Retrieve a frame:
    
    >>> frameData = settings.getFrame(25)
    
    Notes
    -----
    
    Defaults are handled by defaults.py.  To change the defaults, create a
    .keyframerc.py file in the same folder as defaults.py
    
    Movie-wide settings are stored in settings.params
    Keyframes are stored in settings.keyframes
    """
    
    def __init__(self):
        
        self.defaults()
        self.frames = {}
        self.makeFrames()
        
    def __getstate__(self):
        """
        Defined to make self pickleable
        """
        state = {'params': self.params, 'keyframes': self.keyframes}
        return state
        
    def __setstate__(self, newstate):
        """
        Defined to make self un-pickleable
        """
        self.__init__()
        self.params = newstate['params']
        self.keyframes = newstate['keyframes']
        self.makeFrames()
        
    def timeStretch(self, ntNew, adjustFPS=False):
        """
        Increases the number of time steps without changing the relative time of 
        the keyframes.
        
        Note: if the number of timesteps is decreased, multiple keyframes may 
        overlap.  Earlier keyframes are deleted (ignored) first
        
        Parameters
        ----------
        
        ntNew : int
            New number of time steps (frames) for the movie
        adjustFPS : bool
            (optional) If True, the framerate is changed to keep the movie
            runtime constant
        """
        ntOld = self.params['nt']
        
        for k, v in self.keyframes.iteritems():
            
            self.keyframes[k] = timeStretch(v, ntNew, ntOld)
            
        self.params['nt'] = ntNew
        self.makeFrames()
        
        if adjustFPS:
            
            self.params['fps'] *= float(ntNew)/ntOld
        
    def defaults(self, filename=None):
        """
        Sets up the default values for movie parameters
        """
        if filename is None:
            
            filename = fDefaults
            
        g = {}
        execfile(filename, g)
        params = g['params']
        keyframes = g['keyframes']
        logvalues = g['logvalues']
        # Save to self
        self.params = params
        self.keyframes = keyframes
        self._logvalues = logvalues
        
    def makeFrames(self, key=None):
        """
        Generates the frames for key (or all keys) by interpolating the
        keyFrames.  Frames are stored in self.frames[key]
        
        Parameters
        ----------
        
        key : str
            (optional) Key to generate.  If None, all keys are generated
        """
        
        # If there are no arguments, loop through and make frames for alll
        # keys
        if key is None:
            
            for k in self.keyframes.keys():
                
                self.makeFrames(k)
                
            return
        
        # ---------------
        # make frames
        # ---------------
        nt = self.params['nt']
        # Get a copy of the keyframe value
        keyframe = self.keyframes[key].copy()
        # Check for Nones
        for val in keyframe.values():
            
            if val[0] is None:
                
                # Make all values None.  There's no way to interpolate a None
                self.frames[key] = np.array([None]*nt)
                return
                
        # Generate interpolator
        log = key in self._logvalues
        interp = pbmov_utils.interpKeyframes(keyframe, nt, log)      
        # Evaluate interpolator
        self.frames[key] = interp(np.arange(nt))
        
        return
        
    def getFrame(self, frame):
        """
        Retrieves the key, val pairs at a given frame.
        (By default, runs self.makeFrames())
        
        Parameters
        ----------
        
        frame : int
            Frame number
        
        Returns
        -------
        
        frameDict : dict
            Dictionary containing all the key, val pairs at frame
        """
        self.makeFrames()
        frameDict = {}
        for key in self.keyframes.keys():
            
            frameDict[key] = self.frames[key][frame]
            
        return frameDict
        
    
    def addKeyframes(self, key, vals, frames=0, zero_slope=False):
        """
        Adds keyframe(s), specified by the key, frame number(s), and value(s)
        
        Note that vals, frames, and zero_slope should be the same length
        
        Parameters
        ----------
        
        key : str
            key (parameter) that the keyframe controls
        vals : obj or list of obj
            The value(s) at each frame.  If frames is a list, should be a list
            of the same length
        frames : int or list of ints
            The frame number (numbers).  If a list, vals should be a list of
            values
        zero_slope : bool or list of bools
            (see pbmov_utils.interpolate) A flag which tells whether the value
            specified by key should change slowly around the current frame
        """
        
        # Turn into lists
        if not hasattr(frames, '__iter__'):
            
            frames = [frames]
            vals = [vals]
        
        nFrames = len(frames)
        if isinstance(zero_slope, bool):
            
            zero_slope = [zero_slope] * nFrames
            
        if key not in self.keyframes:
            
            self.keyframes[key] = {}
            
        for i in range(nFrames):
            
            self.keyframes[key][frames[i]] = [vals[i], zero_slope[i]]
            
        # Update the frames
        self.makeFrames(key)
        
    def delKeyframes(self, key, frames):
        """
        Deletes keyframe(s) from self, specified by key, frame.  Nothing is
        done if the keyframe is not present.
        
        Parameters
        ----------
        
        key : str
            Key to delete, e.g. 'cam' or 'target'
        frame : int or list of ints
            keyframe number(s) to delete
        """
        if not hasattr(frames, '__iter__'):
            # Assume frames is an int, make it a list
            frames = [frames]
            
        # Loop through all frames
        for frame in frames:
            
            try:
                # Delete the keyframe
                del self.keyframes[key][frame]
            except KeyError:
                # Assume the keyframe is not present
                pass
        
        # Update the frames
        self.makeFrames(key)
        return
        
    def save(self, filename='movieSettings.p'):
        """
        Save to filename
        
        Parameters
        ----------
        
        filename : str
            Filename to save to
        """
        pickle.dump(self, open(filename, 'w'))
        
def timeStretch(keyframes, ntNew, ntOld=None):
    """
    Increases the number of time steps without changing the relative time of 
    the keyframes.
    
    Note: if the number of timesteps is decreased, multiple keyframes may 
    overlap.  Earlier keyframes are deleted (ignored) first
    
    Parameters
    ----------
    
    keyframes : dict
        Keyframes dict (see movieSettings) for a single key.
    ntNew : int
        New number of time steps (frames) for the movie
    ntOld : int
        (optional) old number of time steps (frames) for the movie. If None,
        it the last keyframe is assumed to be the last frame of the movie.
    
    Returns
    -------
    
    newframes : dict
        Updated keyframes
    """
    
    keyframes = keyframes.copy()
    # Do nothing if nt doesn't change
    if ntOld == ntNew:
        
        return keyframes
    
    # Original keys (frame numbers)
    oldKeys = np.array(keyframes.keys())
    oldKeys.sort()
    if ntOld is None:
        ntOld = oldKeys[-1] + 1
    # Scale frame numbers
    newKeys = (float(ntNew)/ntOld) * oldKeys
    newKeys = np.round(newKeys).astype(int)
    # Apply maximum frame number
    newKeys[newKeys >= ntNew-1] = ntNew-1
    # check for duplicates
    if len(set(newKeys)) < len(newKeys):
        
        warn('Duplicate keys detected after timestretch.  Earlier keys will '
        'be ignored')
        
    newframes = {}
    
    for new, old in zip(newKeys, oldKeys):
        
        newframes[new] = keyframes[old]
    
    return newframes