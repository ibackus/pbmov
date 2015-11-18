# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:51:36 2015

@author: ibackus
"""
# External packages
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import numpy as np
import pynbody as pb
SimArray = pb.array.SimArray
import os
from diskpy.utils import pbverbosity
import cPickle as pickle
from copy import copy

# Internal modules
import cubehelix
import ffmpeg_writer
import pbmov_utils
import pbmov
from render import renderFrame
from movieSettings import movieSettings

class movie():
    """
    
    """
    
    def __init__(self, sim, paramname=None, settings=None, renderer=renderFrame):
        
        # Setup the simulation to use
        try:
            
            self.setSim(sim, paramname)
            
        except IOError:
            
            print 'Could not find snapshot {0}.  Add a snapshot for rendering \
with self.setSim'.format(sim)
        # Define the renderer (for rendering frames)
        self._renderer = renderer
        
        # Initialize settings
        if settings is None:
            
            settings = movieSettings()
            
        self._settings = settings
        
        # Reference things from settings
        self.params = settings.params
        self.keyframes = settings.keyframes
        self.addKeyframes = settings.addKeyframes
        self.delKeyframes = settings.delKeyframes
        self.timeStretch = settings.timeStretch
        
    def __getstate__(self):
        """
        Allows for pickling
        """
        # Setup everything that's needed
        savekeys = ['_fSim', '_fParam', '_settings', '_renderer']
        state = {}
        
        for key in savekeys:
            
            state[key] = copy(self.__dict__[key])
        
        return state
        
    def __setstate__(self, s):
        """
        Allows for pickling
        """
        self.__init__(s['_fSim'], s['_fParam'], s['_settings'], s['_renderer'])
        #self.__dict__.update(newstate)
        
    def save(self, filename='movie.p'):
        """
        Saves movie object to filename.  Does not save the associated snapshot
        
        Parameters
        ----------
        
        filename : str
            File to save to
        """
        pickle.dump(self, open(filename, 'w'))
    
    def preview(self, frameNum=0):
        
        p = self._settings.params
        frameSettings = self._settings.getFrame(frameNum)
        self._renderer(frameSettings, p, self.sim, preview=True)
        
    def setSim(self, sim, paramname=None):
        
        # Assume sim is a filename if it's a string
        if isinstance(sim, str):
            
            if sim != '<created>':
                
                sim = pbmov_utils.pbload(sim, paramname=paramname)
                
            else:
                
                raise ValueError, 'Cannot load sim <created>'
            
        # Save to self
        self.sim = sim
        self.pos0 = sim['pos'].copy()
        # Setup filenames
        self._fSim = self.sim.filename
        self._fParam = paramname
        
    def makeMovie(self):
        
        # Make sure the frames have been made from the keyframes
        self._settings.makeFrames()
        
        # Make sure original position is set
        self.pos0 = self.sim['pos'].copy()
        
        # Initialize video writer
        res = self.params['res']
        fps = self.params['fps']
        savename = self.params['savename']
        video_writer = ffmpeg_writer.FFMPEG_VideoWriter(savename, (res,res), fps)
        
        # Make movie
        nt = self.params['nt']
        verbosity = pbverbosity('off')
        
        for i in range(nt):
            
            print '{0} of {1}'.format(i+1, nt)
            # Render frame
            frame = self._settings.getFrame(i)
            image = self._renderer(frame, self.params, self.sim, pos0=self.pos0)
            # write to video
            video_writer.write_frame(image)
            
        # Finalize
        video_writer.close()
        pbverbosity(verbosity)
        self.sim['pos'] = self.pos0