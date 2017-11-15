# -*- coding: utf-8 -*-
"""
DO NOT CHANGE THIS FILE.  TO CHANGE THE DEFAULTS, CREATE .keyframerc.py IN
THE SAME DIRECTORY AS THIS FILE

Defaults for pbmov.keyframes

These can be overridden by including a .keyframerc.py file in the pbmov directory
Just make sure the dicts 'params' and 'keyframes' are created.

Created on Thu Aug 27 11:30:15 2015

@author: ibackus
"""

# params values
params = {}
params['nt'] = 20
params['res'] = 250
params['cmap'] = 'cubehelix'
params['fps'] = 5
params['savename'] = 'movie.mp4'
# Additional keyword arguments to pass to pynbody during rendering
params['pbkwargs'] = None

# Animateable values
keyframes = {}
keyframes['cam'] = {0: [[0,0,100], False]}
keyframes['target'] = {0: [[0,0,0], False]}
keyframes['simRot'] = {0: [[1, 0, 0], False]}
keyframes['vmin'] = {0: [None, False]}
keyframes['vmax'] ={0:  [None, False]}
keyframes['camOrient'] = {0: [0.0, False]}

# Values to interpolate logarithmically
logvalues = ['vmin', 'vmax']