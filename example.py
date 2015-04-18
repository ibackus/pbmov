# -*- coding: utf-8 -*-
"""
An example script demonstrating the use of flyby

Created on Mon Mar 30 22:33:28 2015

@author: ibackus
"""
import pynbody as pb
SimArray = pb.array.SimArray

import flyby

#set the sim path to the simulation you want to see
sim_path = 'wengen_test_final.std'
param_path = 'wengen_test_final.param'
save_path = 'movie_lores.mp4'

# Load the simulation
sim = pb.load(sim_path, paramname=param_path)

# Frame rendering settings
vmin=1e-6
vmax=1e1
width = '62 au'
z_camera = 1000

# perform a 180 degree rotation around x, then a 360 degree rotation around
# y and z, then a 360 deg rotation around all 3
rotations = [[180,0,0], [0,360,360], [360,360,360]]
rot_frames = 100 # number of frames to render per rotation

# Video settings
res = 250
fps = 25
codec = "libx264"
preset = "slow"
quality = 18

# Render the movie for gas only!

im = flyby.rotation_movie(sim.gas, vmin, vmax, width, rotations, savename=save_path, rot_frames=rot_frames,\
res=res,fps=fps,codec=codec,preset=preset,quality=quality, z_camera=z_camera)