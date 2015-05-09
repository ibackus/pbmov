# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:35:23 2015

@author: ibackus
"""
import flyby
import flyby_utils

import numpy as np
import pynbody
SimArray = pynbody.array.SimArray
from scipy.interpolate import interp1d

import isaac

fname = 'data/wengen_test_final.std'
paramname = 'data/wengen_test_final.param'
savename = 'example.mp4'

target = np.array([ 8.70984,  9.2456 ,  0.     ])
vmin = 1e-8
vmax = 1e1
fps = 25
res = 720
tmax = 15.

nt = int(tmax*fps)
t = np.linspace(0, tmax, nt)

# ------------------------------------------
# Set up camera positions
# ------------------------------------------
cam = []
cam.append([0,0,100])
cam.append([0,0,20])
cam.append([5,5,5])
cam.append([8,9,0])
cam.append([8.5,9.1,0])
cam.append([0,0,-100])
cam = np.array(cam)
t_cam = np.array([0,2,5,10,13,tmax])
# Tell camera to slow down by certain of the camera spots
camstop = np.array([0,0,0,1,1,0], dtype=bool)
# could also do:
#   camstop = np.array([3,4])

# Now interpolate the camera locations
cam_spl = flyby_utils.interpolate(t_cam, cam, camstop)
cameras_all = cam_spl(t)
# ------------------------------------------
# Set up target positions
# ------------------------------------------
target = []
target.append([0,0,0])
target.append([ 8.70984,  9.2456 ,  0.     ])
target.append([ 8.70984,  9.2456 ,  0.     ])
target = np.array(target)
# times for target positions
t_target = np.array([0,t_cam[1], tmax])
# now interpolate
target_spl = flyby_utils.interpolate(t_target, target)
targets_all = target_spl(t)
# ------------------------------------------
# Set up a basic, constant camera rotation
# ------------------------------------------
camera_rots = np.linspace(0, 2*np.pi, nt)

# ------------------------------------------
# now render movie!
# ------------------------------------------
f = pynbody.load(fname, paramname=paramname)
isaac.snapshot_defaults(f)

flyby.render_movie(f.g, cameras_all, targets_all, nt, vmin, vmax, camera_rots, \
res, fps = fps, savename=savename)