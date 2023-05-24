import pyECSS.utilities as util

def myview(listOf3Dpoints):
    eye = util.vec(1.0, 1.0, 1.0)
    target = util.vec(0.0, 0.0, 0.0)
    up = util.vec(0.0, 1.0, 0.0)
    view = util.lookat(eye, target, up)

    fov = 60.0
    aspect = 640/360
    nearclip = 1.0
    farclip = 100.0
    projMat = util.perspective(fov, aspect, nearclip, farclip)
    
    points = projMat @ view @ listOf3Dpoints
    print(points)
    return points

myview((3, 1, 2, 1))
