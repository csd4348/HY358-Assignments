import pyECSS.utilities as util

def modelpoints(arrayOf3Dpoints, t_3Dvec, axe, theta, vector3D):
    scaled = util.scale(vector3D[0], vector3D[1], vector3D[2])  
    
    i=0
    for element in axe:
        #print(axe[i], theta[i])
        scaledRotated = scaled @ util.rotate(axe[i], theta[i]) 
        i = i + 1
    
    scaledRotatedTranslated = scaledRotated @ util.translate(t_3Dvec[0], t_3Dvec[1], t_3Dvec[2]) @ arrayOf3Dpoints
    print("Scaled, Rotated and Translated point go to: ", scaledRotatedTranslated)
    return scaledRotatedTranslated

modelpoints([3, 1, 2, 1], (2, 2, 4), [(0, 1, 0)], [90], (3, 3, 3, 1))       #TEST 5%
modelpoints([3, 1, 2, 1], (2, 2, 4), [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [90, 45, 15], (3, 3, 3, 1))       #TEST 5%
#modelpoints([[3, 1, 2, 1],[1, 2, 2, 1],[2, 1, 2, 1],[2, 1, 3, 1]], (2, 2, 4), [(0, 1, 0)], [90], (3, 3, 3, 1))       #TEST 5%