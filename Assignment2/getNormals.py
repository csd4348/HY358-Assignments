from statistics import mode
from turtle import width
# import unittest

import numpy as np
# from sympy import true

import pyECSS.utilities as util
from pyECSS.Entity import Entity
from pyECSS.Component import BasicTransform, Camera, RenderMesh
from pyECSS.System import System, TransformSystem, CameraSystem, RenderSystem
from pyGLV.GL.Scene import Scene
from pyECSS.ECSSManager import ECSSManager
from pyGLV.GUI.Viewer import SDL2Window, ImGUIDecorator, RenderGLStateSystem

from pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from pyGLV.GL.VertexArray import VertexArray

from OpenGL.GL import GL_LINES

import OpenGL.GL as gl
from statistics import mode
from turtle import width
# import unittest

import imgui as ImGui


def Normals(vertexArray, flag):
    A = []
    i = 0
    while(i < 4) :
        if (i == 0) :
            ab = (vertexArray[i+1][0] - vertexArray[i][0], vertexArray[i+1][1] - vertexArray[i][1], vertexArray[i+1][2] - vertexArray[i][2])
            ad = (vertexArray[i+3][0] - vertexArray[i][0], vertexArray[i+3][1] - vertexArray[i][1], vertexArray[i+3][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
            
        elif(i == 3) :
            ab = (vertexArray[i-3][0] - vertexArray[i][0], vertexArray[i-3][1] - vertexArray[i][1], vertexArray[i-3][2] - vertexArray[i][2])
            ad = (vertexArray[i-1][0] - vertexArray[i][0], vertexArray[i-1][1] - vertexArray[i][1], vertexArray[i-1][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
            
        else :
            ab = (vertexArray[i-1][0] - vertexArray[i][0], vertexArray[i-1][1] - vertexArray[i][1], vertexArray[i-1][2] - vertexArray[i][2])
            ad = (vertexArray[i+1][0] - vertexArray[i][0], vertexArray[i+1][1] - vertexArray[i][1], vertexArray[i+1][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
        i+=1
        
    while(i < 8 and flag == 1) :
        if (i == 4) :
            ab = (vertexArray[i+1][0] - vertexArray[i][0], vertexArray[i+1][1] - vertexArray[i][1], vertexArray[i+1][2] - vertexArray[i][2])
            ad = (vertexArray[i+3][0] - vertexArray[i][0], vertexArray[i+3][1] - vertexArray[i][1], vertexArray[i+3][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
            
        elif(i == 7) :
            ab = (vertexArray[i-3][0] - vertexArray[i][0], vertexArray[i-3][1] - vertexArray[i][1], vertexArray[i-3][2] - vertexArray[i][2])
            ad = (vertexArray[i-1][0] - vertexArray[i][0], vertexArray[i-1][1] - vertexArray[i][1], vertexArray[i-1][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
    
        else :
            ab = (vertexArray[i-1][0] - vertexArray[i][0], vertexArray[i-1][1] - vertexArray[i][1], vertexArray[i-1][2] - vertexArray[i][2])
            ad = (vertexArray[i+1][0] - vertexArray[i][0], vertexArray[i+1][1] - vertexArray[i][1], vertexArray[i+1][2] - vertexArray[i][2])
            ae = (0.0,0.0,0.0)
            if(flag == 1):
                ae = (vertexArray[i-4][0] - vertexArray[i][0], vertexArray[i-4][1] - vertexArray[i][1], vertexArray[i-4][2] - vertexArray[i][2])
            else :
                ae = (vertexArray[len(vertexArray)-1][0] - vertexArray[i][0], vertexArray[len(vertexArray)-1][1] - vertexArray[i][1], vertexArray[len(vertexArray)-1][2] - vertexArray[i][2])
            A.append([-(ab[0]+ad[0]+ae[0]),-(ab[1]+ad[1]+ae[1]),-(ab[2]+ad[2]+ae[2]), 1.0])
        i+=1
    
    if(flag == 0):
        i = 0
        vec = [0.0,0.0,0.0,1.0]
        while(i<len(vertexArray)-1) :
            ab = (vertexArray[i][0] - vertexArray[len(vertexArray)-1][0], vertexArray[i][1] - vertexArray[len(vertexArray)-1][1], vertexArray[i][2] - vertexArray[len(vertexArray)-1][2])
            ac = (vertexArray[i+1][0] - vertexArray[len(vertexArray)-1][0], vertexArray[i+1][1] - vertexArray[len(vertexArray)-1][1], vertexArray[i+1][2] - vertexArray[len(vertexArray)-1][2])
            if(i == 3):
                ab = (vertexArray[i][0] - vertexArray[len(vertexArray)-1][0], vertexArray[i][1] - vertexArray[len(vertexArray)-1][1], vertexArray[i][2] - vertexArray[len(vertexArray)-1][2])
                ac = (vertexArray[0][0] - vertexArray[len(vertexArray)-1][0], vertexArray[0][1] - vertexArray[len(vertexArray)-1][1], vertexArray[0][2] - vertexArray[len(vertexArray)-1][2])
            c1 = np.cross(ac,ab)
            vec[0] += c1[0]
            vec[1] += c1[1]
            vec[2] += c1[2]
            i += 1
        A.append([-(vec[0]),-(vec[1]),-(vec[2]), 1.0])
    print(A)
    return A
    
 
""" 
print(Normals(vertexCube, 1))
print(Normals(vertexPyramid, 0))
"""

"""
Common setup for all unit tests
Scenegraph for unit tests:
root
    |---------------------------|           
    entityCam1,                 node4,      
    |-------|                    |--------------|----------|--------------|           
    trans1, entityCam2           trans4,        mesh4,     shaderDec4     vArray4
            |                               
            ortho, trans2                   
"""

scene = Scene()    

# Scenegraph with Entities, Components
rootEntity = scene.world.createEntity(Entity(name="RooT"))
entityCam1 = scene.world.createEntity(Entity(name="entityCam1"))
scene.world.addEntityChild(rootEntity, entityCam1)
trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()))

entityCam2 = scene.world.createEntity(Entity(name="entityCam2"))
scene.world.addEntityChild(entityCam1, entityCam2)
trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()))
orthoCam = scene.world.addComponent(entityCam2, Camera(util.ortho(-100.0, 100.0, -100.0, 100.0, 1.0, 100.0), "orthoCam","Camera","500"))

node4_c1 = scene.world.createEntity(Entity(name="node4_c1"))
scene.world.addEntityChild(rootEntity, node4_c1)
trans4_c1 = scene.world.addComponent(node4_c1, BasicTransform(name="trans4_c1", trs=util.identity()))
mesh4_c1 = scene.world.addComponent(node4_c1, RenderMesh(name="mesh4_c1"))

node4_p = scene.world.createEntity(Entity(name="node4_p"))
scene.world.addEntityChild(rootEntity, node4_p)
trans4_p = scene.world.addComponent(node4_p, BasicTransform(name="trans4_p", trs=util.identity()))
mesh4_p = scene.world.addComponent(node4_p, RenderMesh(name="mesh4_p"))

axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))

normal = scene.world.createEntity(Entity(name="normal"))
scene.world.addEntityChild(rootEntity, normal)
normal_trans = scene.world.addComponent(normal, BasicTransform(name="normal_trans", trs=util.identity()))
normal_mesh = scene.world.addComponent(normal, RenderMesh(name="normal_mesh"))

normal_pyr = scene.world.createEntity(Entity(name="normal_pyr"))
scene.world.addEntityChild(rootEntity, normal_pyr)
normal_pyr_trans = scene.world.addComponent(normal_pyr, BasicTransform(name="normal_pyr_trans", trs=util.identity()))
normal_pyr_mesh = scene.world.addComponent(normal_pyr, RenderMesh(name="normal_pyr_mesh"))

# a simple triangle
vertexData = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
],dtype=np.float32) 
colorVertexData = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)

#Colored Axes
vertexAxes = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
],dtype=np.float32) 
colorAxes = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)

#Simple Cube
vertexCube = np.array([
    [-0.5, -0.5, 0.5, 1.0], 
    [-0.5, 0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0], 
    [-0.5, -0.5, -0.5, 1.0],
    [-0.5, 0.5, -0.5, 1.0], 
    [0.5, 0.5, -0.5, 1.0], 
    [0.5, -0.5, -0.5, 1.0]
],dtype=np.float32) 
colorCube = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0]
], dtype=np.float32)

vertexPyramid = np.array([
    [-0.5, -0.5, 0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0],
    [0.5, -0.5, -0.5, 1.0],
    [-0.5, -0.5, -0.5, 1.0], 
    [0.0, 1.0, 0.0, 1.0]
],dtype=np.float32) 

colorPyramid = np.array([
    [1.0, 0.65, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 0.65, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0], 
    [0.0, 0.0, 1.0, 1.0]
],dtype=np.float32) 

#index arrays for above vertex Arrays
index = np.array((0,1,2), np.uint32) #simple triangle
indexAxes = np.array((0,1,2,3,4,5), np.uint32) #3 simple colored Axes as R,G,B lines
indexCube = np.array((1,0,3, 1,3,2, 
                  2,3,7, 2,7,6,
                  3,0,4, 3,4,7,
                  6,5,1, 6,1,2,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1), np.uint32) #rhombus out of two triangles
indexPyramid = np.array((0,1,2, 0,2,3, 0,1,4, 1,2,4, 2,3,4, 3,0,4), np.uint32) #rhombus out of two triangles

normal_lines_cube = []
Norm = Normals(vertexCube, 1)
for i in range(len(Norm)):
    normal_lines_cube.append([0.0,0.0,0.0,1.0])
    normal_lines_cube.append(Norm[i])
    
colorNormals = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
], dtype=np.float32)
indexNormals = np.array((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), np.uint32) #3 simple colored Axes as R,G,B lines

normal_lines_pyramid = []
Norma = Normals(vertexPyramid, 0)
for i in range(len(Norma)):
    normal_lines_pyramid.append([0.0,0.0,0.0,1.0])
    normal_lines_pyramid.append(Norma[i])

colorNormalsPyr = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
], dtype=np.float32)
indexNormalsPyr = np.array((0,1,2,3,4,5,6,7,8,9), np.uint32) #3 simple colored Axes as R,G,B lines

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

## ADD CUBE ##
# attach a simple cube in a RenderMesh so that VertexArray can pick it up
mesh4_c1.vertex_attributes.append(vertexCube)
mesh4_c1.vertex_attributes.append(colorCube)
mesh4_c1.vertex_index.append(indexCube)
vArray4 = scene.world.addComponent(node4_c1, VertexArray())
shaderDec4_c1 = scene.world.addComponent(node4_c1, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

mesh4_p.vertex_attributes.append(vertexPyramid)
mesh4_p.vertex_attributes.append(colorPyramid)
mesh4_p.vertex_index.append(indexPyramid)
vArray4 = scene.world.addComponent(node4_p, VertexArray())
shaderDec4_p = scene.world.addComponent(node4_p, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

# Generate terrain
from pyGLV.GL.terrain import generateTerrain
vertexTerrain, indexTerrain, colorTerrain= generateTerrain(size=4,N=20)
# Add terrain
terrain = scene.world.createEntity(Entity(name="terrain"))
scene.world.addEntityChild(rootEntity, terrain)
terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.identity()))
terrain_mesh = scene.world.addComponent(terrain, RenderMesh(name="terrain_mesh"))
terrain_mesh.vertex_attributes.append(vertexTerrain) 
terrain_mesh.vertex_attributes.append(colorTerrain)
terrain_mesh.vertex_index.append(indexTerrain)
terrain_vArray = scene.world.addComponent(terrain, VertexArray(primitive=GL_LINES))
terrain_shader = scene.world.addComponent(terrain, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))
# terrain_shader.setUniformVariable(key='modelViewProj', value=mvpMat, mat4=True)

## ADD AXES ##
axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
axes_mesh.vertex_attributes.append(vertexAxes) 
axes_mesh.vertex_attributes.append(colorAxes)
axes_mesh.vertex_index.append(indexNormals)
axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES)) # note the primitive change

axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

## ADD NORMALS FOR CUBE##
normal = scene.world.createEntity(Entity(name="normal"))
scene.world.addEntityChild(rootEntity, normal)
normal_trans = scene.world.addComponent(normal, BasicTransform(name="normal_trans", trs=util.identity()))
normal_mesh = scene.world.addComponent(normal, RenderMesh(name="normal_mesh"))
normal_mesh.vertex_attributes.append(normal_lines_cube) 
normal_mesh.vertex_attributes.append(colorNormals)
normal_mesh.vertex_index.append(indexNormals)
normal_vArray = scene.world.addComponent(normal, VertexArray(primitive=GL_LINES)) # note the primitive change

normal_shader = scene.world.addComponent(normal, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

## ADD NORMALS FOR PYRAMID##
normal_pyr = scene.world.createEntity(Entity(name="normal_pyr"))
scene.world.addEntityChild(rootEntity, normal_pyr)
normal_pyr_trans = scene.world.addComponent(normal_pyr, BasicTransform(name="normal_pyr_trans", trs=util.identity()))
normal_pyr_mesh = scene.world.addComponent(normal_pyr, RenderMesh(name="normal_pyr_mesh"))
normal_pyr_mesh.vertex_attributes.append(normal_lines_pyramid) 
normal_pyr_mesh.vertex_attributes.append(colorNormalsPyr)
normal_pyr_mesh.vertex_index.append(indexNormalsPyr)
normal_pyr_vArray = scene.world.addComponent(normal_pyr, VertexArray(primitive=GL_LINES)) # note the primitive change

normal_pyr_shader = scene.world.addComponent(normal_pyr, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

running = True
# MAIN RENDERING LOOP
scene.init(imgui=True, windowWidth = 1024, windowHeight = 768, windowTitle = "pyglGA test_renderAxesTerrainEVENT")

# pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
# needs an active GL context

# vArrayAxes.primitive = gl.GL_LINES

scene.world.traverse_visit(initUpdate, scene.world.root)

################### EVENT MANAGER ###################

eManager = scene.world.eventManager
gWindow = scene.renderWindow
gGUI = scene.gContext

renderGLEventActuator = RenderGLStateSystem()

eManager._subscribers['OnUpdateWireframe'] = gWindow
eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
eManager._subscribers['OnUpdateCamera'] = gWindow 
eManager._actuators['OnUpdateCamera'] = renderGLEventActuator
# MANOS END
# Add RenderWindow to the EventManager publishers
# eManager._publishers[updateBackground.name] = gGUI

eye = util.vec(2.5, 1.7, 2.5)
target = util.vec(-0.7, 0.0, 0.0)
up = util.vec(0.0, 1.0, -0.03)
view = util.lookat(eye, target, up)
# projMat = util.ortho(-10.0, 10.0, -10.0, 10.0, -1.0, 10.0) ## WORKING
# projMat = util.perspective(90.0, 1.33, 0.1, 100) ## WORKING
projMat = util.perspective(50.0, 1.0, 1.0, 10.0) ## WORKING 

gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

model_terrain_axes = util.translate(0.0,0.0,0.0)
model_normals_cube = util.translate(0.0,0.0,0.0)
model_normals_pyramid = util.translate(0.0,1.7,0.0)
model_cube1 = util.scale(0.3) @ util.translate(0.0,0.0,0.0)
pyramid = util.scale(0.5) @ util.translate(0.0,1.7,0.0)

translate_values_cube1 = model_cube1[0][3], model_cube1[1][3], model_cube1[2][3]
rotate_values_cube1 = 0.0, 0.0, 0.0
scale_values_cube1 = 1.0, 1.0, 1.0

translate_values_pyramid = pyramid[0][3], pyramid[1][3], pyramid[2][3]
rotate_values_pyramid = 0.0, 0.0, 0.0
scale_values_pyramid = 1.0, 1.0, 1.0

state_cube1 = False
state_normals_cube = False
state_pyramid = False
state_normals_pyramid = False

while running:
    running = scene.render(running)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
        
    ImGui.set_next_window_size(400.0, 450.0)
    ImGui.set_next_window_position(10., 10.)
    ImGui.begin("TRS", True)
    # Shows Simple text
    ImGui.text("Tranformatiom : ")

    # Goes to a new line
    ImGui.new_line()
    
    # Creates a simple slider for Translation Cube 1
    changed, translate_values_cube1 = ImGui.slider_float3("Translate Cube With Normals", translate_values_cube1[0], translate_values_cube1[1], translate_values_cube1[2], -2.0, 2.0, format("%.3f"))
    model_cube1[0][3] = translate_values_cube1[0]
    model_cube1[1][3] = translate_values_cube1[1]
    model_cube1[2][3] = translate_values_cube1[2]
    model_normals_cube[0][3] = translate_values_cube1[0]
    model_normals_cube[1][3] = translate_values_cube1[1]
    model_normals_cube[2][3] = translate_values_cube1[2]
    
    # Creates a simple slider for Rotation Cube 1
    changed, rotate_values_cube1 = ImGui.slider_float3("Rotate Cube 1", rotate_values_cube1[0], rotate_values_cube1[1], rotate_values_cube1[2], -3.0, 3.0, format("%.3f"))
    model_cube1 = model_cube1 @ util.rotate((1, 0, 0), rotate_values_cube1[0])
    model_cube1 = model_cube1 @ util.rotate((0, 1, 0), rotate_values_cube1[1])
    model_cube1 = model_cube1 @ util.rotate((0, 0, 1), rotate_values_cube1[2])
    model_normals_cube = model_normals_cube @ util.rotate((1, 0, 0), rotate_values_cube1[0])
    model_normals_cube = model_normals_cube @ util.rotate((0, 1, 0), rotate_values_cube1[1])
    model_normals_cube = model_normals_cube @ util.rotate((0, 0, 1), rotate_values_cube1[2])
    
    scale_values_cube1 = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_cube1 = ImGui.slider_float3("Scale Cube 1", scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2], 0.97, 1.03, format("%.3f"))
    model_cube1 = model_cube1 @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2])
    model_cube1 = model_cube1 @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2])  
    model_cube1 = model_cube1 @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2]) 
    model_normals_cube = model_normals_cube @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2])
    model_normals_cube = model_normals_cube @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2])  
    model_normals_cube = model_normals_cube @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2]) 
    
    # Creates a simple slider for Translation Cube 1
    changed, translate_values_pyramid = ImGui.slider_float3("Translate Pyramid With Normals", translate_values_pyramid[0], translate_values_pyramid[1], translate_values_pyramid[2], -2.0, 2.0, format("%.3f"))
    pyramid[0][3] = translate_values_pyramid[0]
    pyramid[1][3] = translate_values_pyramid[1]
    pyramid[2][3] = translate_values_pyramid[2]
    model_normals_pyramid[0][3] = translate_values_pyramid[0]
    model_normals_pyramid[1][3] = translate_values_pyramid[1]
    model_normals_pyramid[2][3] = translate_values_pyramid[2]
    
    # Creates a simple slider for Rotation Cube 1
    changed, rotate_values_pyramid = ImGui.slider_float3("Rotate Pyramid", rotate_values_pyramid[0], rotate_values_pyramid[1], rotate_values_pyramid[2], -3.0, 3.0, format("%.3f"))
    pyramid = pyramid @ util.rotate((1, 0, 0), rotate_values_pyramid[0])
    pyramid = pyramid @ util.rotate((0, 1, 0), rotate_values_pyramid[1])
    pyramid = pyramid @ util.rotate((0, 0, 1), rotate_values_pyramid[2])
    model_normals_pyramid = model_normals_pyramid @ util.rotate((1, 0, 0), rotate_values_pyramid[0])
    model_normals_pyramid = model_normals_pyramid @ util.rotate((0, 1, 0), rotate_values_pyramid[1])
    model_normals_pyramid = model_normals_pyramid @ util.rotate((0, 0, 1), rotate_values_pyramid[2])
    
    scale_values_pyramid = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_pyramid = ImGui.slider_float3("Scale Pyramid", scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2], 0.97, 1.03, format("%.3f"))
    pyramid = pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2])
    pyramid = pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2])  
    pyramid = pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2]) 
    model_normals_pyramid = model_normals_pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2])
    model_normals_pyramid = model_normals_pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2])  
    model_normals_pyramid = model_normals_pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2]) 
    
    
    view =  gWindow._myCamera # updates view via the imgui
    mvp_cube1 = projMat @ view @ model_cube1
    mvp_pyramid = projMat @ view @ pyramid
    mvp_terrain_axes = projMat @ view @ model_terrain_axes
    mvp_normals_cube = projMat @ view @ model_normals_cube
    mvp_normals_pyramid = projMat @ view @ model_normals_pyramid

    axes_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
    terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
    
    # Creates a checkbox
    changed, checkbox_cube1 = ImGui.checkbox("Show Cube", state_cube1)
    if checkbox_cube1 is False:
        shaderDec4_c1.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_cube1 = False
    if checkbox_cube1 is True:
        shaderDec4_c1.setUniformVariable(key='modelViewProj', value=mvp_cube1, mat4=True)
        state_cube1 = True

    # Creates a checkbox
    changed, checkbox_normals_cube = ImGui.checkbox("Show Normals Cube", state_normals_cube)
    if checkbox_normals_cube is False:
        normal_shader.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_normals_cube = False
    if checkbox_normals_cube is True:
        normal_shader.setUniformVariable(key='modelViewProj', value=mvp_normals_cube, mat4=True)
        state_normals_cube = True
        
    # Creates a checkbox
    changed, checkbox_pyramid = ImGui.checkbox("Show Pyramid", state_pyramid)
    if checkbox_pyramid is False:
        shaderDec4_p.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_pyramid = False
    if checkbox_pyramid is True:
        shaderDec4_p.setUniformVariable(key='modelViewProj', value=mvp_pyramid, mat4=True)
        state_pyramid = True
    
    # Creates a checkbox
    changed, checkbox_normals_pyramid = ImGui.checkbox("Show Normals Pyramid", state_normals_pyramid)
    if checkbox_normals_pyramid is False:
        normal_pyr_shader.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_normals_pyramid = False
    if checkbox_normals_pyramid is True:
        normal_pyr_shader.setUniformVariable(key='modelViewProj', value=mvp_normals_pyramid, mat4=True)
        state_normals_pyramid = True
        
    ImGui.separator()
    ImGui.end()
    scene.render_post()
    
scene.shutdown()