

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

import imgui as ImGui



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

node4_c2 = scene.world.createEntity(Entity(name="node4_c2"))
scene.world.addEntityChild(rootEntity, node4_c2)
trans4_c2 = scene.world.addComponent(node4_c2, BasicTransform(name="trans4_c2", trs=util.identity()))
mesh4_c2 = scene.world.addComponent(node4_c2, RenderMesh(name="mesh4_c2"))

node4_p = scene.world.createEntity(Entity(name="node4_p"))
scene.world.addEntityChild(rootEntity, node4_p)
trans4_p = scene.world.addComponent(node4_p, BasicTransform(name="trans4_p", trs=util.identity()))
mesh4_p = scene.world.addComponent(node4_p, RenderMesh(name="mesh4_p"))


axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))

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
    [-0.5, -1.0, 0.5, 1.0],
    [-0.5, 0.0, 0.5, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, -1.0, 0.5, 1.0], 
    [-0.5, -1.0, -0.5, 1.0], 
    [-0.5, 0.0, -0.5, 1.0], 
    [0.5, 0.0, -0.5, 1.0], 
    [0.5, -1.0, -0.5, 1.0]
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

#Simple Cube
vertexCube1 = np.array([
    [-0.5, -0.5, 0.5, 1.0], # Α
    [-0.5, 0.5, 0.5, 1.0],  # Β
    [0.5, 0.5, 0.5, 1.0],   # Γ
    [0.5, -0.5, 0.5, 1.0],  # Δ
    [-0.5, -0.5, -0.5, 1.0],# Ε
    [-0.5, 0.5, -0.5, 1.0], # Ζ
    [0.5, 0.5, -0.5, 1.0],  # Η
    [0.5, -0.5, -0.5, 1.0],  # Θ
    
    [-0.5, -0.5, 0.5, 1.0], # Α
    [-0.5, 0.5, 0.5, 1.0],  # Β
    [0.5, 0.5, 0.5, 1.0],   # Γ
    [0.5, -0.5, 0.5, 1.0],  # Δ
    [-0.5, -0.5, -0.5, 1.0],# Ε
    [-0.5, 0.5, -0.5, 1.0], # Ζ
    [0.5, 0.5, -0.5, 1.0],  # Η
    [0.5, -0.5, -0.5, 1.0],  # Θ
    
    [-0.5, -0.5, 0.5, 1.0], # Α
    [-0.5, 0.5, 0.5, 1.0],  # Β
    [0.5, 0.5, 0.5, 1.0],   # Γ
    [0.5, -0.5, 0.5, 1.0],  # Δ
    [-0.5, -0.5, -0.5, 1.0],# Ε
    [-0.5, 0.5, -0.5, 1.0], # Ζ
    [0.5, 0.5, -0.5, 1.0],  # Η
    [0.5, -0.5, -0.5, 1.0]  # Θ    24
],dtype=np.float32) 

colorCube1 = np.array([
    [1.0, 0.0, 0.0, 1.0], #0-
    [1.0, 0.0, 0.0, 1.0], #1-
    [1.0, 0.0, 0.0, 1.0], #2-
    [1.0, 0.0, 0.0, 1.0], #3-
    [1.0, 1.0, 0.0, 1.0], #4-
    [1.0, 1.0, 1.0, 1.0], #5-
    [0.0, 0.0, 1.0, 1.0], #6-
    [0.0, 0.0, 1.0, 1.0], #7-
    
    [1.0, 1.0, 0.0, 1.0], #8-
    [1.0, 1.0, 1.0, 1.0], #9-
    [0.0, 0.0, 1.0, 1.0], #10-
    [0.0, 0.0, 1.0, 1.0], #11-
    [1.0, 0.64, 0.0, 1.0], #12-
    [1.0, 0.64, 0.0, 1.0], #13-
    [1.0, 1.0, 1.0, 1.0], #14-
    [1.0, 1.0, 0.0, 1.0], #15-
    
    [0.0, 1.0, 0.0, 1.0], #16
    [0.0, 1.0, 0.0, 1.0], #17
    [1.0, 1.0, 1.0, 1.0], #18-
    [1.0, 1.0, 0.0, 1.0], #19-
    [0.0, 1.0, 0.0, 1.0], #20-
    [0.0, 1.0, 0.0, 1.0], #21-
    [1.0, 0.64, 0.0, 1.0], #22-
    [1.0, 0.64, 0.0, 1.0], #23-
    
], dtype=np.float32)

vertexPyramid = np.array([
    [-0.5, 0.0, 0.5, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, 0.0, -0.5, 1.0],
    [-0.5, 0.0, -0.5, 1.0], 
    [0.0, 1.5, 0.0, 1.0]
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
indexCube1 = np.array((1,0,3, 1,3,2, # μπροστα
                  10,11,7, 10,7,6,      # δεξια
                  19,8,4, 19,4,15,      # κατω
                  14,5,9, 14,9,18,      # πανω
                  12,13,22, 12,22,23,      # πισω
                  21,20,16, 21,16,17), np.uint32) #rhombus out of two triangles  #  αριστερα
indexPyramid = np.array((0,1,2, 0,2,3, 0,1,4, 1,2,4, 2,3,4, 3,0,4), np.uint32) #rhombus out of two triangles


# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

"""
test_renderCubeAxesTerrainEVENT
"""

## ADD CUBE ##
# attach a simple cube in a RenderMesh so that VertexArray can pick it up
mesh4_c1.vertex_attributes.append(vertexCube1)
mesh4_c1.vertex_attributes.append(colorCube1)
mesh4_c1.vertex_index.append(indexCube1)
vArray4 = scene.world.addComponent(node4_c1, VertexArray())
shaderDec4_c1 = scene.world.addComponent(node4_c1, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

mesh4_c2.vertex_attributes.append(vertexCube)
mesh4_c2.vertex_attributes.append(colorCube)
mesh4_c2.vertex_index.append(indexCube)
vArray4 = scene.world.addComponent(node4_c2, VertexArray())
shaderDec4_c2 = scene.world.addComponent(node4_c2, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

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

## ADD AXES ##
axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.identity()))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
axes_mesh.vertex_attributes.append(vertexAxes) 
axes_mesh.vertex_attributes.append(colorAxes)
axes_mesh.vertex_index.append(indexAxes)
axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES)) # note the primitive change

axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))


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
model_cube1 = util.scale(0.4) @ util.translate(0.0,0.0,0.0)
model_cube2 = util.scale(0.3) @ util.translate(0.0,1.0,0.0)
pyramid = util.scale(0.3) @ util.translate(0.0,1.0,0.0)


translate_values_cube1 = model_cube1[0][3], model_cube1[1][3], model_cube1[2][3]
rotate_values_cube1 = 0.0, 0.0, 0.0
scale_values_cube1 = 1.0, 1.0, 1.0

translate_values_cube2 = model_cube2[0][3], model_cube2[1][3], model_cube2[2][3]
rotate_values_cube2 = 0.0, 0.0, 0.0
scale_values_cube2 = 1.0, 1.0, 1.0

translate_values_pyramid = pyramid[0][3], pyramid[1][3], pyramid[2][3]
rotate_values_pyramid = 0.0, 0.0, 0.0
scale_values_pyramid = 1.0, 1.0, 1.0

state_axes = False
state_terrain = False
state_cube1 = False
state_cube2 = False
state_pyramid = False

#Bonus
translate_values_HOME = pyramid[0][3], pyramid[1][3], pyramid[2][3]
rotate_values_HOME = 0.0, 0.0, 0.0
scale_values_HOME = 1.0, 1.0, 1.0
    

while running:
    running = scene.render(running)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    
    ImGui.set_next_window_size(400.0, 500.0)
    ImGui.set_next_window_position(10., 10.)
    ImGui.begin("TRS", True)
    # Shows Simple text
    ImGui.text("Tranformatiom : ")

    # Goes to a new line
    ImGui.new_line()
    
    # Creates a simple slider for Translation Cube 1
    changed, translate_values_cube1 = ImGui.slider_float3("Translate Cube 1", translate_values_cube1[0], translate_values_cube1[1], translate_values_cube1[2], -2.0, 2.0, format("%.3f"))
    model_cube1[0][3] = translate_values_cube1[0]
    model_cube1[1][3] = translate_values_cube1[1]
    model_cube1[2][3] = translate_values_cube1[2]

    # Creates a simple slider for Translation Cube 2
    changed, translate_values_cube2 = ImGui.slider_float3("Translate Cube 2", translate_values_cube2[0], translate_values_cube2[1], translate_values_cube2[2], -2.0, 2.0, format("%.3f"))
    model_cube2[0][3] = translate_values_cube2[0]
    model_cube2[1][3] = translate_values_cube2[1]
    model_cube2[2][3] = translate_values_cube2[2]

    # Creates a simple slider for Translation Cube 2
    changed, translate_values_pyramid = ImGui.slider_float3("Translate Pyramid", translate_values_pyramid[0], translate_values_pyramid[1], translate_values_pyramid[2], -2.0, 2.0, format("%.3f"))
    pyramid[0][3] = translate_values_pyramid[0]
    pyramid[1][3] = translate_values_pyramid[1]
    pyramid[2][3] = translate_values_pyramid[2]
    
    
    # Creates a simple slider for Rotation Cube 1
    changed, rotate_values_cube1 = ImGui.slider_float3("Rotate Cube 1", rotate_values_cube1[0], rotate_values_cube1[1], rotate_values_cube1[2], -3.0, 3.0, format("%.3f"))
    model_cube1 = model_cube1 @ util.rotate((1, 0, 0), rotate_values_cube1[0])
    model_cube1 = model_cube1 @ util.rotate((0, 1, 0), rotate_values_cube1[1])
    model_cube1 = model_cube1 @ util.rotate((0, 0, 1), rotate_values_cube1[2])
    
    # Creates a simple slider for Rotation Cube 2
    changed, rotate_values_cube2 = ImGui.slider_float3("Rotate Cube 2", rotate_values_cube2[0], rotate_values_cube2[1], rotate_values_cube2[2], -3.0, 3.0, format("%.3f"))
    model_cube2 = model_cube2 @ util.rotate((1, 0, 0), rotate_values_cube2[0])
    model_cube2 = model_cube2 @ util.rotate((0, 1, 0), rotate_values_cube2[1])
    model_cube2 = model_cube2 @ util.rotate((0, 0, 1), rotate_values_cube2[2])
    
    # Creates a simple slider for Rotation Cube 2
    changed, rotate_values_pyramid = ImGui.slider_float3("Rotate Pyramid", rotate_values_pyramid[0], rotate_values_pyramid[1], rotate_values_pyramid[2], -3.0, 3.0, format("%.3f"))
    pyramid = pyramid @ util.rotate((1, 0, 0), rotate_values_pyramid[0])
    pyramid = pyramid @ util.rotate((0, 1, 0), rotate_values_pyramid[1])
    pyramid = pyramid @ util.rotate((0, 0, 1), rotate_values_pyramid[2])
    
    
    scale_values_cube1 = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_cube1 = ImGui.slider_float3("Scale Cube 1", scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2], 0.97, 1.03, format("%.3f"))
    model_cube1 = model_cube1 @ util.scale(scale_values_cube1[0], scale_values_cube1[1], scale_values_cube1[2])
    
    scale_values_cube2 = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_cube2 = ImGui.slider_float3("Scale Cube 2", scale_values_cube2[0], scale_values_cube2[1], scale_values_cube2[2], 0.97, 1.03, format("%.3f"))
    model_cube2 = model_cube2 @ util.scale(scale_values_cube2[0], scale_values_cube2[1], scale_values_cube2[2])
   
    scale_values_pyramid = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_pyramid = ImGui.slider_float3("Scale Pyramid", scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2], 0.97, 1.03, format("%.3f"))
    pyramid = pyramid @ util.scale(scale_values_pyramid[0], scale_values_pyramid[1], scale_values_pyramid[2])


    # !!!!!!!!   BONUS   !!!!!!!!!

    # Creates a simple slider for Translation Cube 2
    changed, translate_values_HOME = ImGui.slider_float3("Translate HOME", translate_values_HOME[0], translate_values_HOME[1], translate_values_HOME[2], -2.0, 2.0, format("%.3f"))
    model_cube2[0][3] += translate_values_HOME[0]
    model_cube2[1][3] += translate_values_HOME[1]
    model_cube2[2][3] += translate_values_HOME[2]
    pyramid[0][3] += translate_values_HOME[0]
    pyramid[1][3] += translate_values_HOME[1]
    pyramid[2][3] += translate_values_HOME[2]

    # Creates a simple slider for Rotation Cube 2
    changed, rotate_values_HOME = ImGui.slider_float3("Rotate HOME", rotate_values_HOME[0], rotate_values_HOME[1], rotate_values_HOME[2], -3.0, 3.0, format("%.3f"))
    pyramid = pyramid @ util.rotate((1, 0, 0), rotate_values_HOME[0])
    pyramid = pyramid @ util.rotate((0, 1, 0), rotate_values_HOME[1])
    pyramid = pyramid @ util.rotate((0, 0, 1), rotate_values_HOME[2])
    model_cube2 = model_cube2 @ util.rotate((1, 0, 0), rotate_values_HOME[0])
    model_cube2 = model_cube2 @ util.rotate((0, 1, 0), rotate_values_HOME[1])
    model_cube2 = model_cube2 @ util.rotate((0, 0, 1), rotate_values_HOME[2])

    scale_values_HOME = 1.0, 1.0, 1.0
    # Creates a simple slider for Scaling Cube 1
    changed, scale_values_HOME = ImGui.slider_float3("Scale HOME", scale_values_HOME[0], scale_values_HOME[1], scale_values_HOME[2], 0.97, 1.03, format("%.3f"))
    model_cube2 = model_cube2 @ util.scale(scale_values_HOME[0], scale_values_HOME[1], scale_values_HOME[2])
    pyramid = pyramid @ util.scale(scale_values_HOME[0], scale_values_HOME[1], scale_values_HOME[2])
    
    
    ImGui.separator()


    
    view =  gWindow._myCamera # updates view via the imgui
    mvp_cube1 = projMat @ view @ model_cube1
    mvp_cube2 = projMat @ view @ model_cube2
    mvp_pyramid = projMat @ view @ pyramid
    mvp_terrain_axes = projMat @ view @ model_terrain_axes

    # Goes to a new line
    ImGui.new_line()
        
    # Creates a checkbox
    changed, checkbox_axes = ImGui.checkbox("Show Axes", state_axes)
    if checkbox_axes is False:
        axes_shader.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_axes = False
    if checkbox_axes is True:
        axes_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
        state_axes = True
        
    # Creates a checkbox
    changed, checkbox_terrain = ImGui.checkbox("Show Terrain", state_terrain)
    if checkbox_terrain is False:
        terrain_shader.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_terrain = False
    if checkbox_terrain is True:
        terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
        state_terrain = True
        
    # Creates a checkbox
    changed, checkbox_cube1 = ImGui.checkbox("Show Cube 1", state_cube1)
    if checkbox_cube1 is False:
        shaderDec4_c1.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_cube1 = False
    if checkbox_cube1 is True:
        shaderDec4_c1.setUniformVariable(key='modelViewProj', value=mvp_cube1, mat4=True)
        state_cube1 = True
        
    # Creates a checkbox
    changed, checkbox_cube2 = ImGui.checkbox("Show Cube 2", state_cube2)
    if checkbox_cube2 is False:
        shaderDec4_c2.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_cube2 = False
    if checkbox_cube2 is True:
        shaderDec4_c2.setUniformVariable(key='modelViewProj', value=mvp_cube2, mat4=True)
        checkbox_cube2 = False
        state_cube2 = True
        
    # Creates a checkbox
    changed, checkbox_pyramid = ImGui.checkbox("Show Pyramid", state_pyramid)
    if checkbox_pyramid is False:
        shaderDec4_p.setUniformVariable(key='modelViewProj', value=0, mat4=True)
        state_pyramid = False
    if checkbox_pyramid is True:
        shaderDec4_p.setUniformVariable(key='modelViewProj', value=mvp_pyramid, mat4=True)
        state_pyramid = True
    ImGui.separator()
    
    ImGui.end()
    scene.render_post()
    
scene.shutdown()

