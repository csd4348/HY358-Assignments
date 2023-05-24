import pyECSS.utilities as util
import random
import math
import numpy as np
from statistics import mode
from turtle import width
# import unittesτ
import numpy as np

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

Cubes = []

class Object:
    
    def __init__(self, node, trans, mesh):
        self.node = node;
        self.trans = trans;
        self.mesh = mesh;
    
    def setShader(self, value):
        self.shader = value;
        
def addEntity(name_node, name_trans, name_mesh):
    node4 = scene.world.createEntity(Entity(name=name_node))
    scene.world.addEntityChild(rootEntity, node4)
    trans4 = scene.world.addComponent(node4, BasicTransform(name=name_trans, trs=util.identity()))
    mesh4 = scene.world.addComponent(node4, RenderMesh(name=name_mesh))
    
    obj = Object(node4, trans4, mesh4)
    return obj

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


for i in range(1,82):
    Cubes.append(addEntity("node4_{}".format(i), "trans4_{}".format(i), "mesh4_{}".format(i)))


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
    
    [0.0, 1.0, 0.0, 1.0], #16-
    [0.0, 1.0, 0.0, 1.0], #17-
    [1.0, 1.0, 1.0, 1.0], #18-
    [1.0, 1.0, 0.0, 1.0], #19-
    [0.0, 1.0, 0.0, 1.0], #20-
    [0.0, 1.0, 0.0, 1.0], #21-
    [1.0, 0.64, 0.0, 1.0], #22-
    [1.0, 0.64, 0.0, 1.0], #23-
], dtype=np.float32)

#index arrays for above vertex Arrays
index = np.array((0,1,2), np.uint32) #simple triangle
indexAxes = np.array((0,1,2,3,4,5), np.uint32) #3 simple colored Axes as R,G,B lines
indexCube = np.array((1,0,3, 1,3,2, # μπροστα
                  10,11,7, 10,7,6,      # δεξια
                  19,8,4, 19,4,15,      # κατω
                  14,5,9, 14,9,18,      # πανω
                  12,13,22, 12,22,23,      # πισω
                  21,20,16, 21,16,17), np.uint32) #rhombus out of two triangles  #  αριστερα

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

for c in Cubes:
    c.mesh.vertex_attributes.append(vertexCube1)
    c.mesh.vertex_attributes.append(colorCube1)
    c.mesh.vertex_index.append(indexCube)
    vArray4 = scene.world.addComponent(c.node, VertexArray())
    c.setShader(scene.world.addComponent(c.node, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG))))

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
axes_mesh.vertex_index.append(indexAxes)
axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES)) # note the primitive change
axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

running = True
# MAIN RENDERING LOOP
scene.init(imgui=True, windowWidth = 1024, windowHeight = 768, windowTitle = "pyglGA test_renderAxesTerrainEVENT")

# pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
# needs an active GL context

#vArrayAxes.primitive = gl.GL_LINES

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

eye = util.vec(4.2, 2.0, 4.2)
target = util.vec(0.0, 1.2, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
# projMat = util.ortho(-10.0, 10.0, -10.0, 10.0, -1.0, 10.0) ## WORKING
# projMat = util.perspective(90.0, 1.33, 0.1, 100) ## WORKING
projMat = util.perspective(50.0, 1.0, 1.0, 10.0) ## WORKING 

gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

class points:
    def __init__(self,x,y,z):
        self.x = x;
        self.y = y;
        self.z = z;
        
ArrayOfPoints = [
    points(-1.0,1.0,0.0),
    points(0.0,1.0,0.0),
    points(1.0,1.0,0.0),
    points(-1.0,0.0,0.0),
    points(0.0,0.0,0.0),
    points(1.0,0.0,0.0),
    points(-1.0,-1.0,0.0),
    points(0.0,-1.0,0.0),
    points(1.0,-1.0,0.0),
    points(-1.0,1.0,1.0),
    points(0.0,1.0,1.0),
    points(1.0,1.0,1.0),
    points(-1.0,0.0,1.0),
    points(0.0,0.0,1.0),
    points(1.0,0.0,1.0),
    points(-1.0,-1.0,1.0),
    points(0.0,-1.0,1.0),
    points(1.0,-1.0,1.0),
    points(-1.0,1.0,-1.0),
    points(0.0,1.0,-1.0),
    points(1.0,1.0,-1.0),
    points(-1.0,0.0,-1.0),
    points(0.0,0.0,-1.0),
    points(1.0,0.0,-1.0),
    points(-1.0,-1.0,-1.0),
    points(0.0,-1.0,-1.0),
    points(1.0,-1.0,-1.0)
]

Model_Cubes = []

model_terrain_axes = util.translate(0.0,0.0,0.0)
for i in range(len(ArrayOfPoints)):
    Model_Cubes.append(util.scale(0.3) @ util.translate(ArrayOfPoints[i].x, ArrayOfPoints[i].y, ArrayOfPoints[i].z) @ util.translate(0.0,0.0,5.0))
for i in range(len(ArrayOfPoints)):
    Model_Cubes.append(util.scale(0.3) @ util.translate(ArrayOfPoints[i].x, ArrayOfPoints[i].y, ArrayOfPoints[i].z) @ util.translate(5.0,0.0,0.0))
for i in range(len(ArrayOfPoints)):
    Model_Cubes.append(util.scale(0.3) @ util.translate(ArrayOfPoints[i].x, ArrayOfPoints[i].y, ArrayOfPoints[i].z) @ util.translate(0.0,5.0,0.0))

Mvp_Cubes = [0]*81

while running:
    running = scene.render(running)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    view =  gWindow._myCamera # updates view via the imgui
    
    for i in range(0,9):
        Model_Cubes[i] = util.rotate((0, 0, 1), 2) @ Model_Cubes[i]
    
    for i in range(9,18):
        Model_Cubes[i] = util.rotate((0, 0, 1), -2) @ Model_Cubes[i]
    
    for i in range(18,27):
        Model_Cubes[i] = util.rotate((0, 0, 1), 4) @ Model_Cubes[i]
        
    i = 27
    while(i < 52) :
        Model_Cubes[i] = util.rotate((1, 0, 0), 2) @ Model_Cubes[i]
        i += 3
        
    i = 28
    while(i < 53) :
        Model_Cubes[i] = util.rotate((1, 0, 0), -2) @ Model_Cubes[i]
        i += 3
        
    i = 29
    while(i < 54) :
        Model_Cubes[i] = util.rotate((1, 0, 0), 4) @ Model_Cubes[i]
        i += 3
    
    i = 54
    j = 0
    while(i < 75) :
        Model_Cubes[i] = util.rotate((0, 1, 0), 4) @ Model_Cubes[i]
        i += 1
        j += 1
        if(j == 3): 
            i += 6
            j = 0
    
    i = 57
    j = 0
    while(i < 78) :
        Model_Cubes[i] = util.rotate((0, 1, 0), -4) @ Model_Cubes[i]
        i += 1
        j += 1
        if(j == 3): 
            i += 6
            j = 0
    
    i = 60
    j = 0
    while(i < 81) :
        Model_Cubes[i] = util.rotate((0, 1, 0), 1) @ Model_Cubes[i]
        i += 1
        j += 1
        if(j == 3): 
            i += 6
            j = 0
    
    for i in range(len(Model_Cubes)):
        Mvp_Cubes[i]= projMat @ view @ Model_Cubes[i];

    mvp_terrain_axes = projMat @ view @ model_terrain_axes
    axes_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
    terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain_axes, mat4=True)
    for i in range(len(Mvp_Cubes)):
        Cubes[i].shader.setUniformVariable(key='modelViewProj', value=Mvp_Cubes[i], mat4=True)
    scene.render_post()
    
scene.shutdown()
