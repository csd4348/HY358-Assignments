from statistics import mode
from turtle import width
# import unittest

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

node3_p1 = scene.world.createEntity(Entity(name="node3_p1"))
scene.world.addEntityChild(rootEntity, node3_p1)
trans4_c2 = scene.world.addComponent(node3_p1, BasicTransform(name="trans4_c2", trs=util.identity()))
mesh4_p1 = scene.world.addComponent(node3_p1, RenderMesh(name="mesh4_c2"))

node4_p2 = scene.world.createEntity(Entity(name="node4_p2"))
scene.world.addEntityChild(rootEntity, node4_p2)
trans4_p = scene.world.addComponent(node4_p2, BasicTransform(name="trans4_p", trs=util.identity()))
mesh4_p2 = scene.world.addComponent(node4_p2, RenderMesh(name="mesh4_p"))

#Simple Cube
vertexCube = np.array([
    [-0.5, -0.5, 0.5, 1.0], # Α
    [-0.5, 0.5, 0.5, 1.0],  # Β
    [0.5, 0.5, 0.5, 1.0],   # Γ
    [0.5, -0.5, 0.5, 1.0],  # Δ
    [-0.5, -0.5, -0.5, 1.0],# Ε
    [-0.5, 0.5, -0.5, 1.0], # Ζ
    [0.5, 0.5, -0.5, 1.0],  # Η
    [0.5, -0.5, -0.5, 1.0]  # Θ
],dtype=np.float32)

colorCube = np.array([
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0],
    [0.0, 0.7, 0.7, 1.0]
], dtype=np.float32)

vertexPyramid = np.array([
    [-0.5, 0.0, 0.5, 1.0],
    [0.5, 0.0, 0.5, 1.0],
    [0.5, 0.0, -0.5, 1.0],
    [-0.5, 0.0, -0.5, 1.0], 
    [0.0, 1.5, 0.0, 1.0]
],dtype=np.float32) 

colorPyramid = np.array([
    [0.7, 0.6, 0.0, 1.0],
    [0.7, 0.6, 0.0, 1.0],
    [0.7, 0.6, 0.0, 1.0],
    [0.7, 0.6, 0.0, 1.0], 
    [0.7, 0.6, 0.0, 1.0]
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


# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())



MY_VERT_PHONG_MVP = """
        #version 410

        layout (location=0) in vec4 vPosition;
        layout (location=1) in vec4 vColor;
        layout (location=2) in vec4 vNormal;

        out     vec4 pos;
        out     vec4 color;
        out     vec3 normal;
        
        uniform mat4 modelViewProj;
        uniform mat4 model;
        uniform vec4 my_color;

        void main()
        {
            gl_Position = modelViewProj * vPosition;
            pos = model * vPosition;
            color = my_color;
            normal = mat3(transpose(inverse(model))) * vNormal.xyz;
        }
"""
    
MY_FRAG_PHONG = """
    #version 410

    in vec4 pos;
    in vec4 color;
    in vec3 normal;

    out vec4 outputColor;

    // Phong products
    uniform vec3 ambientColor;
    uniform float ambientStr;

    // Lighting 
    uniform vec3 viewPos;
    uniform vec3 lightPos;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Material
    uniform float shininess;

    void main()
    {
        vec3 norm = normalize(normal);
        vec3 lightDir = normalize(lightPos - pos.xyz);
        vec3 viewDir = normalize(viewPos - pos.xyz);
        vec3 reflectDir = reflect(-lightDir, norm);
        

        // Ambient
        vec3 ambientProduct = ambientStr * ambientColor;
        // Diffuse
        float diffuseStr = max(dot(norm, lightDir), 0.0);
        vec3 diffuseProduct = diffuseStr * lightColor;
        // Specular
        float specularStr = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specularProduct = shininess * specularStr * color.xyz;
        
        vec3 result = (ambientProduct + (diffuseProduct + specularProduct) * lightIntensity) * color;
        outputColor = vec4(result, 1);
    }
"""


MY_VERT_PHONG_MVP_PYR1 = """
        #version 410

        layout (location=0) in vec4 vPosition;
        layout (location=1) in vec4 vColor;
        layout (location=2) in vec4 vNormal;

        out     vec4 pos;
        out     vec4 color;
        out     vec3 normal;
        
        uniform mat4 modelViewProj;
        uniform mat4 model;
        uniform vec4 my_color;

        void main()
        {
            gl_Position = modelViewProj * vPosition;
            pos = model * vPosition;
            color = my_color;
            normal = mat3(transpose(inverse(model))) * vNormal.xyz;
        }
"""
    
MY_FRAG_PHONG_PYR1 = """
    #version 410

    in vec4 pos;
    in vec4 color;
    in vec3 normal;

    out vec4 outputColor;

    // Phong products
    uniform vec3 ambientColor;
    uniform float ambientStr;

    // Lighting 
    uniform vec3 viewPos;
    uniform vec3 lightPos;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Material
    uniform float shininess;

    void main()
    {
        vec3 norm = normalize(normal);
        vec3 lightDir = normalize(lightPos - pos.xyz);
        vec3 viewDir = normalize(viewPos - pos.xyz);
        vec3 reflectDir = reflect(-lightDir, norm);
        

        // Ambient
        vec3 ambientProduct = ambientStr * ambientColor;
        // Diffuse
        float diffuseStr = max(dot(norm, lightDir), 0.0);
        vec3 diffuseProduct = diffuseStr * lightColor;
        // Specular
        float specularStr = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specularProduct = shininess * specularStr * color.xyz;
        
        vec3 result = (ambientProduct + (diffuseProduct + specularProduct) * lightIntensity) * color;
        outputColor = vec4(result, 1);
    }
"""
MY_VERT_PHONG_MVP_PYR2 = """
        #version 410

        layout (location=0) in vec4 vPosition;
        layout (location=1) in vec4 vColor;
        layout (location=2) in vec4 vNormal;

        out     vec4 pos;
        out     vec4 color;
        out     vec3 normal;
        
        uniform mat4 modelViewProj;
        uniform mat4 model;
        uniform vec4 my_color;

        void main()
        {
            gl_Position = modelViewProj * vPosition;
            pos = model * vPosition;
            color = my_color;
            normal = mat3(transpose(inverse(model))) * vNormal.xyz;
        }
"""
    
MY_FRAG_PHONG_PYR2 = """
    #version 410

    in vec4 pos;
    in vec4 color;
    in vec3 normal;

    out vec4 outputColor;

    // Phong products
    uniform vec3 ambientColor;
    uniform float ambientStr;

    // Lighting 
    uniform vec3 viewPos;
    uniform vec3 lightPos;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Material
    uniform float shininess;

    void main()
    {
        vec3 norm = normalize(normal);
        vec3 lightDir = normalize(lightPos - pos.xyz);
        vec3 viewDir = normalize(viewPos - pos.xyz);
        vec3 reflectDir = reflect(-lightDir, norm);
        

        // Ambient
        vec3 ambientProduct = ambientStr * ambientColor;
        // Diffuse
        float diffuseStr = max(dot(norm, lightDir), 0.0);
        vec3 diffuseProduct = diffuseStr * lightColor;
        // Specular
        float specularStr = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specularProduct = shininess * specularStr * color.xyz;
        
        vec3 result = (ambientProduct + (diffuseProduct + specularProduct) * lightIntensity) * color;
        outputColor = vec4(result, 1);
    }
"""

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
            print(vertexArray)
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


NormalsCube = []
NormalsCube = Normals(vertexCube, 1)
NormalsPyramid1 = []
NormalsPyramid1 = Normals(vertexPyramid, 0)
NormalsPyramid2 = []
NormalsPyramid2 = Normals(vertexPyramid, 0)

## ADD CUBE ##
# attach a simple cube in a RenderMesh so that VertexArray can pick it up
mesh4_c1.vertex_attributes.append(vertexCube)
mesh4_c1.vertex_attributes.append(colorCube)
mesh4_c1.vertex_attributes.append(NormalsCube)
mesh4_c1.vertex_index.append(indexCube)
vArray4 = scene.world.addComponent(node4_c1, VertexArray())
#shaderDec4_c1 = scene.world.addComponent(node4_c1, ShaderGLDecorator(Shader(vertex_source = COLOR_VERT_MVP_MANOS, fragment_source=Shader.COLOR_FRAG)))
shaderDec4_c1 = scene.world.addComponent(node4_c1, ShaderGLDecorator(Shader(vertex_source = Shader.VERT_PHONG_MVP, fragment_source=Shader.FRAG_PHONG)))

mesh4_p1.vertex_attributes.append(vertexPyramid)
mesh4_p1.vertex_attributes.append(colorPyramid)
mesh4_c1.vertex_attributes.append(NormalsPyramid1)
mesh4_p1.vertex_index.append(indexPyramid)
vArray4 = scene.world.addComponent(node3_p1, VertexArray())
shaderDec4_p1 = scene.world.addComponent(node3_p1, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

mesh4_p2.vertex_attributes.append(vertexPyramid)
mesh4_p2.vertex_attributes.append(colorPyramid)
mesh4_c1.vertex_attributes.append(NormalsPyramid2)
mesh4_p2.vertex_index.append(indexPyramid)
vArray4 = scene.world.addComponent(node4_p2, VertexArray())
shaderDec4_p2 = scene.world.addComponent(node4_p2, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))


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

eye = util.vec(2.5, 1.2, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
# projMat = util.ortho(-10.0, 10.0, -10.0, 10.0, -1.0, 10.0) ## WORKING
# projMat = util.perspective(90.0, 1.33, 0.1, 100) ## WORKING
projMat = util.perspective(50.0, 1.0, 1.0, 10.0) ## WORKING 

gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

model_cube = util.scale(0.4) @ util.translate(0.0,0.5,0.0)
pyramid1 = util.scale(0.4) @ util.translate(-1.2,0.0,1.5)
pyramid2 = util.scale(0.4) @ util.translate(1.5,0.0,-1.2)

color_cube_values = 1.0, 1.0, 1.0
color_pyramid1_values = 1.0, 1.0, 1.0
color_pyramid2_values = 1.0, 1.0, 1.0

viewPos = 2.0,2.0,2.0
lightPos = 1.5,1.5,1.5
lightColor = 1.0,1.0,1.0
lightIntensity = 100.0
ambientColor = 0.5,0.5,0.5
ambientStr = 10.0

ambientCube = 0.2, 0.2, 0.2, 1.0
diffuseCube = 1.0, 0.8, 0.0, 1.0
specularCube = 1.0, 1.0, 1.0, 1.0
shineCube = 100.0

ambientPyramid1 = 0.2, 0.2, 0.2, 1.0
diffusePyramid1 = 1.0, 0.8, 0.0, 1.0
specularPyramid1 = 1.0, 1.0, 1.0, 1.0
shinePyramid1 = 100.0

ambientPyramid2 = 0.2, 0.2, 0.2, 1.0
diffusePyramid2 = 1.0, 0.8, 0.0, 1.0
specularPyramid2 = 1.0, 1.0, 1.0, 1.0
shinePyramid2 = 100.0

while running:
    running = scene.render(running)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    
    ImGui.set_next_window_size(400.0, 450.0)
    ImGui.set_next_window_position(10., 10.)
    ImGui.begin("TRS", True)
    # Shows Simple text
    ImGui.text("Tranformatiom : ")
        
    view =  gWindow._myCamera # updates view via the imgui
    mvp_cube = projMat @ view @ model_cube
    mvp_pyramid1 = projMat @ view @ pyramid1
    mvp_pyramid2 = projMat @ view @ pyramid2
    # Goes to a new line
    ImGui.new_line()
        
    shaderDec4_c1.setUniformVariable(key='model', value=model_cube, mat4=True)
    shaderDec4_c1.setUniformVariable(key='modelViewProj', value=mvp_cube, mat4=True)
    shaderDec4_c1.setUniformVariable(key='viewPos', value=viewPos, float3=True)
    
    changed, lightPos = ImGui.slider_float3("Light Position", lightPos[0], lightPos[1], lightPos[2], 0.0, 1.0)
    if changed : 
        shaderDec4_c1.setUniformVariable(key='lightPos', value=lightPos, float3=True)
        
    shaderDec4_c1.setUniformVariable(key='lightColor', value=lightColor, float3=True)
    
    changed, lightIntensity = ImGui.slider_float("light Intensity", lightIntensity, 0.0, 100.0)
    if changed : 
        shaderDec4_c1.setUniformVariable(key='lightIntensity', value=lightIntensity, float1=True)
        
    changed, ambientColor = ImGui.slider_float3("Ambient Color", ambientColor[0], ambientColor[1], ambientColor[2], 0.0, 1.0)
    if changed : 
        shaderDec4_c1.setUniformVariable(key='ambientColor', value=ambientColor, float3=True)
    
    changed, ambientStr = ImGui.slider_float("Ambient Str", ambientStr, 0.0, 100.0)
    if changed : 
        shaderDec4_c1.setUniformVariable(key='ambientStr', value=ambientStr, float1=True)
    
    changed, color_cube_values = ImGui.color_edit3("Color Cube", color_cube_values[0], color_cube_values[1], color_cube_values[2])
    if changed:
        shaderDec4_c1.setUniformVariable(key='my_color', value=color_cube_values, float3=True)
    
    
    shaderDec4_p1.setUniformVariable(key='modelViewProj', value=mvp_pyramid1, mat4=True)
    shaderDec4_p2.setUniformVariable(key='modelViewProj', value=mvp_pyramid2, mat4=True)
    
    changed, color_pyramid1_values = ImGui.color_edit3("Color Pyramid1", color_pyramid1_values[0], color_pyramid1_values[1], color_pyramid1_values[2])
    if changed:
        shaderDec4_p1.setUniformVariable(key='my_color', value=color_pyramid1_values, float3=True)
    
    changed, color_pyramid2_values = ImGui.color_edit3("Color Pyramid2", color_pyramid2_values[0], color_pyramid2_values[1], color_pyramid2_values[2])
    if changed:
        shaderDec4_p2.setUniformVariable(key='my_color', value=color_pyramid2_values, float3=True)
   
    ImGui.separator()
    ImGui.end()
    scene.render_post()
    
scene.shutdown()