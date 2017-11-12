

import numpy as np
from vispy import app, gloo

import OpenGL.GL as gl

app.use_app('pyglet')   # Set backend

_vertex_code_colored = """
uniform mat4 u_mv; 
uniform mat4 u_mvp; 
uniform vec3 u_light_eye_pos; 
 
attribute vec3 a_position; 
attribute vec3 a_color; 
 
varying vec3 v_color; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
 
void main() { 
    gl_Position = u_mvp * vec4(a_position, 1.0); 
    v_color = a_color; 
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates 
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light 
} 
"""

_fragment_code_colored = """
uniform float u_light_ambient_w; 
varying vec3 v_color; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
 
void main() { 
    // Face normal in eye coordinates 
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos))); 
 
    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0); 
    float light_w = u_light_ambient_w + 0.5 * light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0; 
    gl_FragColor = vec4(light_w * v_color, 1.0); 
} 
"""

_vertex_code_textured = """
uniform mat4 u_mv; 
uniform mat4 u_mvp; 
uniform vec3 u_light_eye_pos; 

attribute vec3 a_position; 
attribute vec2 a_texcoord; 

varying vec2 v_texcoord; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 

void main() { 
    gl_Position = u_mvp * vec4(a_position, 1.0); 
    v_texcoord = a_texcoord;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates 
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light 
} 
"""

_fragment_code_textured = """
uniform float u_light_ambient_w; 
uniform sampler2D u_tex;

varying vec2 v_texcoord; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 

void main() { 
    // Face normal in eye coordinates 
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos))); 

    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0); 
    float light_w = u_light_ambient_w + 0.5 * light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0; 
    gl_FragColor = texture2D(u_tex, v_texcoord) * light_w;
} 
"""


def singleton(cls):
    instances = {}

    def get_instance(size, cam):
        if cls not in instances:
            instances[cls] = cls(size, cam)
        return instances[cls]
    return get_instance


@singleton  # Don't throw GL context into trash when having more than one Renderer instance
class Renderer(app.Canvas):

    def __init__(self, size, cam):

        app.Canvas.__init__(self, show=False, size=size)
        self.shape = (size[1], size[0])
        self.yz_flip = np.eye(4, dtype=np.float32)
        self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1

        self.set_cam(cam)

        # Set up shader programs
        self.program_col = gloo.Program(_vertex_code_colored, _fragment_code_colored)
        self.program_tex = gloo.Program(_vertex_code_textured, _fragment_code_textured)

        # Texture where we render the color/depth and its FBO
        self.col_tex = gloo.Texture2D(shape=self.shape + (3,))
        self.fbo = gloo.FrameBuffer(self.col_tex, gloo.RenderBuffer(self.shape))
        self.fbo.activate()
        gloo.set_state(depth_test=True, blend=False, cull_face=True)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gloo.set_clear_color((0.0, 0.0, 0.0))
        gloo.set_viewport(0, 0, *self.size)

    def set_cam(self, cam, clip_near=0.01, clip_far=10.0):
        self.cam = cam
        self.clip_near = clip_near
        self.clip_far = clip_far
        self.mat_proj = self.build_projection(cam, 0, 0,
                                              self.shape[1], self.shape[0],
                                              clip_near, clip_far)


    def clear(self):
        gloo.clear(color=True, depth=True)

    def finish(self):

        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_FLOAT)
        rgb = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape+(3,))[::-1, :]  # Read buffer and flip Y
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        dep = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape+(1,))[::-1, :]  # Read buffer and flip Y

        # Convert z-buffer to depth map
        mult = (self.clip_near*self.clip_far)/(self.clip_near-self.clip_far)
        addi = self.clip_far/(self.clip_near-self.clip_far)
        bg = dep == 1
        dep = mult/(dep + addi)
        dep[bg] = 0
        return rgb, np.squeeze(dep)

    def draw_model(self, model, pose, ambient_weight=0.5, light=(0, 0, 0)):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T    # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        used_program = self.program_col
        if model.texcoord is not None:
            used_program = self.program_tex
            used_program['u_tex'] = model.texture

        used_program.bind(model.vertex_buffer)
        used_program['u_light_eye_pos'] = light
        used_program['u_light_ambient_w'] = ambient_weight
        used_program['u_mv'] = mv
        used_program['u_mvp'] = mvp
        used_program.draw('triangles', model.index_buffer)

    def draw_boundingbox(self, model, pose):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        self.program_col.bind(model.bb_vbuffer)
        self.program_col['u_light_eye_pos'] = (0, 0, 0)
        self.program_col['u_light_ambient_w'] = 1
        self.program_col['u_mv'] = mv
        self.program_col['u_mvp'] = mvp
        self.program_col.draw('lines', model.bb_ibuffer)

    def build_projection(self, cam, x0, y0, w, h, nc, fc):

        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        proj = np.array([
            [2 * cam[0, 0] / w, -2 * cam[0, 1] / w, (-2 * cam[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * cam[1, 1] / h, (-2 * cam[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ])

        # Compensate for the flipped image
        proj[1, :] *= -1.0
        return proj.T

    def compute_metrical_clip(self, pose, diameter):

        width = self.cam[0, 0] * diameter / pose[2, 3]  # X coordinate == shape[1]
        height = self.cam[1, 1] * diameter / pose[2, 3]  # Y coordinate == shape[0]
        proj = np.matmul(self.cam, pose[0:3, 3])
        proj /= proj[2]
        cut = np.asarray([proj[1] - height//2, proj[0] - width//2, proj[1] + height//2, proj[0] + width//2], dtype=int)

        # Can lead to offsetted extractions, not really nice...
        cut[0] = np.clip(cut[0], 0, self.shape[0])
        cut[2] = np.clip(cut[2], 0, self.shape[0])
        cut[1] = np.clip(cut[1], 0, self.shape[1])
        cut[3] = np.clip(cut[3], 0, self.shape[1])
        return cut

    def render_view_metrical_clip(self, model, pose, diameter):

        cut = self.compute_metrical_clip(pose, diameter)
        self.clear()
        self.draw_model(model, pose)
        col, dep = self.finish()
        return col[cut[0]:cut[2], cut[1]:cut[3]], dep[cut[0]:cut[2], cut[1]:cut[3]]
