import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from functools import partial


_vertex_shader = '''
uniform mat4 ModelViewProjectionMatrix;

#ifdef USE_WORLD_CLIP_PLANES
uniform mat4 ModelMatrix;
#endif

in vec3 pos;
in vec2 uv;
out vec2 uv_interp;

void main()
{
  gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
  uv_interp = uv;

#ifdef USE_WORLD_CLIP_PLANES
  world_clip_planes_calc_clip_distance((ModelMatrix * vec4(pos, 1.0)).xyz);
#endif
}
'''


_fragment_shader = '''
uniform sampler2D image;

in vec2 uv_interp;
out vec4 frag;

void main()
{
  frag = texture(image, uv_interp);
}
'''


class ImageHandle:
    def __init__(self):
        shader = gpu.types.GPUShader(_vertex_shader, _fragment_shader)
        self._handle = None
        self._shader = shader
        self._xbatch = self._create_batch((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1))
        self._ybatch = self._create_batch((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))
        self._zbatch = self._create_batch((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))

    def _create_batch(self, *coords):
        return batch_for_shader(self._shader, 'TRIS',
            {
                "pos": coords,
                "uv": ((0, 0), (1, 0), (0, 1), (1, 1)),
            },
            indices=((0, 1, 2), (1, 3, 2))
        )

    @staticmethod
    def _draw(xtex, ytex, ztex, shader, xbatch, ybatch, zbatch):
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        shader.bind()

        shader.uniform_sampler("image", xtex)
        xbatch.draw(shader)

        shader.uniform_sampler("image", ytex)
        ybatch.draw(shader)

        shader.uniform_sampler("image", ztex)
        zbatch.draw(shader)

        gpu.state.depth_mask_set(False)

    def __del__(self):
        self.clear()

    def clear(self):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

    def set_images(self, xtex, ytex, ztex):
        func = partial(
            self._draw,
            gpu.texture.from_image(xtex),
            gpu.texture.from_image(ytex),
            gpu.texture.from_image(ztex),
            self._shader,
            self._xbatch,
            self._ybatch,
            self._zbatch,
            )

        self.clear()
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            func, (), 'WINDOW', 'POST_VIEW')
