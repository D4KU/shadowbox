import bpy
import gpu
import mathutils
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
    def __init__(self, xtex, ytex, ztex):
        self._shader = gpu.types.GPUShader(_vertex_shader, _fragment_shader)
        self._xtex = gpu.texture.from_image(xtex)
        self._ytex = gpu.texture.from_image(ytex)
        self._ztex = gpu.texture.from_image(ztex)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), 'WINDOW', 'POST_VIEW')

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

    def _draw(self):
        xbatch = self._create_batch((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1))
        ybatch = self._create_batch((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))
        zbatch = self._create_batch((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))

        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        self._shader.bind()

        self._shader.uniform_sampler("image", self._xtex)
        xbatch.draw(self._shader)

        self._shader.uniform_sampler("image", self._ytex)
        ybatch.draw(self._shader)

        self._shader.uniform_sampler("image", self._ztex)
        zbatch.draw(self._shader)

        gpu.state.depth_mask_set(False)

    def _create_batch(self, *coords):
        mat = bpy.context.object.matrix_world
        pos = [mat @ mathutils.Vector(i) for i in coords]
        return batch_for_shader(self._shader, 'TRIS',
            {
                "pos": pos,
                "uv": ((0, 0), (1, 0), (0, 1), (1, 1)),
            },
            indices=((0, 1, 2), (1, 3, 2))
        )
