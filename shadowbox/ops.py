import bpy
import numpy as np
import sys
from . import utils


def _gather_images(cls, context):
    a = []
    for item in bpy.data.images:
        a.append((
            item.name_full,
            item.name,
            item.filepath,
        ))
    return a


class Shadowbox(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadow Box"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}

    img_x: bpy.props.EnumProperty(
        name="Image X",
        items=_gather_images,
        default=1,
    )
    img_y: bpy.props.EnumProperty(
        name="Image Y",
        items=_gather_images,
        default=2,
    )
    img_z: bpy.props.EnumProperty(
        name="Image Z",
        items=_gather_images,
        default=3,
    )

    @classmethod
    def poll(cls, context):
        a = context.mode == 'OBJECT'
        b = context.object is not None
        return a and b and context.object.type == 'MESH'

    @classmethod
    def _as_array(cls, name):
        img = bpy.data.images[name]
        pxs = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(img.size[0], img.size[1], -1)
        return pxs[:, :, 0]

    @classmethod
    def debug_log(cls, mul):
        np.set_printoptions(threshold=sys.maxsize)
        string = np.array2string(mul)
        f = open("D://a.txt", "w")
        f.write(string)
        f.close()

    @classmethod
    def set_mesh(cls, mesh, verts, polys):
        mesh.vertices.add(len(verts))
        mesh.vertices.foreach_set('co', verts.ravel())

        mesh.loops.add(polys.size)
        mesh.loops.foreach_set('vertex_index', polys.ravel())

        loop_start = np.arange(0, polys.size, polys.shape[1])
        loop_total = np.repeat(polys.shape[1], len(polys))

        mesh.polygons.add(len(polys))
        mesh.polygons.foreach_set('loop_start', loop_start)
        mesh.polygons.foreach_set('loop_total', loop_total)

        mesh.update()
        mesh.validate()

    def execute(self, context):
        x = self._as_array(self.img_x)
        y = self._as_array(self.img_y)
        z = self._as_array(self.img_z)

        if z.shape[0] != y.shape[0] or \
           z.shape[1] != x.shape[0] or \
           y.shape[1] != x.shape[1]:
            return {'CANCELLED'}

        sb = utils.SharedLib('core')
        mesh = bpy.data.meshes.new("123")
        self.set_mesh(mesh, *sb.shadowbox(x, y, z))

        if context.active_object:
            context.active_object.data = mesh

        return {'FINISHED'}
