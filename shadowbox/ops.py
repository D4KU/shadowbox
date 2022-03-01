import bpy
import bmesh
import numpy as np
import sys
from . import utils


class Shadowbox(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadow Box"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        a = context.mode == 'OBJECT'
        return a
        b = context.object is not None
        return a and b and context.object.type == 'MESH'

    @classmethod
    def preprocess(cls, name):
        img = bpy.data.images[name]
        arr = np.asanyarray(img.pixels[:])
        arr = arr.reshape(img.size[0], img.size[1], -1)
        arr = arr[:, :, 0]
        return arr

    @classmethod
    def debug_log(cls, mul):
        np.set_printoptions(threshold=sys.maxsize)
        string = np.array2string(mul)
        f = open("D://a.txt", "w")
        f.write(string)
        f.close()

    def execute(self, context):
        sb = utils.SharedLib('core')

        x = self.preprocess("x.png")
        y = self.preprocess("y.png")
        z = self.preprocess("z.png")

        (V, F) = sb.shadowbox(x, y, z)

        mesh = bpy.data.meshes.new("123")

        mesh.vertices.add(len(V))
        mesh.vertices.foreach_set('co', V.ravel())

        mesh.loops.add(F.size)
        mesh.loops.foreach_set('vertex_index', F.ravel())

        mesh.polygons.add(len(F))
        mesh.polygons.foreach_set('loop_start', np.arange(0, F.size, F.shape[1]))
        mesh.polygons.foreach_set('loop_total', np.repeat(F.shape[1], len(F)))

        mesh.update()
        mesh.validate()

        if context.active_object:
            context.active_object.data = mesh

        return {'FINISHED'}
