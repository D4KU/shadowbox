import bpy
import numpy as np
from . import utils
from . import choose_images


class UpdateMesh(bpy.types.Operator):
    bl_idname = "shadowbox.update_mesh"
    bl_label = "Update Mesh"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}

    _mesh = None

    @classmethod
    def poll(cls, context):
        if choose_images.ChooseImages.grid is None:
            cls.poll_message_set("No images chosen")
            return False
        return True

    @classmethod
    def _add_geometry(cls, verts, polys):
        mesh = cls._mesh
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

    def execute(self, context):
        if UpdateMesh._mesh:
            UpdateMesh._mesh.clear_geometry()
        else:
            UpdateMesh._mesh = bpy.data.meshes.new("shadowbox")

        sb = utils.SharedLib('core')
        a, b = sb.create_mesh(
            choose_images.ChooseImages.x,
            choose_images.ChooseImages.y,
            choose_images.ChooseImages.z,
            choose_images.ChooseImages.grid,
            choose_images.ChooseImages.gridres,
        )
        self._add_geometry(a, b)

        return {'FINISHED'}
