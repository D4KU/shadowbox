import bpy
import numpy as np
from functools import partial
from . import utils
from . import choose_images


class UpdateMesh(bpy.types.Operator):
    bl_idname = "shadowbox.update_mesh"
    bl_label = "Update Mesh"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}

    iso: bpy.props.FloatProperty(
        name="Iso",
        min=0,
        max=1,
        step=2,
        default=0,
    )

    @classmethod
    def poll(cls, context):
        if choose_images.ChooseImages.x is None:
            cls.poll_message_set("No images chosen")
            return False
        return True

    @classmethod
    def _add_geometry(cls, mesh, verts, polys):
        mesh.clear_geometry()
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

    # def on_depsgraph_update(self, scene, depsgraph):
    #     self._execute(bpy.context)

    def _execute(self, context):
        mesh = bpy.data.meshes["shadowbox"]
        sb = utils.SharedLib('core')
        a, b = sb.create_mesh(
            choose_images.ChooseImages.x,
            choose_images.ChooseImages.y,
            choose_images.ChooseImages.z,
            self.iso,
        )
        self._add_geometry(mesh, a, b)

    def execute(self, context):
        try:
            mesh = bpy.data.meshes["shadowbox"]
        except KeyError:
            mesh = bpy.data.meshes.new("shadowbox")

        self._execute(context)

        ob = context.object
        if ob and ob.type == 'MESH':
            ob.data = mesh

        # handlers = bpy.app.handlers.depsgraph_update_post
        # if (handlers.index(self.on_depsgraph_update) < 0):
        #     func = partial(self.on_depsgraph_update, self)
        #     handlers.append(func)

        return {'FINISHED'}
