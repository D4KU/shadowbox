import bpy
import numpy as np
from . import utils
from . import state


class UpdateMesh(bpy.types.Operator):
    bl_idname = "shadowbox.update_mesh"
    bl_label = "Update Mesh"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}
    _mesh_name = "shadowbox"


    iso: bpy.props.FloatProperty(
        name="Iso",
        min=0,
        max=1,
        step=2,
        default=0,
    )

    @classmethod
    def poll(cls, context):
        if state.arr_x is None:
            cls.poll_message_set("No images chosen")
            return False
        return True

    @staticmethod
    def _set_geometry(mesh, verts, polys):
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

    @staticmethod
    def _on_depsgraph_update(scene, depsgraph):
        UpdateMesh._execute()

    @classmethod
    def _execute(cls):
        sb = utils.SharedLib('core')
        geo = sb.create_mesh(
            state.arr_x,
            state.arr_y,
            state.arr_z,
            state.iso,
        )
        mesh = bpy.data.meshes[cls._mesh_name]
        cls._set_geometry(mesh, *geo)

    def execute(self, context):
        try:
            mesh = bpy.data.meshes[self._mesh_name]
        except KeyError:
            mesh = bpy.data.meshes.new(self._mesh_name)

        state.iso = self.iso
        self._execute()

        ob = context.object
        if ob and ob.type == 'MESH':
            ob.data = mesh

        handlers = bpy.app.handlers.depsgraph_update_post
        if self._on_depsgraph_update not in handlers:
            handlers.append(self._on_depsgraph_update)

        return {'FINISHED'}
