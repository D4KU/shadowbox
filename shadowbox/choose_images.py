import bpy
import numpy as np
from . import utils
from . import image_handle


def _gather_images(cls, context):
    a = []
    for item in bpy.data.images:
        a.append((
            item.name_full,
            item.name,
            item.filepath,
        ))
    return a


class ChooseImages(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadowbox"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None
    _mesh_name = "shadowbox"

    xname: bpy.props.EnumProperty(
        name="Image X",
        items=_gather_images,
        default=2,
    )
    yname: bpy.props.EnumProperty(
        name="Image Y",
        items=_gather_images,
        default=3,
    )
    zname: bpy.props.EnumProperty(
        name="Image Z",
        items=_gather_images,
        default=4,
    )
    iso: bpy.props.FloatProperty(
        name="Iso",
        min=0,
        max=1,
        step=2,
        default=0,
    )

    @staticmethod
    def _as_array(img):
        pxs = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(img.size[0], img.size[1], -1)
        return pxs[:, :, 0]

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
        ctx = bpy.context
        if ctx.object and ctx.object.select_get() and ctx.mode == 'OBJECT':
            return

        ChooseImages._handle.clear()
        bpy.app.handlers.depsgraph_update_post.remove(
            ChooseImages._on_depsgraph_update)

    def execute(self, context):
        ximg = bpy.data.images[self.xname]
        yimg = bpy.data.images[self.yname]
        zimg = bpy.data.images[self.zname]

        if zimg.size[0] != yimg.size[0] or \
           zimg.size[1] != ximg.size[0] or \
           yimg.size[1] != ximg.size[1]:
            self.report({'ERROR'}, "No fitting shape")
            return {'FINISHED'}

        if not self._handle:
            ChooseImages._handle = image_handle.ImageHandle()
        self._handle.set_images(ximg, yimg, zimg)

        try:
            mesh = bpy.data.meshes[self._mesh_name]
        except KeyError:
            mesh = bpy.data.meshes.new(self._mesh_name)

        sb = utils.SharedLib('core')
        geo = sb.create_mesh(
            self._as_array(ximg),
            self._as_array(yimg),
            self._as_array(zimg),
            self.iso,
        )
        self._set_geometry(mesh, *geo)

        ob = context.object
        if ob and ob.type == 'MESH':
            ob.data = mesh

        bpy.app.handlers.depsgraph_update_post.append(
            self._on_depsgraph_update)
        return {'FINISHED'}
