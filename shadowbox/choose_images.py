import bpy
import numpy as np
from . import utils
from . import image_handle
import core
# sb = utils.SharedLib('core')


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
    _MESH_NAME = "shadowbox"
    _handle = None
    _runs_modal = False
    _mesh = None
    _ximg = None
    _yimg = None
    _zimg = None

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
        default=0.5,
    )
    run_modal: bpy.props.BoolProperty(
        name="Modal",
        default=False,
    )

    @staticmethod
    def debug_log(arr):
        np.set_printoptions(threshold=sys.maxsize)
        string = np.array2string(arr)
        f = open("D://a.txt", "w")
        f.write(string)
        f.close()

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

    def _init(self, context):
        ximg = bpy.data.images[self.xname]
        yimg = bpy.data.images[self.yname]
        zimg = bpy.data.images[self.zname]

        if zimg.size[0] != yimg.size[0] or \
           zimg.size[1] != ximg.size[0] or \
           yimg.size[1] != ximg.size[1]:
            self.report({'ERROR'}, "No fitting shape")
            return False

        ChooseImages._ximg = ximg
        ChooseImages._yimg = yimg
        ChooseImages._zimg = zimg

        if not self._handle:
            ChooseImages._handle = image_handle.ImageHandle()
        self._handle.set_images(ximg, yimg, zimg)

        try:
            ChooseImages._mesh = bpy.data.meshes[self._MESH_NAME]
        except KeyError:
            ChooseImages._mesh = bpy.data.meshes.new(self._MESH_NAME)

        ob = context.object
        if ob and ob.type == 'MESH':
            ob.data = self._mesh

        handler = bpy.app.handlers.depsgraph_update_post
        if self._on_depsgraph_update not in handler:
            handler.append(self._on_depsgraph_update)

        return True

    def execute(self, context):
        if self._init(context):
            self._execute(context)
        return {'FINISHED'}

    def _execute(self, context):
        geo = core.create_mesh(
            self._as_array(self._ximg),
            self._as_array(self._yimg),
            self._as_array(self._zimg),
            self.iso,
        )
        self._set_geometry(self._mesh, *geo)

    def modal(self, context, event):
        if event.type == 'ESC':
            ChooseImages._runs_modal = False
            return {'FINISHED'}
        self._execute(context)
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        print("invoke")
        if self.run_modal:
            if not self._runs_modal:
                if not self._init(context):
                    return {'FINISHED'}
                ChooseImages._runs_modal = True
                context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)
