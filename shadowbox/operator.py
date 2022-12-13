import time
import bpy
import numpy as np
from .image_handle import ImageHandle
import core


def _gather_images(cls, context):
    return [(i.name_full, i.name, i.filepath) for i in bpy.data.images]


def _debug_log(arr):
    np.set_printoptions(threshold=sys.maxsize)
    string = np.array2string(arr)
    f = open("D://a.txt", "w")
    f.write(string)
    f.close()


def _as_array(img):
    pxs = np.empty(len(img.pixels), dtype=np.float32, order='F')
    img.pixels.foreach_get(pxs)
    pxs = pxs.reshape(img.size[0], img.size[1], -1)
    return np.asfortranarray(pxs[:, :, 0])


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


class Shadowbox(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadowbox"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}
    menu = bpy.types.VIEW3D_MT_add
    _MESH_NAME = "shadowbox"
    _handle = None
    _runs_modal = False
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
        name="Iso Value",
        min=0,
        max=1,
        step=2,
        default=0.5,
    )
    adaptivity: bpy.props.FloatProperty(
        name="Adaptivity",
        min=0,
        max=1,
        step=2,
        default=1,
    )
    run_modal: bpy.props.BoolProperty(
        name="Modal",
        default=False,
    )
    new_mesh: bpy.props.BoolProperty(
        name="Assign New Mesh",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' \
            and context.object \
            and context.object.type == 'MESH'

    @classmethod
    def on_unregister(cls):
        cls.dispose()

    @classmethod
    def dispose(cls):
        if cls._handle:
            cls._handle.dispose()
            cls._handle = None
        bpy.app.handlers.depsgraph_update_post.remove(
            cls._on_depsgraph_update)

    @classmethod
    def _on_depsgraph_update(cls, scene, depsgraph):
        ctx = bpy.context
        if ctx.object and ctx.object.select_get() and ctx.mode == 'OBJECT':
            return
        cls.dispose()

    def _init(self, context):
        cls = type(self)
        try:
            ximg = bpy.data.images[self.xname]
            yimg = bpy.data.images[self.yname]
            zimg = bpy.data.images[self.zname]
        except KeyError:
            return False

        if zimg.size[0] != yimg.size[0] or \
           zimg.size[1] != ximg.size[0] or \
           yimg.size[1] != ximg.size[1]:
            self.report({'ERROR'}, "No fitting shape")
            return False

        if (self.new_mesh):
            context.object.data = bpy.data.meshes.new(self._MESH_NAME)

        if not cls._handle:
            cls._handle = ImageHandle(ximg, yimg, zimg)

        handler = bpy.app.handlers.depsgraph_update_post
        if cls._on_depsgraph_update not in handler:
            handler.append(cls._on_depsgraph_update)

        cls._ximg = ximg
        cls._yimg = yimg
        cls._zimg = zimg
        return True

    def execute(self, context):
        if self._init(context):
            self._execute(context)
        return {'FINISHED'}

    def _execute(self, context):
        t1 = time.time()
        geo = core.create_mesh(
            _as_array(self._ximg),
            _as_array(self._yimg),
            _as_array(self._zimg),
            self.iso,
            self.adaptivity,
        )
        t2 = time.time()
        print(t2-t1)

        t3 = time.time()
        _set_geometry(context.object.data, *geo)
        t4 = time.time()
        print(t4-t3)

    def modal(self, context, event):
        if event.type == 'ESC':
            type(self)._runs_modal = False
            return {'FINISHED'}
        self._execute(context)
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        cls = type(self)
        if self.run_modal:
            if not cls._runs_modal:
                if not cls._init(context):
                    return {'FINISHED'}
                cls._runs_modal = True
                context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)
