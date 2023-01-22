import bpy
import numpy as np
from .image_handle import ImageHandle
import core


def _gather_images(cls, context):
    return [(i.name_full, i.name, i.filepath) for i in bpy.data.images if not i.name_full.startswith('.')]


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


def _set_geometry(mesh, verts, polys, loop_starts, loop_totals):
    mesh.clear_geometry()
    mesh.vertices.add(int(len(verts) / 3))
    mesh.vertices.foreach_set('co', verts)

    mesh.loops.add(len(polys))
    mesh.loops.foreach_set('vertex_index', polys)

    mesh.polygons.add(len(loop_starts))
    mesh.polygons.foreach_set('loop_start', loop_starts)
    mesh.polygons.foreach_set('loop_total', loop_totals)

    mesh.update()


def _prep_img(name, *size):
    imgs = bpy.data.images
    copy_name = ".shadowbox_" + name

    if copy_name in imgs:
        copy = imgs[copy_name]
    else:
        copy = imgs[name].copy()
        copy.name = copy_name

    if copy.size[:] != size:
        copy.scale(*size)
    return copy


class Shadowbox(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadowbox"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}
    menu = bpy.types.VIEW3D_MT_add
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
    res: bpy.props.IntVectorProperty(
        name="Resolution",
        default=(256, 256, 256),
        min=8,
        max=2048,
        soft_max=1024,
        subtype='XYZ_LENGTH',
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
        cls.dispose_handle()

    @classmethod
    def dispose_handle(cls):
        if cls._handle:
            cls._handle.dispose()
            cls._handle = None
        try:
            bpy.app.handlers.depsgraph_update_post.remove(
                cls._on_depsgraph_update)
        except ValueError:
            pass

    @classmethod
    def _on_depsgraph_update(cls, scene, depsgraph):
        ctx = bpy.context
        if ctx.object and ctx.object.select_get() and ctx.mode == 'OBJECT':
            return
        cls.dispose_handle()

    def _init(self, context):
        cls = type(self)
        (xres, yres, zres) = self.res

        try:
            cls._ximg = _prep_img(self.xname, zres, yres)
            cls._yimg = _prep_img(self.yname, xres, zres)
            cls._zimg = _prep_img(self.zname, xres, yres)
        except KeyError:
            return None

        if (self.new_mesh):
            context.object.data = bpy.data.meshes.new("shadowbox")

        cls.dispose_handle()
        cls._handle = ImageHandle(cls._ximg, cls._yimg, cls._zimg)

        handler = bpy.app.handlers.depsgraph_update_post
        if cls._on_depsgraph_update not in handler:
            handler.append(cls._on_depsgraph_update)
        return True

    def execute(self, context):
        if self._init(context):
            self._execute(context)
        return {'FINISHED'}

    def _execute(self, context):
        geo = core.create_mesh(
            _as_array(self._ximg),
            _as_array(self._yimg),
            _as_array(self._zimg),
            self.iso,
            self.adaptivity,
        )
        _set_geometry(context.object.data, *geo)

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
