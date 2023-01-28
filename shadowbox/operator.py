import bpy
import numpy as np
from itertools import combinations
from .image_handle import ImageHandle
import core


def _gather_images(cls, context):
    return [(i.name_full, i.name, i.filepath) \
        for i in bpy.data.images \
        if i.size[0] > 0 \
            and i.size[1] > 0 \
            and not i.name_full.startswith('.')]


def _debug_log(arr):
    np.set_printoptions(threshold=sys.maxsize)
    string = np.array2string(arr)
    f = open("D://a.txt", "w")
    f.write(string)
    f.close()


def _as_array(img, size):
    tmp = None
    try:
        if img.size[:] != size:
            tmp = bpy.data.images.new(".shadowbox", *img.size)
            pxs = np.empty(len(img.pixels), dtype=np.float32)
            img.pixels.foreach_get(pxs)
            tmp.pixels.foreach_set(pxs)
            tmp.scale(*size)
            img = tmp

        pxs = np.empty(len(img.pixels), dtype=np.float32, order='F')
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(*reversed(size), -1)
        pxs = np.asfortranarray(pxs[:, :, 0])
    finally:
        if tmp:
            bpy.data.images.remove(tmp)
    return pxs


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


class Shadowbox(bpy.types.Operator):
    bl_idname = "object.shadowbox"
    bl_label = "Shadowbox"
    bl_description = (
        "Generate a mesh from 3 images describing its silhouette from "
        "each axis"
    )
    bl_options = {'REGISTER', 'UNDO'}
    menu = bpy.types.VIEW3D_MT_add
    _handle = None
    _runs_modal = False
    _imgs = None, None, None

    xname: bpy.props.EnumProperty(
        name="Image X",
        description="Image describing the silhouette along the x-axis",
        items=_gather_images,
        default=0,
    )
    yname: bpy.props.EnumProperty(
        name="Image Y",
        description="Image describing the silhouette along the y-axis",
        items=_gather_images,
        default=1,
    )
    zname: bpy.props.EnumProperty(
        name="Image Z",
        description="Image describing the silhouette along the z-axis",
        items=_gather_images,
        default=2,
    )
    res: bpy.props.IntVectorProperty(
        name="Resolution",
        description="Loop cut count of the generated mesh on each axis",
        default=(256, 256, 256),
        min=1,
        max=2048,
        soft_max=512,
        subtype='XYZ_LENGTH',
    )
    iso: bpy.props.FloatProperty(
        name="Iso Value",
        description=(
            "Pixels brighter than this value are considered part of "
            "the silhouette, pixels darker part of the background"
        ),
        default=0.5,
        min=0,
        max=1,
        step=2,
    )
    adaptivity: bpy.props.FloatProperty(
        name="Adaptivity",
        description="Higher values merge more vertices close together",
        default=1,
        min=0,
        max=1,
        step=2,
    )
    run_modal: bpy.props.BoolProperty(
        name="Run in background",
        description=(
            "If on, the mesh is regenerated every frame until the "
            "ESCAPE key is pressed. Useful to update the mesh while "
            "drawing into images"
        ),
        default=False,
    )
    new_mesh: bpy.props.BoolProperty(
        name="Assign New Mesh",
        description=(
            "If on, a new mesh is created and assigned to the active "
            "object. If off, the mesh of the last execution is reused "
            "and its data overwritten"
        ),
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' \
            and context.object \
            and context.object.type == 'MESH'

    @classmethod
    def on_unregister(cls):
        cls._dispose_handle()
        func = cls._on_depsgraph_update
        handler = bpy.app.handlers.depsgraph_update_post
        if func in handler:
            handler.remove(func)

    @classmethod
    def _dispose_handle(cls):
        if cls._handle:
            cls._handle.dispose()
            cls._handle = None

    @classmethod
    def _on_depsgraph_update(cls, scene, depsgraph):
        ctx = bpy.context
        if ctx.object and ctx.object.select_get() and ctx.mode == 'OBJECT':
            return
        cls.on_unregister()

    def __del__(self):
        type(self).on_unregister()

    def _init(self, context):
        cls = type(self)
        imgs = bpy.data.images

        try:
            cls._imgs = imgs[self.xname], imgs[self.yname], imgs[self.zname]
        except KeyError:
            return False

        if (self.new_mesh):
            context.object.data = bpy.data.meshes.new("shadowbox")

        func = cls._on_depsgraph_update
        handler = bpy.app.handlers.depsgraph_update_post
        if func not in handler:
            handler.append(func)
        return True

    def execute(self, context):
        if self._init(context):
            self._execute(context)
        return {'FINISHED'}

    def _execute(self, context):
        cls = type(self)
        cls._dispose_handle()
        cls._handle = ImageHandle(*cls._imgs)

        new_sizes = reversed(tuple(combinations(self.res, 2)))
        imgs = map(_as_array, cls._imgs, new_sizes)
        geo = core.create_mesh(*imgs, self.iso, self.adaptivity)
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
                if not self._init(context):
                    return {'FINISHED'}
                cls._runs_modal = True
                context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)
