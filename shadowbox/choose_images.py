import bpy
import numpy as np
from . import image_handle
from . import state


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
    bl_idname = "shadowbox.choose_images"
    bl_label = "Choose Images"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

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

    @staticmethod
    def _as_array(img):
        pxs = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(img.size[0], img.size[1], -1)
        return pxs[:, :, 0]

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

        xarr = self._as_array(ximg)
        yarr = self._as_array(yimg)
        zarr = self._as_array(zimg)

        if not self._handle:
            ChooseImages._handle = image_handle.ImageHandle()

        self._handle.set_images(ximg, yimg, zimg)
        state.img_x = ximg
        state.img_y = yimg
        state.img_z = zimg
        state.arr_x = xarr
        state.arr_y = yarr
        state.arr_z = zarr

        bpy.app.handlers.depsgraph_update_post.append(
            self._on_depsgraph_update)
        return {'FINISHED'}
