import bpy
import numpy as np
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
    bl_idname = "shadowbox.choose_images"
    bl_label = "Choose Images"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO'}

    x = None
    y = None
    z = None
    img_handle = None

    img_x: bpy.props.EnumProperty(
        name="Image X",
        items=_gather_images,
        default=2,
    )
    img_y: bpy.props.EnumProperty(
        name="Image Y",
        items=_gather_images,
        default=3,
    )
    img_z: bpy.props.EnumProperty(
        name="Image Z",
        items=_gather_images,
        default=4,
    )

    @classmethod
    def _as_array(cls, img):
        pxs = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(img.size[0], img.size[1], -1)
        return pxs[:, :, 0]

    @staticmethod
    def on_depsgraph_update(scene, depsgraph):
        C = bpy.context
        if not C.object or not C.object.select_get() or C.mode != 'OBJECT':
            ChooseImages.img_handle.clear()
            bpy.app.handlers.depsgraph_update_post.remove(ChooseImages.on_depsgraph_update)

    def execute(self, context):
        ximg = bpy.data.images[self.img_x]
        yimg = bpy.data.images[self.img_y]
        zimg = bpy.data.images[self.img_z]

        x = self._as_array(ximg)
        y = self._as_array(yimg)
        z = self._as_array(zimg)

        if z.shape[0] != y.shape[0] or \
           z.shape[1] != x.shape[0] or \
           y.shape[1] != x.shape[1]:
            self.report({'ERROR'}, "No fitting shape")
            return {'FINISHED'}

        if not self.img_handle:
            ChooseImages.img_handle = image_handle.ImageHandle()

        self.img_handle.set_images(ximg, yimg, zimg)
        ChooseImages.x = x
        ChooseImages.y = y
        ChooseImages.z = z

        bpy.app.handlers.depsgraph_update_post.append(ChooseImages.on_depsgraph_update)
        return {'FINISHED'}
