import bpy
import numpy as np
from . import utils


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
    grid = None
    gridres = None

    img_x: bpy.props.EnumProperty(
        name="Image X",
        items=_gather_images,
        default=1,
    )
    img_y: bpy.props.EnumProperty(
        name="Image Y",
        items=_gather_images,
        default=2,
    )
    img_z: bpy.props.EnumProperty(
        name="Image Z",
        items=_gather_images,
        default=3,
    )

    @classmethod
    def _as_array(cls, name):
        img = bpy.data.images[name]
        pxs = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(img.size[0], img.size[1], -1)
        return pxs[:, :, 0]

    def execute(self, context):
        x = self._as_array(self.img_x)
        y = self._as_array(self.img_y)
        z = self._as_array(self.img_z)

        if z.shape[0] != y.shape[0] or \
           z.shape[1] != x.shape[0] or \
           y.shape[1] != x.shape[1]:
            return {'CANCELLED'}

        sb = utils.SharedLib('core')
        (ChooseImages.grid, ChooseImages.gridres) = sb.choose_images(x, y, z)
        ChooseImages.x = x
        ChooseImages.y = y
        ChooseImages.z = z
        return {'FINISHED'}
