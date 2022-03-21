bl_info = {
    "name": "Shadowbox",
    "blender": (3, 0, 1),
    "category": "Mesh",
}

import os
import sys

mpath = os.path.dirname(os.path.realpath(__file__))
if not any(mpath in v for v in sys.path):
    sys.path.insert(0, mpath)

if 'bpy' in locals():
    import importlib
    importlib.reload(choose_images)
    importlib.reload(update_mesh)
else:
    import bpy
    from . import choose_images
    from . import update_mesh


classes = (
    choose_images.ChooseImages,
    update_mesh.UpdateMesh,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
