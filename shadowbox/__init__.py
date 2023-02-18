bl_info = {
    "name": "Shadowbox",
    "description": (
        "Generate a mesh from 3 images describing its silhouette from "
        "each axis"
    ),
    "version": (1, 0, 0),
    "blender": (3, 3, 0),
    "category": "Mesh",
    "doc_url": "https://github.com/D4KU/shadowbox",
}

import os
import sys

mpath = os.path.dirname(os.path.realpath(__file__))
if not any(mpath in v for v in sys.path):
    sys.path.insert(0, mpath)

if 'bpy' in locals():
    import importlib
    importlib.reload(operator)
else:
    import bpy
    from . import operator


def _call_globals(attr_name):
    for m in globals().values():
        if hasattr(m, attr_name):
            getattr(m, attr_name)()


def register():
    _call_globals("register")


def unregister():
    _call_globals("unregister")
