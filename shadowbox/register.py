import bpy
import sys


def register(*menus):
    """
    Class decorator which adds register and unregister functions to
    the module in which the decorated class is defined and adds the
    class to each given menu.
    """
    def inner(cls):
        def draw_menu(self, context):
            self.layout.operator(cls.bl_idname)

        def register():
            bpy.utils.register_class(cls)
            for m in menus:
                m.append(draw_menu)

        def unregister():
            bpy.utils.unregister_class(cls)
            for m in menus:
                m.remove(draw_menu)

        module = sys.modules[cls.__module__]
        setattr(module, 'register', register)
        setattr(module, 'unregister', unregister)
        return cls
    return inner
