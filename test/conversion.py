import bpy
import os

# Manually install + enable the OBJ importer
addon_path = os.path.expanduser("~/blender/4.4/scripts/addons/io_scene_obj/__init__.py")
bpy.ops.wm.addon_install(filepath=addon_path)
bpy.ops.preferences.addon_enable(module="io_scene_obj")

