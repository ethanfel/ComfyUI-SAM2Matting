"""ComfyUI-SAM2Matting: genuine temporal video matting nodes."""

# ComfyUI imports custom-node directories as packages. Pytest imports the root
# of a hyphenated repository as a standalone ``__init__`` module while
# collecting tests, in which case relative imports (and ComfyUI itself) are not
# available. Keep that collection context inert without hiding real ComfyUI
# import failures.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
