import os, traceback
from .version import __version__

DEBUG = os.getenv("GEMINI_NODE_DEBUG")
def _d(msg):
	if DEBUG:
		print(f"[gemini_image_gen][init] {msg}")

try:
	from .version import __version__ as _naba_version  # noqa: E402
	from .gemini_image import NODE_CLASS_MAPPINGS as _NCM, NODE_DISPLAY_NAME_MAPPINGS as _NDM
	NODE_CLASS_MAPPINGS = dict(_NCM)
	# Append version to primary node display name
	NODE_DISPLAY_NAME_MAPPINGS = {"NABAImageNode": f"NABA Image (Gemini API) v{_naba_version}"}
	_d("Imported gemini_image OK: classes=" + ",".join(NODE_CLASS_MAPPINGS.keys()))
except Exception as e:  # noqa: BLE001
	_d(f"Import failed: {e}")
	if DEBUG:
		traceback.print_exc()
	# Fallback stub to show something in UI so user sees failure context
	class NABAImportErrorStub:
		@classmethod
		def INPUT_TYPES(cls):
			return {"required": {"message": ("STRING", {"default": "Gemini node import failed; check terminal."})}}
		RETURN_TYPES = ("IMAGE",)
		FUNCTION = "run"
		CATEGORY = "image/NABA"
		def run(self, message):
			import torch
			import numpy as np
			arr = np.zeros((1,16,16,3), dtype=np.float32)
			return (torch.from_numpy(arr.copy()),)
	NODE_CLASS_MAPPINGS = {"NABAImportErrorStub": NABAImportErrorStub}
	NODE_DISPLAY_NAME_MAPPINGS = {"NABAImportErrorStub": "NABA Import ERROR (see console)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

