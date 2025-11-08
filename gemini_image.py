"""
NABA Gemini Image Node (REST version, padded aspect ratio)

- REST-only (stable)
- Optional reference images (up to 2) with adjustable strengths
- Aspect ratio dropdown (pads instead of stretch)
- Default prompt + seed, temperature, top_p, top_k
"""

import os, io, base64, math, requests
import torch, numpy as np
from PIL import Image

def _pad_to_aspect(img, target_w, target_h):
    """Pad (not stretch) to target aspect ratio with gray bars"""
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h
    if abs(src_ratio - tgt_ratio) < 0.01:
        return img.resize((target_w, target_h), Image.LANCZOS)

    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    bg = Image.new("RGB", (target_w, target_h), (32, 32, 32))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    bg.paste(img_resized, (paste_x, paste_y))
    return bg

def _resolve_aspect(ar: str, base: int = 1024):
    """Compute width, height from aspect ratio string"""
    try:
        w_s, h_s = ar.split(":")
        w_r, h_r = int(w_s), int(h_s)
    except Exception:
        return base, base
    area = base * base
    width = int(round(math.sqrt(area * w_r / h_r) / 2) * 2)
    height = int(round(area / width / 2) * 2)
    return max(64, width), max(64, height)

def _image_to_base64(image_tensor):
    """Convert torch image tensor to base64 PNG"""
    arr = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class NABAImageNodeREST:
    """
    Stable Gemini REST image generation node (minimal, proven).
    
    Parameters:
    - prompt: Text description for image generation
    - aspect_ratio: Output canvas ratio (9 choices: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21)
                   Images are padded (not cropped) to fit exact ratio, base area ~1024²
    - seed: Optional seed for deterministic generation (0 = random)
    - image_1, image_2: Optional reference images (inline PNG base64)
    - image_1_strength, image_2_strength: Placeholder (0.0-2.0, not yet applied by API)
    - temperature: Sampling randomness (0.0-1.0, default 0.6)
    - top_p: Nucleus sampling (0.1-1.0, default 0.9)
    - top_k: Token limit (1-256, default 64)
    
    Output: RGB image tensor, aspect-padded with gray bars (#202020)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        aspect_choices = ["1:1","3:2","2:3","4:3","3:4","16:9","9:16","21:9","9:21"]
        temp_choices = ["0.0","0.2","0.4","0.6","0.8","1.0"]
        top_p_choices = ["0.1","0.3","0.5","0.7","0.9","1.0"]
        top_k_choices = ["1","8","32","64","128","256"]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "Gorgeous petite, slender woman in high heel pumps"}),
                "aspect_ratio": ("STRING", {"default": "1:1", "choices": aspect_choices}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "image_1": ("IMAGE", {}),
                "image_1_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_2": ("IMAGE", {}),
                "image_2_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature": ("STRING", {"default": "0.6", "choices": temp_choices}),
                "top_p": ("STRING", {"default": "0.9", "choices": top_p_choices}),
                "top_k": ("STRING", {"default": "64", "choices": top_k_choices}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-evaluation when parameters change
        return float("nan")

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/NABA"

    def generate(self, prompt, aspect_ratio="1:1", seed=0,
                 image_1=None, image_1_strength=0.5,
                 image_2=None, image_2_strength=0.5,
                 temperature="0.6", top_p="0.9", top_k="64"):

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("❌ Missing GEMINI_API_KEY")

        url = os.getenv("GEMINI_URL",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent")

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        # Prompt parts
        parts = [{"text": prompt}]
        for img in [image_1, image_2]:
            if img is not None:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": _image_to_base64(img)
                    }
                })

        # Parse generation parameters
        try:
            _temp = float(temperature)
        except:
            _temp = 0.6
        try:
            _top_p = float(top_p)
        except:
            _top_p = 0.9
        try:
            _top_k = int(top_k)
        except:
            _top_k = 64

        generation_config = {"temperature": _temp, "topP": _top_p, "topK": _top_k}
        if seed and seed > 0:
            generation_config["seed"] = seed

        payload = {"contents": [{"parts": parts}], "generationConfig": generation_config}
        
        # Debug logging
        print(f"[NABAImageNodeREST] Generation config being sent:")
        print(f"  temperature: {generation_config.get('temperature')}")
        print(f"  topP: {generation_config.get('topP')}")
        print(f"  topK: {generation_config.get('topK')}")
        print(f"  seed: {generation_config.get('seed', 'not set')}")

        # Call REST API
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Parse image (handles inlineData, inline_data, blob)
        img = None
        try:
            parts = result["candidates"][0]["content"]["parts"]
            for p in parts:
                if "inlineData" in p:
                    raw = base64.b64decode(p["inlineData"]["data"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
                elif "inline_data" in p:
                    raw = base64.b64decode(p["inline_data"]["data"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
                elif "blob" in p:
                    raw = base64.b64decode(p["blob"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
        except Exception as e:
            print(f"[NABAImageNodeREST] decode fail: {e}")

        # If no image returned
        if img is None:
            print("[NABAImageNodeREST] no image returned; using fallback")
            target_w, target_h = _resolve_aspect(aspect_ratio, 1024)
            img = Image.new("RGB", (target_w, target_h), (64, 64, 64))

        # Pad to aspect
        target_w, target_h = _resolve_aspect(aspect_ratio, 1024)
        img = _pad_to_aspect(img, target_w, target_h)

        # Convert to tensor
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        return (torch.from_numpy(arr).unsqueeze(0).contiguous(),)

class NABAImageNode:
    """
    Enhanced Gemini image generation node with expanded capabilities.
    
    Features:
    - Up to 5 reference images (vs 2 in REST node)
    - 15 aspect ratios including ultra-wide (21:9, 9:21) and portrait/landscape variants
    - 4 size presets: 512², 768², 1024² (default), 1536² for quality control
    - Extended temperature range (0.0-1.2) for more creative control
    - Enhanced top_p and top_k options
    
    Parameters:
    - prompt: Text description for image generation
    - aspect_ratio: Output canvas ratio (padded, not cropped). Choices:
        Standard: 1:1, 3:2, 2:3, 4:3, 3:4
        Widescreen: 16:9, 9:16, 21:9, 9:21
        Extended: 5:4, 4:5, 3:1, 1:3, 2:1, 1:2
    - size_preset: Base resolution (affects detail level):
        Small: 256², 384², 512² (fast)
        Medium: 640², 768², 832², 896² (balanced)
        Large: 1024² (default), 1152², 1280² (high quality)
        XL: 1440², 1536², 1792², 2048² (highest quality, slower)
    - seed: Optional deterministic seed (0 = random)
    - image_1 to image_5: Optional reference images (PNG inline)
    - image_X_strength: Reference influence (0.0-2.0, placeholder - not yet applied by API)
    - temperature: Sampling randomness (0.0=focused, 1.2=very creative)
    - top_p: Nucleus sampling threshold (0.1-1.0)
    - top_k: Token sampling limit (1-256)
    
    Output: Single RGB image tensor, padded to requested aspect ratio
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        aspect_choices = ["1:1","3:2","2:3","4:3","3:4","16:9","9:16","21:9","9:21","5:4","4:5","3:1","1:3","2:1","1:2"]
        size_presets = ["256x256","384x384","512x512","640x640","768x768","832x832","896x896","1024x1024","1152x1152","1280x1280","1440x1440","1536x1536","1792x1792","2048x2048"]
        temp_choices = ["0.0","0.2","0.4","0.6","0.8","1.0","1.2"]
        top_p_choices = ["0.1","0.3","0.5","0.7","0.9","0.95","1.0"]
        top_k_choices = ["1","8","16","32","64","128","256"]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Terminator protecting pretty woman"}),
                "aspect_ratio": ("STRING", {"default": "1:1", "choices": aspect_choices}),
                "size_preset": ("STRING", {"default": "1024x1024", "choices": size_presets}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "image_1": ("IMAGE", {}),
                "image_1_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_2": ("IMAGE", {}),
                "image_2_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_3": ("IMAGE", {}),
                "image_3_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_4": ("IMAGE", {}),
                "image_4_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "image_5": ("IMAGE", {}),
                "image_5_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature": ("STRING", {"default": "0.6", "choices": temp_choices}),
                "top_p": ("STRING", {"default": "0.9", "choices": top_p_choices}),
                "top_k": ("STRING", {"default": "64", "choices": top_k_choices}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-evaluation when parameters change
        return float("nan")

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/NABA"

    def generate(self, prompt, aspect_ratio="1:1", size_preset="1024x1024", seed=0,
                 image_1=None, image_1_strength=0.5, image_2=None, image_2_strength=0.5,
                 image_3=None, image_3_strength=0.5, image_4=None, image_4_strength=0.5,
                 image_5=None, image_5_strength=0.5, temperature="0.6", top_p="0.9", top_k="64"):

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("❌ Missing GEMINI_API_KEY")

        url = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent")
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        parts = [{"text": prompt}]
        for img in [image_1, image_2, image_3, image_4, image_5]:
            if img is not None:
                parts.append({"inline_data": {"mime_type": "image/png", "data": _image_to_base64(img)}})

        try:
            _temp = float(temperature)
        except:
            _temp = 0.6
        try:
            _top_p = float(top_p)
        except:
            _top_p = 0.9
        try:
            _top_k = int(top_k)
        except:
            _top_k = 64

        generation_config = {"temperature": _temp, "topP": _top_p, "topK": _top_k}
        if seed and seed > 0:
            generation_config["seed"] = seed

        payload = {"contents": [{"parts": parts}], "generationConfig": generation_config}
        
        # Debug logging
        print(f"[NABAImageNode] Generation config being sent:")
        print(f"  temperature: {generation_config.get('temperature')}")
        print(f"  topP: {generation_config.get('topP')}")
        print(f"  topK: {generation_config.get('topK')}")
        print(f"  seed: {generation_config.get('seed', 'not set')}")
        print(f"  size_preset: {size_preset} -> base_size will be: {size_preset.split('x')[0]}")

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        img = None
        try:
            candidates_parts = result["candidates"][0]["content"]["parts"]
            for p in candidates_parts:
                if "inlineData" in p:
                    raw = base64.b64decode(p["inlineData"]["data"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
                elif "inline_data" in p:
                    raw = base64.b64decode(p["inline_data"]["data"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
                elif "blob" in p:
                    raw = base64.b64decode(p["blob"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    break
        except Exception as e:
            print(f"[NABAImageNode] decode fail: {e}")

        try:
            base_size = int(size_preset.split("x")[0])
        except:
            base_size = 1024

        if img is None:
            print("[NABAImageNode] no image returned; using fallback")
            target_w, target_h = _resolve_aspect(aspect_ratio, base_size)
            img = Image.new("RGB", (target_w, target_h), (64, 64, 64))
        else:
            target_w, target_h = _resolve_aspect(aspect_ratio, base_size)
            img = _pad_to_aspect(img, target_w, target_h)

        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        return (torch.from_numpy(arr).unsqueeze(0).contiguous(),)


NODE_CLASS_MAPPINGS = {"NABAImageNodeREST": NABAImageNodeREST, "NABAImageNode": NABAImageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NABAImageNodeREST": "NABA Image (Gemini REST)", "NABAImageNode": "NABA Image (Gemini Enhanced)"}
