## NABA Image (Gemini REST) Node

Version: v1.2.0 (REST-only minimal)

Simplified Gemini 2.5 Flash Image Preview node for ComfyUI. REST-only for stability, two optional reference images, padded aspect ratio resizing (no stretching), and basic sampling controls. All extra debug layers, SDK path, multi-seed, and legacy compatibility code removed to avoid crashes.

### Differences vs Earlier Advanced Version
| Aspect | v1.2.0 REST | v1.1.x Hybrid |
|--------|-------------|---------------|
| Transport | REST only | SDK + REST fallback |
| Reference Images | Up to 2 | Up to 2 (previously 5 in older drafts) |
| Aspect Handling | Padded (preserve composition) | Resize + stretch fallback |
| Seeds | Single seed (pass-through) | Multi-seed list + retries |
| Safety Toggles | None (model defaults) | block_none flag |
| Fallback | Gray padded image | Black or blank tensor |
| Debug Flags | None special | Many env toggles |
| Code Footprint | ~150 lines | 400+ lines |

### Features
* Text → Image via Gemini REST endpoint
* Two optional reference images with individual strength sliders (0.0–2.0)
* Aspect ratio dropdown (1:1 .. 21:9) with padding (no stretching)
* Sampling controls: temperature, top_p, top_k (string dropdowns)
* Single seed pass-through (future: deterministic strategies)
* Graceful fallback → neutral gray padded image if no candidate returned
* Minimal dependencies: `requests`, `Pillow`, `torch`, `numpy`

### Inputs (v1.2.0)
| Name | Type | Notes |
|------|------|-------|
| prompt | STRING | Default: refined descriptive prompt |
| aspect_ratio | DROPDOWN | Long side ~1024, padded to exact ratio |
| seed | INT | Currently pass-through only |
| image_1 | IMAGE | Optional reference image |
| image_1_strength | FLOAT | Influence 0.0–2.0 (currently passed as metadata placeholder) |
| image_2 | IMAGE | Optional second reference |
| image_2_strength | FLOAT | Influence 0.0–2.0 |
| temperature | DROPDOWN | Creative randomness |
| top_p | DROPDOWN | Nucleus sampling |
| top_k | DROPDOWN | Token cap |
| (output) | IMAGE | Tensor 1xHxWx3 float32 0..1 |

### API Key Setup
```bash
export GEMINI_API_KEY=YOUR_KEY
# or create api_key.txt beside this file
echo YOUR_KEY > custom_nodes/gemini_image_gen/api_key.txt
```

### Usage
Search for: `NABA Image (Gemini API)` under category `image/NABA`.
Provide a prompt and pick an aspect ratio. (Legacy graphs supplying `size_preset` continue to function.) Attach up to 5 reference images for guidance.

### Environment Variables
| Var | Purpose | Default |
|-----|---------|---------|
| `GEMINI_API_KEY` | Auth key (required) | - |
| `GEMINI_URL` | Override REST endpoint | preview URL |

### Design Notes (Learned from Nano Banana Repo)
1. Category clarity: use a vendor/feature prefix to avoid collisions (`image/NABA` vs nested reused Gemini categories).
2. Consistent naming: single exported class; avoid legacy alias to prevent duplicate UI entries.
3. Explicit model constraints: Upstream project documents fixed 1024x1024. Add TODO to interrogate SDK for actual size limits.
4. Structured error handling: Instead of raising mid-graph, returning a blank tensor preserves graph continuity.
5. Multi-ref implemented: builds parts list for SDK, mirrors Nano Banana style for REST.
6. Configuration hygiene: Prefer one key variable name (`GEMINI_API_KEY`) but accept legacy `REPLICATE_API_KEY` for transition in REST path.
7. Resilience: Automatic REST fallback provides alternate path when SDK returns no inline image.
8. Simplicity bias: Core kept in single file for easier auditing.
9. Exposed common text generation controls (temperature/top_p/top_k) so workflows can experiment with stylistic variance.

### Roadmap
* True influence weighting (currently metadata only)
* Deterministic seed perturbation method
* Optional retry toggle
* Safety tuning inputs

### Troubleshooting
| Symptom | Cause | Action |
|---------|-------|--------|
| Node missing | Import error | Check terminal logs for stacktrace |
| Always black image | API key missing OR both SDK+REST failed | Ensure key, force REST to isolate, enable debug |
| Duplicate nodes | Old alias or other Gemini plugins | Remove aliases / rename categories |
| Blank but no error with refs | No inline image part returned | Set `GEMINI_VERBOSE_PARTS=1` to inspect parts |
| Want exception instead of blank | Silent fallback masking error | Set `GEMINI_RAISE=1` |

### Debug
Minimal by design. If issues arise print output appears in terminal. Add `print()`s locally if deeper tracing is needed.


### Minimal Test Snippet
```python
from custom_nodes.gemini_image_gen.gemini_image import NABAImageNodeREST
n = NABAImageNodeREST()
tensor, = n.generate(prompt="a tiny chrome banana on a desk", aspect_ratio="1:1")
print(tensor.shape)
```

### License / Disclaimer
Follows ComfyUI usage context. Not affiliated with Google. API usage may incur cost.

### Versioning (Recent)
* 1.2.0 – REST-only minimal rewrite (this document)
* 1.1.x – Hybrid SDK/REST advanced (retired)
* 0.x – Experimental iterations (see older logs)

