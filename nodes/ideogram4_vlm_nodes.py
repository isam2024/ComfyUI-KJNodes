"""Ideogram 4 VLM nodes.

Describes an input IMAGE as Ideogram 4's scene-composition JSON using a local
vision-language model (GGUF + mmproj via llama-cpp-python's MTMD handler). The
JSON schema is compiled to a token-level grammar, so the output is structurally
guaranteed: exact keys, uppercase hex palettes, 4-int bboxes, no prose, no fences.

The resulting JSON is a drop-in for the Ideogram 4 Prompt Builder nodes
(import_json input) or can be encoded directly.
"""

import base64
import gc
import io
import json
import os

import numpy as np

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _LLAMA_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - environment dependent
    Llama = None
    MTMDChatHandler = None
    _LLAMA_IMPORT_ERROR = e

try:
    import folder_paths
    _GGUF_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")
    os.makedirs(_GGUF_DIR, exist_ok=True)
    if "llm_gguf" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["llm_gguf"] = ([_GGUF_DIR], {".gguf"})
except Exception:  # pragma: no cover
    folder_paths = None
    _GGUF_DIR = None


# ---------------------------------------------------------------------------
# Model cache: one (model, mmproj) pair resident at a time. llama.cpp VRAM is
# invisible to ComfyUI's memory management and survives its "unload models"
# button, so we aggressively keep at most one instance and offer an explicit
# unload-after-generate option.
# ---------------------------------------------------------------------------
_VLM_CACHE = {}


def _unload_all():
    for llm in _VLM_CACHE.values():
        # MTMDChatHandler registers mtmd_free on an ExitStack but has no close()
        # or __del__, and llm.close() never touches the handler — without this the
        # vision tower leaks GPU memory on every unload/reload cycle.
        handler = getattr(llm, "chat_handler", None)
        try:
            if handler is not None and getattr(handler, "_exit_stack", None) is not None:
                handler._exit_stack.close()
                handler.mtmd_ctx = None
        except Exception:
            pass
        try:
            llm.close()
        except Exception:
            pass
    _VLM_CACHE.clear()
    gc.collect()


def _free_comfy_vram():
    """Ask ComfyUI to release its model VRAM before llama.cpp allocates.

    The VLM (model + KV cache + mmproj vision tower) needs ~7.5 GB that ComfyUI's
    memory management can't see, so loading while diffusion models are resident
    fails inside mtmd with an opaque 'Failed to load mtmd context' error.
    """
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass


def _get_vlm(model_path, mmproj_path, n_ctx, n_threads, n_gpu_layers, vision_on_gpu=True):
    key = (
        os.path.abspath(model_path),
        os.path.abspath(mmproj_path),
        int(n_ctx),
        int(n_threads),
        int(n_gpu_layers),
        bool(vision_on_gpu),
    )
    cached = _VLM_CACHE.get(key)
    if cached is not None:
        return cached
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed (or failed to import). Install it into "
            "ComfyUI's Python: pip install llama-cpp-python\n"
            f"(original import error: {_LLAMA_IMPORT_ERROR})"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    if not os.path.isfile(mmproj_path):
        raise FileNotFoundError(f"mmproj (vision projector) GGUF not found: {mmproj_path}")
    _unload_all()
    _free_comfy_vram()
    chat_handler = MTMDChatHandler(
        clip_model_path=mmproj_path, verbose=True, use_gpu=bool(vision_on_gpu)
    )
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=int(n_ctx),
        n_threads=int(n_threads) if int(n_threads) > 0 else None,
        n_gpu_layers=int(n_gpu_layers),
        verbose=True,  # prints "offloaded X/Y layers to GPU" so you can confirm GPU use
    )
    _VLM_CACHE[key] = llm
    return llm


# ---------------------------------------------------------------------------
# Schema: identical shape to the Ideogram 4 scene-composition spec used by the
# prompt builder nodes. llama.cpp compiles it to an enforcing grammar.
# ---------------------------------------------------------------------------
_HEX = {"type": "string", "pattern": "^#[0-9A-F]{6}$"}

SCENE_COMPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "high_level_description": {"type": "string", "maxLength": 300},
        "style_description": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "aesthetics": {"type": "string", "maxLength": 180},
                "lighting": {"type": "string", "maxLength": 180},
                "photo": {"type": "string", "maxLength": 180},
                "medium": {"type": "string", "maxLength": 50},
                "color_palette": {"type": "array", "items": _HEX, "minItems": 3, "maxItems": 6},
            },
            "required": ["aesthetics", "lighting", "photo", "medium", "color_palette"],
        },
        "compositional_deconstruction": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "background": {"type": "string", "maxLength": 350},
                "elements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["obj"]},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "desc": {"type": "string", "maxLength": 300},
                            "color_palette": {
                                "type": "array",
                                "items": _HEX,
                                "minItems": 2,
                                "maxItems": 5,
                            },
                        },
                        "required": ["type", "bbox", "desc", "color_palette"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ],
}


def _schema_with_bounds(min_elements, max_elements):
    """Clone the schema with the element-count bounds baked in.

    The bounds become part of the compiled grammar, so the model is FORCED to emit
    at least min_elements — the array literally cannot close earlier. This is the
    reliable lever against lazy single-element summaries (e.g. of collages).
    """
    schema = json.loads(json.dumps(SCENE_COMPOSITION_SCHEMA))
    elements = schema["properties"]["compositional_deconstruction"]["properties"]["elements"]
    lo = max(1, min(int(min_elements), 64))
    hi = max(lo, min(int(max_elements), 64))
    elements["minItems"] = lo
    elements["maxItems"] = hi
    return schema


DEFAULT_SYSTEM_PROMPT = """You are a scene deconstruction assistant. You are shown an image. You output a single JSON document that describes that image in a structured, render-ready form. You output JSON only — no prose, no markdown fences, no commentary. Describe only what is actually visible in the image; never invent objects, text, or details that are not there.

# Output format

Your response MUST be a single valid JSON object matching exactly this shape and key set:

```
{
  "high_level_description": "",
  "style_description": {
    "aesthetics": "",
    "lighting": "",
    "photo": "",
    "medium": "",
    "color_palette": []
  },
  "compositional_deconstruction": {
    "background": "",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 0, 0, 0],
        "desc": "",
        "color_palette": []
      }
    ]
  }
}
```

All keys above are required and must appear exactly as named. Do not add, rename, or remove any keys.

# Field rules

## high_level_description

- String, **50-word hard cap**. ONE long sentence preferred, never more than two. Start immediately with the subject — no "this image shows", "depicts", "captures". Identify the main subject(s), medium, and overall composition. General terms (`various`, `multiple`) are fine here; granular detail belongs in element `desc`s and `background`.

## style_description

A flat object describing how the image is rendered, independent of what it depicts.

- `aesthetics` (string): Overall visual style and treatment as observed.
- `lighting` (string): Light source, direction, quality, and color temperature as observed.
- `photo` (string): Camera/lens/photographic specifics when the image is photographic (focal length feel, depth of field, angle). Use an empty string "" if the medium is not photographic.
- `medium` (string): The medium category (e.g. "photography", "oil painting", "3D render", "watercolor", "digital illustration").
- `color_palette` (array of strings): 3–6 dominant colors of the overall image as uppercase hex codes in #RRGGBB form, estimated from the actual pixels.

## compositional_deconstruction.background

- String, **60-word cap**. Describe only the environment behind and around the subjects, plus scene-wide lighting/atmosphere and any shadows. Do NOT describe any element listed in `elements`.

## compositional_deconstruction.elements

Array with at least 1 item, listed roughly background-to-foreground.

- Identify each distinct subject as its OWN element; prefer 3–8 elements for typical scenes. A single element is only correct when the image truly contains exactly one subject on a background.
- If the image is a collage, grid, contact sheet, or multi-panel layout, output one element PER PANEL (each with that panel's own bbox), merging only the least distinct panels if there are more panels than the element budget. NEVER summarize a multi-subject image as one element whose bbox spans the whole canvas.

Each element:

- `type` (string): Always "obj".
- `bbox` (array of 4 integers): [x_min, y_min, x_max, y_max] of the element's location in PIXEL COORDINATES of the image, with origin at the top-left, x increasing rightward, y increasing downward. Make the box tight around the visible element. Report the pixels exactly as you see them — coordinates are rescaled automatically afterward.
- `desc` (string): **30–60 words, 60-word HARD CAP.** Identity FIRST (a standalone catalog entry — open with what the thing is, not "the X"), then major attributes briefly (people: skin tone, hair, each garment + color, expression, pose; objects: shape, material, color, distinctive parts), then one distinguishing detail. **One subject = one element** — anatomical/structural parts go in that element's desc, never as separate elements. Do NOT include: camera/render language (DoF, bokeh, focus, grain, lens flare); shadow language (scene-wide shadows go in `background`); metaphor/impression words (luminous, radiant, vibrant, lush, stunning, breathtaking) — use observable properties instead. Do not restate global background or style information.
- `color_palette` (array of strings): 2–5 dominant colors of THIS element as uppercase hex codes in #RRGGBB form, estimated from the actual pixels.

# Hard constraints

- Output valid JSON and nothing else.
- Use only the keys defined above, exactly as spelled. No extra fields.
- Do not wrap the JSON in code fences or add explanations.
- Describe the image as-is. If text is legible in the image, quote it exactly in the relevant `desc`.

# Instruction

Deconstruct the provided image into the JSON now."""


# Keys whose values are structural noise for the optional flattened-prompt output.
_SKIP_FLATTEN_KEYS = {"bbox", "type", "color_palette"}


def _flatten_to_prompt(data):
    """A comma-joined positive prompt from the descriptive strings (bboxes/hex skipped)."""
    parts = []

    def collect(value, key=None):
        if key in _SKIP_FLATTEN_KEYS:
            return
        if isinstance(value, str):
            v = value.strip()
            if v and not v.startswith("#"):
                parts.append(v)
        elif isinstance(value, list):
            for item in value:
                collect(item, key)
        elif isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)

    collect(data)
    return ", ".join(parts)


def _image_to_data_uri(image, max_side):
    """ComfyUI IMAGE tensor (B,H,W,C float 0..1) -> (PNG data URI, (w, h)) of the first frame."""
    from PIL import Image

    frame = image[0] if image.ndim == 4 else image
    arr = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max_side > 0 and max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        new_size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64, pil.size


def _rescale_bboxes(data, img_w, img_h):
    """Map model bboxes from sent-image pixel coordinates onto the 1000x1000 canvas.

    VLMs ground boxes in pixels (x-first) far more accurately than they do arithmetic,
    so the system prompt asks for [x_min, y_min, x_max, y_max] pixel coordinates and
    the mapping happens here, where the sent dimensions are known exactly. The output
    order is Ideogram 4's canonical [ymin, xmin, ymax, xmax] (matching the Prompt
    Builder's exporter). Degenerate/out-of-range boxes are repaired.
    """
    for el in data.get("compositional_deconstruction", {}).get("elements", []):
        x0, y0, x1, y1 = el["bbox"][:4]
        x0, x1 = sorted((round(x0 * 1000 / img_w), round(x1 * 1000 / img_w)))
        y0, y1 = sorted((round(y0 * 1000 / img_h), round(y1 * 1000 / img_h)))
        x0 = min(max(x0, 0), 999)
        y0 = min(max(y0, 0), 999)
        x1 = min(max(x1, x0 + 1), 1000)
        y1 = min(max(y1, y0 + 1), 1000)
        el["bbox"] = [y0, x0, y1, x1]
    return data


class Ideogram4ImageToJSONKJ:
    """Describe an input image as Ideogram 4 scene-composition JSON with a local VLM."""

    @classmethod
    def INPUT_TYPES(cls):
        if folder_paths is not None:
            try:
                gguf_files = folder_paths.get_filename_list("llm_gguf")
            except Exception:
                gguf_files = []
        else:
            gguf_files = []
        # Split the folder into projector files and model files so the two dropdowns
        # can't default to the same file. Falls back to the full list if the naming
        # convention doesn't match.
        mmproj_files = [f for f in gguf_files if "mmproj" in os.path.basename(f).lower()]
        model_files = [f for f in gguf_files if f not in mmproj_files]
        # Vision-capable models first so the widget default isn't a text-only LLM,
        # which mtmd rejects with an opaque "Failed to load mtmd context" error.
        model_files.sort(key=lambda f: (not any(
            tag in os.path.basename(f).lower() for tag in ("vl", "vision", "llava", "minicpm-v")
        ), os.path.basename(f).lower()))
        placeholder = ["<put a .gguf in models/llm_gguf>"]
        model_choices = model_files or gguf_files or placeholder
        mmproj_choices = mmproj_files or gguf_files or placeholder

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_choices, {"tooltip": "Vision-capable GGUF (e.g. Qwen2.5-VL Instruct)"}),
                "mmproj_name": (mmproj_choices, {"tooltip": "Matching mmproj (vision projector) GGUF for the model"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 16384,
                                       "tooltip": "Budget ~100 tokens per element plus ~300 overhead — raise this "
                                                  "(and n_ctx) when raising max_elements."}),
            },
            "optional": {
                "instructions": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "Optional extra guidance, e.g. 'focus on the two figures, ignore the watermark'"},
                ),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "model_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Absolute path to the model .gguf (overrides model_name)"},
                ),
                "mmproj_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Absolute path to the mmproj .gguf (overrides mmproj_name)"},
                ),
                "n_ctx": ("INT", {"default": 8192, "min": 1024, "max": 65536,
                                  "tooltip": "Image tokens count against context; keep ≥8192"}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 128}),
                "n_gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999,
                     "tooltip": "-1 = offload all layers to GPU (needs CUDA build). 0 = CPU only."},
                ),
                "max_image_side": (
                    "INT",
                    {"default": 768, "min": 0, "max": 4096,
                     "tooltip": "Downscale the image's long side before encoding (0 = no downscale). Vision attention memory grows quadratically with resolution — 1024+ can need a ~5 GB encode buffer and OOM a 16 GB card that also holds the 7B model."},
                ),
                "vision_on_cpu": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Run the vision encoder (mmproj) on CPU. Slower per image, but avoids the large GPU encode buffer when VRAM is tight."},
                ),
                "unload_after_generate": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Free the VLM from VRAM after each generation. llama.cpp memory is NOT visible to ComfyUI's memory management and survives its unload button — enable this when running close to the VRAM limit."},
                ),
                # New widgets are appended at the END only: ComfyUI saves workflow
                # widget values by position, and inserting mid-list shifts every
                # later value into the wrong widget on existing workflows.
                "min_elements": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64,
                     "tooltip": "Grammar-enforced minimum element count — the model cannot emit fewer. "
                                "Raise for collages/grids that the model lazily summarizes as one element."},
                ),
                "max_elements": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64,
                     "tooltip": "Element budget for the whole caption. Ideogram 4 guidance prefers 3-8; beyond that "
                                "expect token bloat and weaker per-region adherence. High counts need max_tokens "
                                "and n_ctx headroom (~100 output tokens per element)."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "prompt")
    FUNCTION = "generate"
    CATEGORY = "KJNodes/ideogram4"
    DESCRIPTION = (
        "Runs the input image through a local vision-language model (GGUF + mmproj, "
        "llama-cpp-python) and emits grammar-enforced Ideogram 4 scene-composition JSON. "
        "Feed the json output into the Ideogram 4 Prompt Builder's import_json input, or "
        "encode it directly."
    )

    def generate(
        self,
        image,
        model_name,
        mmproj_name,
        seed,
        temperature,
        max_tokens,
        min_elements=1,
        max_elements=8,
        instructions="",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_path_override="",
        mmproj_path_override="",
        n_ctx=8192,
        n_threads=0,
        n_gpu_layers=-1,
        max_image_side=768,
        vision_on_cpu=False,
        unload_after_generate=False,
    ):
        def resolve(override, name, what):
            if override.strip():
                return override.strip()
            if folder_paths is not None:
                path = folder_paths.get_full_path("llm_gguf", name)
                if not path:
                    raise FileNotFoundError(
                        f"'{name}' not found in models/llm_gguf. Drop the {what} GGUF there."
                    )
                return path
            return os.path.join(_GGUF_DIR or "", name)

        model_path = resolve(model_path_override, model_name, "model")
        mmproj_path = resolve(mmproj_path_override, mmproj_name, "mmproj")
        if os.path.abspath(model_path) == os.path.abspath(mmproj_path):
            raise ValueError(
                "model_name and mmproj_name point to the same file — select the main "
                "model GGUF and its separate mmproj (vision projector) GGUF."
            )

        data_uri, (img_w, img_h) = _image_to_data_uri(image, int(max_image_side))

        user_content = [{"type": "image_url", "image_url": {"url": data_uri}}]
        extra = instructions.strip()
        user_text = extra if extra else "Deconstruct this image into the JSON."
        if int(min_elements) > 1:
            # The grammar already forces the count; saying it up front lets the model
            # plan N regions instead of being cornered into filler mid-generation.
            user_text += (
                f" You MUST identify at least {int(min_elements)} distinct subjects or "
                "panels, each as its own element with its own tight bbox covering only "
                "that subject — do not reuse the same bbox or describe the image as a whole."
            )
        if int(max_elements) != 8:
            user_text += f" Your element budget is {int(max_elements)} elements."
        user_content.append({"type": "text", "text": user_text})

        # The grammar floor makes output length a function of min_elements, so a
        # too-small max_tokens guarantees a truncated document. Auto-raise it.
        needed_tokens = 400 + 120 * max(1, int(min_elements))
        eff_max_tokens = max(int(max_tokens), needed_tokens)
        if eff_max_tokens > int(max_tokens):
            print(f"[Ideogram4ImageToJSONKJ] max_tokens={int(max_tokens)} cannot fit "
                  f"min_elements={int(min_elements)} (~120 tokens each); using {eff_max_tokens}.")

        # Always evict ComfyUI-managed models first — the vision encode buffer is
        # allocated per image, so even a cached VLM collides with diffusion models
        # loaded by a render since the last call.
        _free_comfy_vram()
        llm = _get_vlm(
            model_path, mmproj_path, n_ctx, n_threads, n_gpu_layers,
            vision_on_gpu=not vision_on_cpu,
        )
        try:
            try:
                result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                # Schema -> token-level grammar: structure, key set, hex pattern and
                # bbox arity are enforced during sampling. Malformed JSON is impossible.
                    response_format={"type": "json_object",
                                     "schema": _schema_with_bounds(min_elements, max_elements)},
                    temperature=float(temperature),
                    max_tokens=eff_max_tokens,
                    seed=int(seed),
                )
            except OSError as e:
                # llama.cpp's mtmd helper doesn't propagate a failed CUDA allocation
                # during image encode and dereferences null instead — the access
                # violation surfaces here as OSError. The instance is unusable.
                _unload_all()
                raise RuntimeError(
                    "The vision encoder crashed, almost certainly after running out of "
                    "VRAM while encoding the image (check the log for 'cudaMalloc "
                    "failed: out of memory' just above). Vision attention memory grows "
                    "quadratically with image size. Fixes, in order: lower "
                    "max_image_side (768 or 640), enable vision_on_cpu, enable "
                    "unload_after_generate, or free VRAM from other apps. "
                    f"Original error: {e}"
                ) from e
            except ValueError as e:
                if "mtmd" not in str(e).lower():
                    raise
                arch = (llm.metadata or {}).get("general.architecture", "unknown")
                name = (llm.metadata or {}).get("general.name", os.path.basename(model_path))
                _unload_all()  # the pair is unusable; don't leave it resident
                raise ValueError(
                    f"The vision projector could not attach to the selected model "
                    f"'{name}' (architecture: {arch}). This usually means model_name is a "
                    f"text-only LLM or doesn't match the mmproj — select the vision model "
                    f"the mmproj belongs to (e.g. Qwen2.5-VL-*-Instruct with its own "
                    f"mmproj file). If the pair is correct, the load may have run out of "
                    f"VRAM. Original error: {e}"
                ) from e
        finally:
            if unload_after_generate:
                _unload_all()

        raw = result["choices"][0]["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # Every string and array in the schema is bounded, so the grammar forces a
            # complete document — failure here means generation hit max_tokens early.
            raise ValueError(
                f"Model output was not valid JSON ({e}). It likely hit max_tokens "
                f"({eff_max_tokens}) before the document closed — raise max_tokens, and "
                f"check n_ctx has room for the prompt plus output (the context window "
                f"caps generation regardless of max_tokens). "
                f"First 200 chars: {raw[:200]!r}"
            )
        data = _rescale_bboxes(data, img_w, img_h)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return (pretty, _flatten_to_prompt(data))
