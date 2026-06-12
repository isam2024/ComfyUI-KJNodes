"""Ideogram 4 caption reroller.

Seed-driven randomizer for an Ideogram 4 scene-composition JSON: rerolls color
palettes (style-level and per-element), style descriptors (aesthetics, lighting,
medium), and injects keywords from an editable pool. Pure Python — no model in
the loop — so rerolls are instant and fully deterministic per seed. Wire the
output into the Prompt Builder's import_json (import_mode "always") or straight
to the encoder, and set the seed widget to "randomize" to reroll every queue.
"""

import colorsys
import json
import random

# ---------------------------------------------------------------------------
# Built-in pools. Deliberately concrete and observable (no impression words),
# matching the caption discipline the rest of the Ideogram 4 stack expects.
# Each can be overridden from the node via the *_pool inputs.
# ---------------------------------------------------------------------------
MEDIUM_POOL = [
    "photography", "oil painting", "watercolor", "gouache", "acrylic painting",
    "digital illustration", "3D render", "ink drawing", "charcoal sketch",
    "pastel drawing", "anime illustration", "comic book art", "pixel art",
    "low poly 3D", "linocut print", "art nouveau poster", "stained glass",
    "paper collage", "embroidery", "claymation still",
]

AESTHETICS_POOL = [
    "minimalist and clean", "maximalist and ornate", "gritty and weathered",
    "soft and dreamlike", "bold graphic shapes with flat color",
    "retro-futurist", "art deco geometry", "cottagecore rustic",
    "cyberpunk neon", "vaporwave pastel", "dark fantasy", "solarpunk lush",
    "brutalist and stark", "baroque drama", "ukiyo-e flatness",
    "mid-century modern", "documentary realism", "surrealist dream logic",
    "storybook whimsy", "industrial decay",
]

LIGHTING_POOL = [
    "golden hour sunlight from the left", "overcast diffuse daylight",
    "hard noon sun with deep shadows", "blue hour twilight",
    "warm candlelight from below", "cool moonlight through a window",
    "neon signage glow in magenta and cyan", "single softbox from the upper right",
    "rim lighting against a dark background", "dappled light through foliage",
    "foggy backlight with visible rays", "firelight flicker, orange and warm",
    "clinical fluorescent overhead light", "stage spotlight from above",
    "bioluminescent ambient glow", "stormy light with breaks in cloud cover",
    "lantern-lit night market glow", "early morning mist with low sun",
]

KEYWORD_POOL = [
    "intricate detail", "weathered texture", "atmospheric haze",
    "high contrast", "muted palette", "film grain", "long shadows",
    "symmetrical composition", "off-center framing", "generous negative space",
    "dense layered composition", "strong silhouettes", "visible brushwork",
    "crisp edges", "soft gradients", "iridescent surfaces", "wet reflections",
    "floating particles", "depth fog", "ornamental borders",
]

COLOR_MODES = ["keep", "shuffle", "hue shift", "jitter", "random harmony"]
COLOR_SCOPES = ["both", "style palette", "element palettes"]
KEYWORD_TARGETS = ["aesthetics", "high_level_description", "output only"]


def _parse_pool(text, fallback):
    """Newline/comma-separated user pool -> list of entries; blank -> built-in."""
    entries = [e.strip() for chunk in (text or "").split("\n") for e in chunk.split(",")]
    entries = [e for e in entries if e]
    return entries if entries else fallback


def _hex_to_hsv(hx):
    r, g, b = (int(hx[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)


def _hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def _valid_hex(hx):
    if not (isinstance(hx, str) and len(hx) == 7 and hx.startswith("#")):
        return False
    try:
        int(hx[1:], 16)
        return True
    except ValueError:
        return False


def _harmony_palette(rng, n):
    """Fresh n-color palette from a random color-theory scheme around a base hue."""
    base = rng.random()
    scheme = rng.choice([
        [0.0, 0.04, -0.04, 0.08, -0.08, 0.12],            # analogous
        [0.0, 0.5, 0.04, 0.54, -0.04, 0.46],              # complementary
        [0.0, 1 / 3, 2 / 3, 0.04, 1 / 3 + 0.04, 2 / 3 + 0.04],  # triadic
        [0.0, 0.42, 0.58, 0.04, 0.46, 0.54],              # split-complementary
    ])
    out = []
    for i in range(n):
        h = base + scheme[i % len(scheme)]
        s = rng.uniform(0.35, 0.95)
        v = rng.uniform(0.35, 0.95)
        out.append(_hsv_to_hex(h, s, v))
    return out


def _reroll_palette(rng, palette, mode, hue_delta):
    """Transform one palette (list of hex strings) per the chosen mode."""
    if mode == "keep" or not isinstance(palette, list) or not palette:
        return palette
    colors = [c for c in palette if _valid_hex(c)]
    if not colors:
        return palette
    if mode == "shuffle":
        rng.shuffle(colors)
        return colors
    if mode == "hue shift":
        # One global rotation (hue_delta) preserves the palette's internal harmony.
        return [_hsv_to_hex(h + hue_delta, s, v)
                for h, s, v in (_hex_to_hsv(c) for c in colors)]
    if mode == "jitter":
        out = []
        for c in colors:
            h, s, v = _hex_to_hsv(c)
            out.append(_hsv_to_hex(h + rng.uniform(-0.09, 0.09),
                                   s + rng.uniform(-0.15, 0.15),
                                   v + rng.uniform(-0.15, 0.15)))
        return out
    if mode == "random harmony":
        return _harmony_palette(rng, len(colors))
    return palette


class Ideogram4CaptionRerollKJ:
    """Seed-driven randomizer for Ideogram 4 caption JSON: colors, styles, keywords."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_in": ("STRING", {"forceInput": True,
                                       "tooltip": "Ideogram 4 caption JSON (e.g. from the VLM node or a builder)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Drives every random choice. Set the widget to 'randomize' to reroll each queue."}),
                "color_mode": (COLOR_MODES, {"tooltip": "shuffle = permute existing; hue shift = rotate all hues together; "
                                                        "jitter = perturb each color; random harmony = fresh palette from color theory."}),
                "color_scope": (COLOR_SCOPES,),
                "reroll_aesthetics": ("BOOLEAN", {"default": False}),
                "reroll_lighting": ("BOOLEAN", {"default": False}),
                "reroll_medium": ("BOOLEAN", {"default": False}),
                "keyword_count": ("INT", {"default": 0, "min": 0, "max": 10,
                                          "tooltip": "How many keywords to draw from the pool (0 = none)."}),
                "keyword_target": (KEYWORD_TARGETS, {"tooltip": "Where drawn keywords go: appended to style aesthetics, "
                                                                "appended to high_level_description, or only to the keywords output."}),
            },
            "optional": {
                "keyword_pool": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Comma/newline-separated keywords. Blank = built-in pool."}),
                "aesthetics_pool": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Comma/newline-separated aesthetics. Blank = built-in pool."}),
                "lighting_pool": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Comma/newline-separated lighting setups. Blank = built-in pool."}),
                "medium_pool": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Comma/newline-separated mediums. Blank = built-in pool."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "keywords")
    FUNCTION = "reroll"
    CATEGORY = "KJNodes/ideogram4"
    DESCRIPTION = (
        "Deterministically rerolls parts of an Ideogram 4 caption JSON from a seed: "
        "color palettes (style and per-element), style descriptors (aesthetics, "
        "lighting, medium from editable pools), and keywords drawn from an editable "
        "pool. Wire json into the Prompt Builder's import_json (import_mode 'always') "
        "or directly to the encoder."
    )

    def reroll(
        self,
        json_in,
        seed,
        color_mode,
        color_scope,
        reroll_aesthetics,
        reroll_lighting,
        reroll_medium,
        keyword_count,
        keyword_target,
        keyword_pool="",
        aesthetics_pool="",
        lighting_pool="",
        medium_pool="",
    ):
        try:
            data = json.loads(json_in)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"json_in is not valid JSON ({e}). Wire the caption JSON output here "
                "(not the flattened prompt string)."
            )
        if not isinstance(data, dict):
            raise ValueError("json_in must be a JSON object (an Ideogram 4 caption).")

        rng = random.Random(int(seed))
        # Drawn once so "hue shift" rotates every palette in the caption together,
        # keeping global color coherence between elements and the style palette.
        hue_delta = rng.uniform(0.08, 0.92)

        # ── colors ──
        sd = data.get("style_description")
        if color_scope in ("both", "style palette") and isinstance(sd, dict):
            pal = _reroll_palette(rng, sd.get("color_palette"), color_mode, hue_delta)
            if pal:
                sd["color_palette"] = pal
        if color_scope in ("both", "element palettes"):
            elements = (data.get("compositional_deconstruction") or {}).get("elements") or []
            for el in elements:
                if isinstance(el, dict):
                    pal = _reroll_palette(rng, el.get("color_palette"), color_mode, hue_delta)
                    if pal:
                        el["color_palette"] = pal

        # ── styles ──
        if reroll_aesthetics or reroll_lighting or reroll_medium:
            if not isinstance(sd, dict):
                # No style block yet: create a minimal photo-agnostic one. The builder
                # treats a present-but-empty "photo" key as the photo style kind.
                sd = {"aesthetics": "", "lighting": "", "photo": "", "medium": ""}
                data["style_description"] = sd
            if reroll_aesthetics:
                sd["aesthetics"] = rng.choice(_parse_pool(aesthetics_pool, AESTHETICS_POOL))
            if reroll_lighting:
                sd["lighting"] = rng.choice(_parse_pool(lighting_pool, LIGHTING_POOL))
            if reroll_medium:
                sd["medium"] = rng.choice(_parse_pool(medium_pool, MEDIUM_POOL))

        # ── keywords ──
        keywords = []
        if int(keyword_count) > 0:
            pool = _parse_pool(keyword_pool, KEYWORD_POOL)
            keywords = rng.sample(pool, min(int(keyword_count), len(pool)))
            joined = ", ".join(keywords)
            if keyword_target == "aesthetics":
                if not isinstance(sd, dict):
                    sd = {"aesthetics": "", "lighting": "", "photo": "", "medium": ""}
                    data["style_description"] = sd
                base = (sd.get("aesthetics") or "").strip()
                sd["aesthetics"] = f"{base}, {joined}" if base else joined
            elif keyword_target == "high_level_description":
                base = (data.get("high_level_description") or "").strip()
                base = base.rstrip(".")
                data["high_level_description"] = f"{base}, {joined}" if base else joined

        return (json.dumps(data, indent=2, ensure_ascii=False), ", ".join(keywords))
