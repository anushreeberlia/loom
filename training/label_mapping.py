"""
iMaterialist 228 labels → Loom Style schema mapping.

iMaterialist gives us: category (105 types), color (19), material (30), pattern, style, neckline, sleeve
We need:           category (6), primary_color (16), material, style_tags, occasion_tags
We skip for now:   fit, season_tags (not in iMaterialist)

Strategy:
- category: map 105 granular types → our 6 buckets
- color: map 19 colors → our 16 values
- material: use as-is (their labels match ours)
- style_tags: DERIVE from their granular category + style labels
- occasion_tags: DERIVE from their granular category names
"""

# ── iMaterialist category (105 labels) → our 6 categories ──────────────────

CATEGORY_MAP = {
    # top
    "Athletic Shirts": "top",
    "Batwing Tops": "top",
    "Blouses": "top",
    "Crop Tops": "top",
    "Dress Shirts": "top",
    "Halter Tops": "top",
    "Casual Shirts": "top",
    "Hoodies & Sweatshirts": "top",
    "Jerseys": "top",
    "Polos": "top",
    "Pullover Sweaters": "top",
    "Sheer Tops": "top",
    "T-Shirts": "top",
    "Tank Tops": "top",
    "Tube Tops": "top",
    "Undershirts": "top",
    "Vests": "top",
    "Corsets": "top",
    "Bodysuits": "top",
    "Sports Bras": "top",

    # bottom
    "Athletic Pants": "bottom",
    "Athletic Shorts": "bottom",
    "Baggy Jeans": "bottom",
    "Capri Pants": "bottom",
    "Cargo Pants": "bottom",
    "Cargo Shorts": "bottom",
    "Casual Pants": "bottom",
    "Casual Shorts": "bottom",
    "Drawstring Pants": "bottom",
    "Harem Pants": "bottom",
    "Jeans": "bottom",
    "Leggings": "bottom",
    "Pencil Skirts": "bottom",
    "Shorts": "bottom",
    "Skinny Jeans": "bottom",
    "Skirts": "bottom",
    "Sweatpants": "bottom",
    "Trousers": "bottom",
    "Yoga Pants": "bottom",

    # dress
    "Casual Dresses": "dress",
    "Clubbing Dresses": "dress",
    "Cocktail Dresses": "dress",
    "Dresses": "dress",
    "Formal Dresses": "dress",
    "Jumpsuits Overalls & Rompers": "dress",
    "Nightgowns": "dress",
    "Party Dresses": "dress",
    "Prom Dresses": "dress",
    "Wedding Dresses": "dress",
    "Kimonos": "dress",
    "Tutus": "dress",

    # layer
    "Bubble Coats": "layer",
    "Capes & Capelets": "layer",
    "Cardigans": "layer",
    "Jackets": "layer",
    "Peacoats": "layer",
    "Raincoats": "layer",
    "Suits & Blazers": "layer",
    "Trench Coats": "layer",
    "Three Piece Suits": "layer",

    # shoes
    "Boots": "shoes",
    "Business Shoes": "shoes",
    "Casual Shoes": "shoes",
    "Cleats": "shoes",
    "Flats": "shoes",
    "Heels": "shoes",
    "Hiking Boots": "shoes",
    "Loafers & Slip-on Shoes": "shoes",
    "Rain Boots": "shoes",
    "Running Shoes": "shoes",
    "Sandals": "shoes",
    "Slippers": "shoes",
    "Sneakers": "shoes",
    "Stilettos": "shoes",
    "Wedges & Platforms": "shoes",
    "Winter Boots": "shoes",

    # accessory
    "Bra Straps": "accessory",
    "Shoe Accessories": "accessory",
    "Shoe Inserts": "accessory",
    "Shoelaces": "accessory",

    # skip — don't map these (swimwear, underwear, costumes, etc.)
    "Athletic Sets": None,
    "Beach & Swim Wear": None,
    "Bikinis": None,
    "Binders": None,
    "Costumes & Cosplay": None,
    "Custom Made Clothing": None,
    "Dance Wear": None,
    "Fashion Sets": None,
    "Hosiery, Stockings, Tights": None,
    "Jilbaab": None,
    "Lingerie Sleepwear & Underwear": None,
    "Maternity": None,
    "Padded Bras": None,
    "Pajamas": None,
    "Pasties": None,
    "Petticoats": None,
    "Robes": None,
    "Swim Trunks": None,
    "Swimsuit Cover-ups": None,
    "Swimsuits": None,
    "Thermal Underwear": None,
    "Thigh Highs": None,
    "Thongs": None,
    "Underwear": None,
    "Uniforms": None,
}

# ── iMaterialist color → our primary_color ──────────────────────────────────

COLOR_MAP = {
    "Black": "black",
    "White": "white",
    "Gray": "gray",
    "Beige": "beige",
    "Tan": "beige",
    "Brown": "brown",
    "Blue": "blue",
    "Teal": "blue",
    "Green": "green",
    "Yellow": "yellow",
    "Orange": "orange",
    "Red": "red",
    "Maroon": "red",
    "Pink": "pink",
    "Peach": "pink",
    "Purple": "purple",
    "Gold": "metallic",
    "Silver": "metallic",
    "Bronze": "metallic",
    "Multi Color": "multi",
    "Clear": "unknown",
}

# ── iMaterialist material → our material (mostly 1:1) ───────────────────────

MATERIAL_MAP = {
    "Canvas": "canvas",
    "Cashmere": "cashmere",
    "Chambray": "chambray",
    "Chiffon": "chiffon",
    "Corduroy": "corduroy",
    "Cotton": "cotton",
    "Denim": "denim",
    "Faux Fur": "faux fur",
    "Flannel": "flannel",
    "Fleece": "fleece",
    "Gingham": "gingham",
    "Knit": "knit",
    "Lace": "lace",
    "Leather": "leather",
    "Linen": "linen",
    "Neoprene": "neoprene",
    "Nylon": "nylon",
    "Organza": "organza",
    "Patent": "patent leather",
    "Plush": "plush",
    "Polyester": "polyester",
    "Rayon": "rayon",
    "Satin": "satin",
    "Silk": "silk",
    "Spandex": "spandex",
    "Suede": "suede",
    "Taffeta": "taffeta",
    "Tulle": "tulle",
    "Tweed": "tweed",
    "Twill": "twill",
    "Velour": "velour",
    "Velvet": "velvet",
    "Vinyl": "vinyl",
    "Wool": "wool",
}

# ── Derive style_tags from iMaterialist category + style labels ─────────────
# Key insight: their granular categories carry style/occasion signal

CATEGORY_TO_STYLE = {
    # sporty / athletic
    "Athletic Shirts": ["sporty", "athletic"],
    "Athletic Pants": ["sporty", "athletic"],
    "Athletic Shorts": ["sporty", "athletic"],
    "Athletic Sets": ["sporty", "athletic"],
    "Sports Bras": ["sporty", "activewear"],
    "Yoga Pants": ["sporty", "activewear"],
    "Running Shoes": ["sporty", "athletic"],
    "Cleats": ["sporty", "athletic"],
    "Hiking Boots": ["sporty"],
    "Sneakers": ["sporty", "casual"],
    "Jerseys": ["sporty", "casual"],
    "Sweatpants": ["sporty", "casual"],
    "Hoodies & Sweatshirts": ["casual", "streetwear"],

    # casual
    "Casual Dresses": ["casual"],
    "Casual Shirts": ["casual"],
    "Casual Pants": ["casual"],
    "Casual Shorts": ["casual"],
    "Casual Shoes": ["casual"],
    "T-Shirts": ["casual"],
    "Tank Tops": ["casual"],
    "Jeans": ["casual"],
    "Baggy Jeans": ["casual", "streetwear"],
    "Cargo Pants": ["casual", "streetwear"],
    "Cargo Shorts": ["casual"],
    "Shorts": ["casual"],
    "Leggings": ["casual"],
    "Flats": ["casual"],
    "Sandals": ["casual"],
    "Slippers": ["casual"],
    "Loafers & Slip-on Shoes": ["casual", "classic"],

    # elegant / formal
    "Formal Dresses": ["elegant", "glamorous"],
    "Cocktail Dresses": ["elegant", "chic"],
    "Prom Dresses": ["glamorous", "statement"],
    "Wedding Dresses": ["glamorous", "elegant"],
    "Stilettos": ["elegant", "sexy"],
    "Heels": ["elegant", "chic"],
    "Wedges & Platforms": ["chic"],
    "Business Shoes": ["classic", "workwear"],

    # going-out / sexy
    "Clubbing Dresses": ["sexy", "glamorous"],
    "Party Dresses": ["glamorous", "chic"],
    "Bodysuits": ["sexy", "edgy"],
    "Corsets": ["sexy", "edgy"],
    "Tube Tops": ["sexy", "trendy"],
    "Halter Tops": ["sexy", "chic"],
    "Sheer Tops": ["sexy", "edgy"],
    "Crop Tops": ["trendy", "casual"],

    # classic / work
    "Dress Shirts": ["classic", "workwear"],
    "Suits & Blazers": ["classic", "workwear"],
    "Three Piece Suits": ["classic", "workwear"],
    "Trousers": ["classic"],
    "Pencil Skirts": ["classic", "workwear"],
    "Blouses": ["classic", "elegant"],
    "Polos": ["classic", "preppy"],

    # outerwear
    "Jackets": ["casual"],
    "Trench Coats": ["classic", "elegant"],
    "Peacoats": ["classic"],
    "Bubble Coats": ["casual"],
    "Raincoats": ["casual"],
    "Cardigans": ["casual"],
    "Capes & Capelets": ["elegant", "statement"],
    "Vests": ["casual"],

    # other
    "Pullover Sweaters": ["casual"],
    "Skinny Jeans": ["casual", "trendy"],
    "Skirts": ["casual"],
    "Boots": ["edgy"],
    "Rain Boots": ["casual"],
    "Winter Boots": ["casual"],
    "Kimonos": ["bohemian"],
    "Jumpsuits Overalls & Rompers": ["casual", "trendy"],
    "Harem Pants": ["bohemian"],
    "Capri Pants": ["casual"],
    "Drawstring Pants": ["casual"],
    "Undershirts": ["casual"],
    "Batwing Tops": ["casual", "trendy"],
    "Dresses": ["casual"],
    "Nightgowns": ["casual"],
    "Tutus": ["statement"],
}

# iMaterialist "style" group labels → our style_tags
IMAT_STYLE_TO_OURS = {
    "Bodycon": ["sexy", "glamorous"],
    "Vintage Retro": ["vintage"],
    "Summer": ["casual"],
    "Wrap": ["elegant"],
    "Pleated": ["elegant", "classic"],
    "Peplum": ["chic", "elegant"],
    "Tunic": ["casual", "bohemian"],
    "Embroidered": ["bohemian", "statement"],
    "Beaded": ["glamorous", "statement"],
    "Rhinestone Studded": ["glamorous", "statement"],
    "Printed": ["trendy"],
    "Two-Tone": ["chic"],
    "Hi-Lo": ["trendy"],
    "Spaghetti Straps": ["sexy"],
    "Bandage": ["sexy", "glamorous"],
    "Bandeaus": ["sexy"],
    "Criss Cross": ["edgy"],
    "Hollow-Out": ["sexy", "edgy"],
    "Furry": ["statement"],
    "Reversible": [],
}

# ── Derive occasion_tags from iMaterialist category ─────────────────────────

CATEGORY_TO_OCCASION = {
    # gym / workout
    "Athletic Shirts": ["gym", "workout"],
    "Athletic Pants": ["gym", "workout"],
    "Athletic Shorts": ["gym", "workout"],
    "Athletic Sets": ["gym", "workout"],
    "Sports Bras": ["gym", "workout"],
    "Yoga Pants": ["gym", "workout"],
    "Running Shoes": ["gym", "workout"],
    "Cleats": ["gym", "workout"],

    # everyday / casual
    "Casual Dresses": ["everyday", "casual"],
    "Casual Shirts": ["everyday", "casual"],
    "Casual Pants": ["everyday", "casual"],
    "Casual Shorts": ["everyday", "casual"],
    "Casual Shoes": ["everyday", "casual"],
    "T-Shirts": ["everyday", "casual"],
    "Tank Tops": ["everyday", "casual"],
    "Jeans": ["everyday", "casual"],
    "Shorts": ["everyday", "casual"],
    "Sneakers": ["everyday", "casual"],
    "Flats": ["everyday", "casual"],
    "Sandals": ["everyday", "casual", "vacation"],
    "Slippers": ["lounge"],
    "Hoodies & Sweatshirts": ["everyday", "casual"],
    "Sweatpants": ["lounge", "casual"],
    "Leggings": ["everyday", "casual"],

    # party / going-out
    "Clubbing Dresses": ["clubbing", "going-out", "night-out"],
    "Party Dresses": ["party", "going-out", "night-out"],
    "Cocktail Dresses": ["party", "dinner", "date"],
    "Stilettos": ["party", "going-out"],
    "Heels": ["party", "dinner", "going-out"],

    # formal
    "Formal Dresses": ["formal"],
    "Prom Dresses": ["formal", "party"],
    "Wedding Dresses": ["formal"],
    "Three Piece Suits": ["formal", "work"],

    # work
    "Dress Shirts": ["work"],
    "Suits & Blazers": ["work", "formal"],
    "Business Shoes": ["work"],
    "Pencil Skirts": ["work"],
    "Blouses": ["work", "everyday"],
    "Trousers": ["work", "everyday"],
    "Loafers & Slip-on Shoes": ["work", "everyday"],
}

# ── Label ID → label name lookup (from iMat_fashion_2018_label_map_228.csv) ─

LABEL_ID_TO_NAME = {
    1: ("Argyle", "pattern"), 2: ("Asymmetric", "neckline"), 3: ("Athletic Pants", "category"),
    4: ("Athletic Sets", "category"), 5: ("Athletic Shirts", "category"), 6: ("Athletic Shorts", "category"),
    7: ("Backless Dresses", "neckline"), 8: ("Baggy Jeans", "category"), 9: ("Bandage", "style"),
    10: ("Bandeaus", "style"), 11: ("Batwing Tops", "category"), 12: ("Beach & Swim Wear", "category"),
    13: ("Beaded", "style"), 14: ("Beige", "color"), 15: ("Bikinis", "category"),
    16: ("Binders", "category"), 17: ("Black", "color"), 18: ("Blouses", "category"),
    19: ("Blue", "color"), 20: ("Bodycon", "style"), 21: ("Bodysuits", "category"),
    22: ("Boots", "category"), 23: ("Bra Straps", "category"), 24: ("Bronze", "color"),
    25: ("Brown", "color"), 26: ("Bubble Coats", "category"), 27: ("Business Shoes", "category"),
    28: ("Camouflage", "pattern"), 29: ("Canvas", "material"), 30: ("Capes & Capelets", "category"),
    31: ("Capri Pants", "category"), 32: ("Cardigans", "category"), 33: ("Cargo Pants", "category"),
    34: ("Cargo Shorts", "category"), 35: ("Cashmere", "material"), 36: ("Casual Dresses", "category"),
    37: ("Casual Pants", "category"), 38: ("Casual Shirts", "category"), 39: ("Casual Shoes", "category"),
    40: ("Casual Shorts", "category"), 41: ("Chambray", "material"), 42: ("Checkered", "pattern"),
    43: ("Chevron", "pattern"), 44: ("Chiffon", "material"), 45: ("Clear", "color"),
    46: ("Cleats", "category"), 47: ("Clubbing Dresses", "category"), 48: ("Cocktail Dresses", "category"),
    49: ("Collared", "neckline"), 50: ("Corduroy", "material"), 51: ("Corsets", "category"),
    52: ("Costumes & Cosplay", "category"), 53: ("Cotton", "material"), 54: ("Criss Cross", "style"),
    55: ("Crochet", "pattern"), 56: ("Crop Tops", "category"), 57: ("Custom Made Clothing", "category"),
    58: ("Dance Wear", "category"), 59: ("Denim", "material"), 60: ("Drawstring Pants", "category"),
    61: ("Dress Shirts", "category"), 62: ("Dresses", "category"), 63: ("Embroidered", "style"),
    64: ("Fashion Sets", "category"), 65: ("Faux Fur", "material"), 66: ("Female", "gender"),
    67: ("Flannel", "material"), 68: ("Flats", "category"), 69: ("Fleece", "material"),
    70: ("Floral", "pattern"), 71: ("Formal Dresses", "category"), 72: ("Fringe", "pattern"),
    73: ("Furry", "style"), 74: ("Galaxy", "pattern"), 75: ("Geometric", "pattern"),
    76: ("Gingham", "material"), 77: ("Gold", "color"), 78: ("Gray", "color"),
    79: ("Green", "color"), 80: ("Halter Tops", "category"), 81: ("Harem Pants", "category"),
    82: ("Hearts", "pattern"), 83: ("Heels", "category"), 84: ("Herringbone", "pattern"),
    85: ("Hi-Lo", "style"), 86: ("Hiking Boots", "category"), 87: ("Hollow-Out", "style"),
    88: ("Hoodies & Sweatshirts", "category"), 89: ("Hosiery, Stockings, Tights", "category"),
    90: ("Houndstooth", "pattern"), 91: ("Jackets", "category"), 92: ("Jeans", "category"),
    93: ("Jerseys", "category"), 94: ("Jilbaab", "category"), 95: ("Jumpsuits Overalls & Rompers", "category"),
    96: ("Kimonos", "category"), 97: ("Knit", "material"), 98: ("Lace", "material"),
    99: ("Leather", "material"), 100: ("Leggings", "category"), 101: ("Leopard And Cheetah", "pattern"),
    102: ("Linen", "material"), 103: ("Lingerie Sleepwear & Underwear", "category"),
    104: ("Loafers & Slip-on Shoes", "category"), 105: ("Long Sleeved", "sleeve"),
    106: ("Male", "gender"), 107: ("Marbled", "pattern"), 108: ("Maroon", "color"),
    109: ("Maternity", "category"), 110: ("Mesh", "pattern"), 111: ("Multi Color", "color"),
    112: ("Neoprene", "material"), 113: ("Neutral", "gender"), 114: ("Nightgowns", "category"),
    115: ("Nylon", "material"), 116: ("Off The Shoulder", "neckline"), 117: ("Orange", "color"),
    118: ("Organza", "material"), 119: ("Padded Bras", "category"), 120: ("Paisley", "pattern"),
    121: ("Pajamas", "category"), 122: ("Party Dresses", "category"), 123: ("Pasties", "category"),
    124: ("Patent", "material"), 125: ("Peach", "color"), 126: ("Peacoats", "category"),
    127: ("Pencil Skirts", "category"), 128: ("Peplum", "style"), 129: ("Petticoats", "category"),
    130: ("Pin Stripes", "pattern"), 131: ("Pink", "color"), 132: ("Plaid", "pattern"),
    133: ("Pleated", "style"), 134: ("Plush", "material"), 135: ("Polka Dot", "pattern"),
    136: ("Polos", "category"), 137: ("Polyester", "material"), 138: ("Printed", "style"),
    139: ("Prom Dresses", "category"), 140: ("Puff Sleeves", "sleeve"),
    141: ("Pullover Sweaters", "category"), 142: ("Purple", "color"), 143: ("Quilted", "pattern"),
    144: ("Racerback", "neckline"), 145: ("Rain Boots", "category"), 146: ("Raincoats", "category"),
    147: ("Rayon", "material"), 148: ("Red", "color"), 149: ("Reversible", "style"),
    150: ("Rhinestone Studded", "style"), 151: ("Ripped", "pattern"), 152: ("Robes", "category"),
    153: ("Round Neck", "neckline"), 154: ("Ruched", "pattern"), 155: ("Ruffles", "pattern"),
    156: ("Running Shoes", "category"), 157: ("Sandals", "category"), 158: ("Satin", "material"),
    159: ("Sequins", "pattern"), 160: ("Sheer Tops", "category"), 161: ("Shoe Accessories", "category"),
    162: ("Shoe Inserts", "category"), 163: ("Shoelaces", "category"), 164: ("Short Sleeves", "sleeve"),
    165: ("Shorts", "category"), 166: ("Shoulder Drapes", "neckline"), 167: ("Silk", "material"),
    168: ("Silver", "color"), 169: ("Skinny Jeans", "category"), 170: ("Skirts", "category"),
    171: ("Sleeveless", "sleeve"), 172: ("Slippers", "category"), 173: ("Snakeskin", "pattern"),
    174: ("Sneakers", "category"), 175: ("Spaghetti Straps", "style"), 176: ("Spandex", "material"),
    177: ("Sports Bras", "category"), 178: ("Square Necked", "neckline"), 179: ("Stilettos", "category"),
    180: ("Strapless", "sleeve"), 181: ("Stripes", "pattern"), 182: ("Suede", "material"),
    183: ("Suits & Blazers", "category"), 184: ("Summer", "style"), 185: ("Sweatpants", "category"),
    186: ("Sweetheart Neckline", "neckline"), 187: ("Swim Trunks", "category"),
    188: ("Swimsuit Cover-ups", "category"), 189: ("Swimsuits", "category"), 190: ("T-Shirts", "category"),
    191: ("Taffeta", "material"), 192: ("Tan", "color"), 193: ("Tank Tops", "category"),
    194: ("Teal", "color"), 195: ("Thermal Underwear", "category"), 196: ("Thigh Highs", "category"),
    197: ("Thongs", "category"), 198: ("Three Piece Suits", "category"), 199: ("Tie Dye", "pattern"),
    200: ("Trench Coats", "category"), 201: ("Trousers", "category"), 202: ("Tube Tops", "category"),
    203: ("Tulle", "material"), 204: ("Tunic", "style"), 205: ("Turtlenecks", "neckline"),
    206: ("Tutus", "category"), 207: ("Tweed", "material"), 208: ("Twill", "material"),
    209: ("Two-Tone", "style"), 210: ("U-Necks", "neckline"), 211: ("Undershirts", "category"),
    212: ("Underwear", "category"), 213: ("Uniforms", "category"), 214: ("V-Necks", "neckline"),
    215: ("Velour", "material"), 216: ("Velvet", "material"), 217: ("Vests", "category"),
    218: ("Vintage Retro", "style"), 219: ("Vinyl", "material"), 220: ("Wedding Dresses", "category"),
    221: ("Wedges & Platforms", "category"), 222: ("White", "color"), 223: ("Winter Boots", "category"),
    224: ("Wool", "material"), 225: ("Wrap", "style"), 226: ("Yellow", "color"),
    227: ("Yoga Pants", "category"), 228: ("Zebra", "pattern"),
}


def convert_imat_labels(label_ids: list[int]) -> dict | None:
    """
    Convert a list of iMaterialist label IDs into our schema.
    Returns None if the item should be skipped (unmapped category, swimwear, etc.)

    Output matches our GPT-4o-mini vision.py schema:
    {
        "category": "top",
        "primary_color": "black",
        "material": "cotton",
        "style_tags": ["casual", "sporty"],
        "occasion_tags": ["everyday", "gym"],
    }
    """
    result = {
        "category": None,
        "primary_color": "unknown",
        "material": None,
        "style_tags": set(),
        "occasion_tags": set(),
    }

    imat_category_name = None

    for lid in label_ids:
        if lid not in LABEL_ID_TO_NAME:
            continue

        name, group = LABEL_ID_TO_NAME[lid]

        if group == "category":
            mapped = CATEGORY_MAP.get(name)
            if mapped is None:
                return None  # skip swimwear, underwear, etc.
            if result["category"] is None:
                result["category"] = mapped
                imat_category_name = name

            # Derive style + occasion from granular category
            for tag in CATEGORY_TO_STYLE.get(name, []):
                result["style_tags"].add(tag)
            for tag in CATEGORY_TO_OCCASION.get(name, []):
                result["occasion_tags"].add(tag)

        elif group == "color":
            mapped = COLOR_MAP.get(name, "unknown")
            if result["primary_color"] == "unknown":
                result["primary_color"] = mapped

        elif group == "material":
            mapped = MATERIAL_MAP.get(name)
            if mapped and result["material"] is None:
                result["material"] = mapped

        elif group == "style":
            for tag in IMAT_STYLE_TO_OURS.get(name, []):
                result["style_tags"].add(tag)

    # Must have at least a category to be useful
    if result["category"] is None:
        return None

    # Default occasion if none derived
    if not result["occasion_tags"]:
        result["occasion_tags"].add("everyday")

    # Default style if none derived
    if not result["style_tags"]:
        result["style_tags"].add("casual")

    # Convert sets to sorted lists for deterministic output
    result["style_tags"] = sorted(result["style_tags"])
    result["occasion_tags"] = sorted(result["occasion_tags"])

    return result


def format_as_training_target(labels: dict) -> str:
    """
    Format the mapped labels as the JSON string Florence-2 should learn to output.
    Matches the exact schema from services/vision.py.
    """
    import json
    return json.dumps(labels, separators=(",", ":"))


if __name__ == "__main__":
    # Quick test
    examples = [
        [5, 17, 137, 164],        # Athletic Shirts + Black + Polyester + Short Sleeves
        [36, 131, 53, 70, 171],   # Casual Dresses + Pink + Cotton + Floral + Sleeveless
        [47, 148, 158, 20],       # Clubbing Dresses + Red + Satin + Bodycon
        [174, 222, 99],           # Sneakers + White + Leather
        [183, 78, 224],           # Suits & Blazers + Gray + Wool
        [227, 19, 176],           # Yoga Pants + Blue + Spandex
    ]

    for label_ids in examples:
        result = convert_imat_labels(label_ids)
        names = [LABEL_ID_TO_NAME[lid][0] for lid in label_ids]
        print(f"Input: {names}")
        if result:
            print(f"Output: {format_as_training_target(result)}")
        else:
            print("Output: SKIPPED")
        print()
