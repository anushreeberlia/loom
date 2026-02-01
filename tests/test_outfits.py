#!/usr/bin/env python3
"""
Lightweight test harness for outfit generation.
Runs a set of test images and checks for common violations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

# Test fixtures - use existing catalog images as test inputs
TEST_IMAGES = [
    "catalog/images/20276.jpg",   # Top
    "catalog/images/27717.jpg",   # Top  
    "catalog/images/43811.jpg",   # Top
    "catalog/images/46596.jpg",   # Top
    "catalog/images/57082.jpg",   # Top
    "catalog/images/7019.jpg",    # Top
    "catalog/images/32685.jpg",   # Top
    "catalog/images/32655.jpg",   # Top
    "catalog/images/58518.jpg",   # Top
    "catalog/images/20519.jpg",   # Top
]

# Violation checks
BANNED_KEYWORDS = {
    "swimsuit", "swimwear", "bikini", "swim",
    "hosiery", "stockings", "tights", "socks",
    "girl's", "girls", "kid", "kids", "children", "boy",
    "dupatta", "innerwear", "underwear", "bra", "lingerie",
    "sleepwear", "nightwear", "pyjama", "pajama",
}

NEUTRALS = {"black", "white", "grey", "gray", "beige", "tan", "cream", "navy", "brown", "khaki"}


def check_banned_keywords(item_name: str) -> list[str]:
    """Check if item name contains banned keywords."""
    name_lower = item_name.lower()
    violations = []
    for kw in BANNED_KEYWORDS:
        if kw in name_lower:
            violations.append(f"banned keyword '{kw}'")
    return violations


def check_same_neutral(outfit: dict) -> list[str]:
    """Check for same-neutral violation in Trendy outfit."""
    violations = []
    if outfit.get("direction") != "Trendy":
        return violations
    
    items = outfit.get("items", [])
    bottom = next((i for i in items if i.get("slot") == "bottom"), None)
    shoes = next((i for i in items if i.get("slot") == "shoes"), None)
    
    if bottom and shoes:
        bottom_color = bottom.get("primary_color", "").lower()
        shoes_color = shoes.get("primary_color", "").lower()
        
        if (bottom_color in NEUTRALS and 
            shoes_color in NEUTRALS and 
            bottom_color == shoes_color and
            bottom_color not in {"black", "white"}):  # black/white same is okay
            violations.append(f"same neutral: bottom={bottom_color}, shoes={shoes_color}")
    
    return violations


def check_repeated_items(outfits: list[dict]) -> list[str]:
    """Check if same item ID appears in multiple outfits."""
    violations = []
    seen_ids = {}  # id -> list of outfit indices
    
    for idx, outfit in enumerate(outfits):
        for item in outfit.get("items", []):
            # Support both "id" and "item_id" formats
            item_id = item.get("id") or item.get("item_id")
            if item_id:
                if item_id not in seen_ids:
                    seen_ids[item_id] = []
                seen_ids[item_id].append(idx)
    
    for item_id, indices in seen_ids.items():
        if len(indices) > 1:
            violations.append(f"item {item_id} repeated in outfits {indices}")
    
    return violations


def check_subtype_diversity(outfits: list[dict]) -> list[str]:
    """Check if all outfits have same bottom/shoe subtype."""
    violations = []
    
    for slot in ["bottom", "shoes"]:
        subtypes = []
        for outfit in outfits:
            for item in outfit.get("items", []):
                if item.get("slot") == slot:
                    name_lower = item.get("name", "").lower()
                    # Extract subtype hint from name
                    if "skirt" in name_lower:
                        subtypes.append("skirt")
                    elif "jeans" in name_lower:
                        subtypes.append("jeans")
                    elif "trousers" in name_lower or "pants" in name_lower:
                        subtypes.append("trousers")
                    elif "shorts" in name_lower:
                        subtypes.append("shorts")
                    elif "heels" in name_lower:
                        subtypes.append("heels")
                    elif "flats" in name_lower:
                        subtypes.append("flats")
                    elif "sneakers" in name_lower:
                        subtypes.append("sneakers")
                    elif "sandals" in name_lower:
                        subtypes.append("sandals")
                    elif "boots" in name_lower:
                        subtypes.append("boots")
                    else:
                        subtypes.append("other")
        
        if len(subtypes) == 3 and len(set(subtypes)) == 1:
            violations.append(f"all 3 outfits have same {slot} subtype: {subtypes[0]}")
    
    return violations


def run_test(image_path: str, verbose: bool = True) -> dict:
    """Run a single test and return results."""
    import httpx
    
    result = {
        "image": image_path,
        "success": False,
        "violations": [],
        "base_item": None,
        "outfits": [],
    }
    
    full_path = Path(image_path)
    if not full_path.exists():
        result["violations"].append(f"Image not found: {image_path}")
        return result
    
    try:
        with open(full_path, "rb") as f:
            files = {"file": (full_path.name, f, "image/jpeg")}
            response = httpx.post(
                "http://localhost:8000/v1/outfits:generate",
                files=files,
                timeout=120.0
            )
        
        if response.status_code != 200:
            result["violations"].append(f"API error: {response.status_code} - {response.text[:200]}")
            return result
        
        data = response.json()
        result["success"] = True
        result["base_item"] = data.get("base_item", {})
        result["outfits"] = data.get("outfits", [])
        
        # Run violation checks
        for outfit in result["outfits"]:
            # Check API-reported violations
            api_violations = outfit.get("violations", [])
            for v in api_violations:
                result["violations"].append(f"{outfit['direction']}: {v}")
            
            # Check each item for banned keywords (redundant but good sanity check)
            for item in outfit.get("items", []):
                kw_violations = check_banned_keywords(item.get("name", ""))
                for v in kw_violations:
                    result["violations"].append(f"{outfit['direction']}/{item['slot']}: {v}")
            
            # Check same-neutral
            neutral_violations = check_same_neutral(outfit)
            result["violations"].extend(neutral_violations)
        
        # Check repeated items across outfits
        repeated = check_repeated_items(result["outfits"])
        result["violations"].extend(repeated)
        
        # Check subtype diversity
        diversity = check_subtype_diversity(result["outfits"])
        result["violations"].extend(diversity)
        
    except Exception as e:
        result["violations"].append(f"Exception: {str(e)}")
    
    return result


def print_result(result: dict, verbose: bool = True):
    """Pretty print test result."""
    print(f"\n{'='*60}")
    print(f"Image: {result['image']}")
    print(f"{'='*60}")
    
    if not result["success"]:
        print(f"❌ FAILED: {result['violations']}")
        return
    
    # Base item
    base = result["base_item"]
    print(f"\n📦 Base Item:")
    print(f"   Category: {base.get('category', 'unknown')}")
    print(f"   Color: {base.get('primary_color', 'unknown')}")
    print(f"   Style: {', '.join(base.get('style_tags', []))}")
    
    # Outfits
    for outfit in result["outfits"]:
        direction = outfit.get("direction", "?")
        score = outfit.get("score", "?")
        print(f"\n🎨 {direction} Outfit (score: {score}):")
        for item in outfit.get("items", []):
            slot = item.get("slot", "?")
            name = item.get("name", "?")[:50]
            item_id = item.get("id", "?")
            color = item.get("primary_color", "?")
            print(f"   [{slot:10}] #{item_id}: {name} ({color})")
        
        # Show score breakdown if available
        breakdown = outfit.get("score_breakdown", {})
        if breakdown and verbose:
            print(f"   Score: sim={breakdown.get('sim_weighted', 0):.2f} + dir_bonus={breakdown.get('direction_bonus', 0):.2f} - penalties={breakdown.get('color_penalty', 0) + breakdown.get('formality_penalty', 0):.2f}")
    
    # Violations
    if result["violations"]:
        print(f"\n⚠️  VIOLATIONS ({len(result['violations'])}):")
        for v in result["violations"]:
            print(f"   - {v}")
    else:
        print(f"\n✅ No violations")


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test outfit generation")
    parser.add_argument("--image", "-i", help="Test single image")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show violations")
    parser.add_argument("--limit", "-n", type=int, default=len(TEST_IMAGES), help="Limit number of tests")
    args = parser.parse_args()
    
    if args.image:
        images = [args.image]
    else:
        images = TEST_IMAGES[:args.limit]
    
    print(f"Running {len(images)} tests...")
    
    total_violations = 0
    passed = 0
    failed = 0
    
    for img in images:
        result = run_test(img)
        
        if not args.quiet or result["violations"]:
            print_result(result, verbose=not args.quiet)
        
        if result["success"] and not result["violations"]:
            passed += 1
        else:
            failed += 1
            total_violations += len(result["violations"])
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(images)} passed, {failed} failed")
    print(f"Total violations: {total_violations}")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

