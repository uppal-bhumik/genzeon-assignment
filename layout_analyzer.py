# src/layout_analyzer.py

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from statistics import mean, median


# Default directories (resolved relative to this file)
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# Classification thresholds and keywords
HEADER_FONT_SIZE_THRESHOLD = 16.0
HEADER_SECONDARY_THRESHOLD = 14.0  # For subtitles/subheaders
TABLE_KEYWORDS = [
    "table", "total", "sum", "amount", "quantity", "price",
    "column", "row", "$", "%", "rate", "balance", "debit", "credit"
]
FOOTER_KEYWORDS = ["page", "copyright", "©", "all rights reserved", "confidential"]
FIGURE_KEYWORDS = ["figure", "fig.", "chart", "graph", "diagram", "illustration"]

# Layout thresholds
PAGE_WIDTH_COVERAGE_THRESHOLD = 0.75  # 75% of page width suggests table/header
FOOTER_POSITION_THRESHOLD = 0.9  # Bottom 10% of page
HEADER_POSITION_THRESHOLD = 0.15  # Top 15% of page


def normalize_bbox(
    bbox: List[float], 
    page_width: float, 
    page_height: float
) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Absolute coordinates [x0, y0, x1, y1]
        page_width: Page width in absolute units
        page_height: Page height in absolute units
    
    Returns:
        Normalized bbox [x0_norm, y0_norm, x1_norm, y1_norm]
    """
    x0, y0, x1, y1 = bbox
    
    # Ensure coordinates are within page bounds
    x0 = max(0, min(x0, page_width))
    x1 = max(0, min(x1, page_width))
    y0 = max(0, min(y0, page_height))
    y1 = max(0, min(y1, page_height))
    
    # Normalize to [0, 1]
    return [
        round(x0 / page_width, 4),
        round(y0 / page_height, 4),
        round(x1 / page_width, 4),
        round(y1 / page_height, 4)
    ]


def get_block_metrics(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate various metrics for a block to aid classification.
    
    Args:
        block: Block dictionary with spans
    
    Returns:
        Dictionary with calculated metrics
    """
    spans = block.get("spans", [])
    
    if not spans:
        return {
            "avg_font_size": 0,
            "max_font_size": 0,
            "min_font_size": 0,
            "median_font_size": 0,
            "text_length": 0,
            "num_spans": 0,
            "has_bold": False,
            "has_italic": False,
            "unique_fonts": set()
        }
    
    font_sizes = [span.get("font_size", 0) for span in spans]
    font_names = [span.get("font_name", "") for span in spans]
    full_text = " ".join(span.get("text", "") for span in spans)
    
    # Check for bold/italic indicators in font names
    has_bold = any("bold" in name.lower() for name in font_names)
    has_italic = any("italic" in name.lower() or "oblique" in name.lower() for name in font_names)
    
    return {
        "avg_font_size": mean(font_sizes) if font_sizes else 0,
        "max_font_size": max(font_sizes) if font_sizes else 0,
        "min_font_size": min(font_sizes) if font_sizes else 0,
        "median_font_size": median(font_sizes) if font_sizes else 0,
        "text_length": len(full_text),
        "num_spans": len(spans),
        "has_bold": has_bold,
        "has_italic": has_italic,
        "unique_fonts": set(font_names)
    }


def classify_block(
    block: Dict[str, Any],
    normalized_bbox: List[float],
    page_width: float,
    page_height: float
) -> str:
    """
    Classify a block based on heuristic rules.
    
    Args:
        block: Block dictionary with bbox and spans
        normalized_bbox: Normalized bounding box [0-1]
        page_width: Page width in absolute units
        page_height: Page height in absolute units
    
    Returns:
        Classification label: "Header", "Table", "Text", "Footer", or "Figure"
    """
    # Get block metrics
    metrics = get_block_metrics(block)
    
    # Concatenate all text for keyword analysis
    full_text = " ".join(span.get("text", "") for span in block.get("spans", []))
    text_lower = full_text.lower()
    
    # Calculate spatial features
    x0_norm, y0_norm, x1_norm, y1_norm = normalized_bbox
    block_width_ratio = x1_norm - x0_norm
    block_height_ratio = y1_norm - y0_norm
    vertical_position = (y0_norm + y1_norm) / 2  # Center Y position
    
    # 1. Check for Footer (bottom of page + keywords)
    if vertical_position > FOOTER_POSITION_THRESHOLD:
        if any(keyword in text_lower for keyword in FOOTER_KEYWORDS):
            return "Footer"
        # Small text at bottom might still be footer
        if metrics["avg_font_size"] < 10 and block_height_ratio < 0.05:
            return "Footer"
    
    # 2. Check for Header (large font OR top position + bold)
    if metrics["avg_font_size"] >= HEADER_FONT_SIZE_THRESHOLD:
        return "Header"
    
    # Secondary header check: moderately large font + bold/position
    if metrics["avg_font_size"] >= HEADER_SECONDARY_THRESHOLD:
        if metrics["has_bold"] or vertical_position < HEADER_POSITION_THRESHOLD:
            return "Header"
    
    # Check for short, bold text that might be headers
    if metrics["text_length"] < 100 and metrics["has_bold"] and metrics["avg_font_size"] > 12:
        return "Header"
    
    # 3. Check for Figure (keywords + positioning)
    if any(keyword in text_lower for keyword in FIGURE_KEYWORDS):
        # Figure captions are usually small text
        if metrics["avg_font_size"] < 11 or metrics["has_italic"]:
            return "Figure"
    
    # 4. Check for Table
    # Check for table keywords
    table_keyword_count = sum(1 for keyword in TABLE_KEYWORDS if keyword in text_lower)
    
    # Strong table indicators
    if table_keyword_count >= 3:
        return "Table"
    
    # Check if block spans most of page width (common for tables)
    if block_width_ratio > PAGE_WIDTH_COVERAGE_THRESHOLD and table_keyword_count >= 1:
        return "Table"
    
    # Check for numeric patterns (common in tables)
    numeric_pattern = re.findall(r'\d+[.,]?\d*', full_text)
    if len(numeric_pattern) > 5 and table_keyword_count >= 1:
        return "Table"
    
    # Check for structured patterns (multiple short spans might indicate table cells)
    if metrics["num_spans"] > 10 and metrics["avg_font_size"] < 12:
        avg_span_length = metrics["text_length"] / metrics["num_spans"]
        if avg_span_length < 20:  # Short spans suggest table cells
            return "Table"
    
    # 5. Default to Text
    return "Text"


def process_page(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single page: normalize bboxes, classify blocks, extract text.
    
    Args:
        page_data: Page dictionary from document_all.json
    
    Returns:
        Processed page with sections
    """
    page_num = page_data.get("page", 1)
    image_name = page_data.get("image_name", f"page_{page_num:03d}.png")
    width = page_data.get("width", 0)
    height = page_data.get("height", 0)
    blocks = page_data.get("blocks", [])
    
    sections = []
    
    for block_idx, block in enumerate(blocks):
        # Get original bbox
        bbox = block.get("bbox", [0, 0, 0, 0])
        
        # Normalize bbox
        normalized_bbox = normalize_bbox(bbox, width, height)
        
        # Classify block
        block_class = classify_block(block, normalized_bbox, width, height)
        
        # Concatenate all span texts
        spans = block.get("spans", [])
        text_parts = []
        for span in spans:
            span_text = span.get("text", "").strip()
            if span_text:
                text_parts.append(span_text)
        
        full_text = " ".join(text_parts)
        
        # Skip empty blocks
        if not full_text:
            continue
        
        # Create section entry
        section = {
            "bbox": normalized_bbox,
            "class": block_class,
            "text": full_text
        }
        
        sections.append(section)
    
    # Return processed page
    return {
        "image_name": image_name,
        "width": width,
        "height": height,
        "sections": sections
    }


def analyze_layout(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Main orchestrator: load document_all.json, process all pages, save results.
    
    Args:
        input_path: Path to document_all.json (default: output/temp/document_all.json)
        output_dir: Directory to save outputs (default: output/temp/)
    
    Returns:
        List of processed page dictionaries
    """
    # Set default paths
    if input_path is None:
        input_path = TEMP_DIR / "document_all.json"
    else:
        input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = TEMP_DIR
    else:
        output_dir = Path(output_dir)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please run pdf_processor.py first to generate document_all.json"
        )
    
    # Load document data
    print(f"📄 Loading document data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        document_data = json.load(f)
    
    if not isinstance(document_data, list):
        raise ValueError("Expected document_all.json to contain a list of pages")
    
    total_pages = len(document_data)
    print(f"📊 Found {total_pages} pages to analyze\n")
    
    # Process each page
    processed_pages = []
    
    for page_idx, page_data in enumerate(document_data, 1):
        # Process page
        processed_page = process_page(page_data)
        processed_pages.append(processed_page)
        
        # Save individual page layout
        page_num = page_data.get("page", page_idx)
        layout_json_path = output_dir / f"layout_page_{page_num:03d}.json"
        with open(layout_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_page, f, ensure_ascii=False, indent=2)
        
        # Progress indicator with statistics
        num_sections = len(processed_page["sections"])
        class_counts = {}
        for section in processed_page["sections"]:
            cls = section["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        stats_str = ", ".join(f"{cls}: {count}" for cls, count in sorted(class_counts.items()))
        print(f"✓ Processed layout for page {page_idx}/{total_pages} - {num_sections} sections ({stats_str})")
    
    # Save combined layout file
    layout_all_path = output_dir / "layout_all.json"
    with open(layout_all_path, "w", encoding="utf-8") as f:
        json.dump(processed_pages, f, ensure_ascii=False, indent=2)
    
    # Summary statistics
    print("\n" + "="*60)
    print("📈 ANALYSIS COMPLETE - Summary Statistics:")
    print("="*60)
    
    total_sections = sum(len(page["sections"]) for page in processed_pages)
    all_classes = {}
    for page in processed_pages:
        for section in page["sections"]:
            cls = section["class"]
            all_classes[cls] = all_classes.get(cls, 0) + 1
    
    print(f"📄 Total pages processed: {total_pages}")
    print(f"📦 Total sections extracted: {total_sections}")
    print(f"📊 Average sections per page: {total_sections/total_pages:.1f}")
    print("\n🏷️  Section breakdown:")
    for cls, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_sections) * 100
        print(f"   • {cls:8s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n✅ Success! Layout analysis complete.")
    print(f"📁 Results saved in: {output_dir}")
    print(f"   • Per-page layouts: layout_page_XXX.json")
    print(f"   • Combined layout: layout_all.json")
    
    return processed_pages


def main():
    """CLI entry point for the layout analyzer."""
    import sys
    
    # Parse command line arguments
    input_path = None
    output_dir = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Usage: python src/layout_analyzer.py [INPUT_PATH] [OUTPUT_DIR]")
            print("\nAnalyze layout from document_all.json and classify text blocks.")
            print("\nArguments:")
            print("  INPUT_PATH   Path to document_all.json (default: output/temp/document_all.json)")
            print("  OUTPUT_DIR   Directory for output files (default: output/temp/)")
            print("\nExamples:")
            print("  python src/layout_analyzer.py")
            print("  python src/layout_analyzer.py custom/document_all.json")
            print("  python src/layout_analyzer.py document_all.json custom/output/")
            sys.exit(0)
        
        input_path = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    try:
        # Run layout analysis
        analyze_layout(input_path, output_dir)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()