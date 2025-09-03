# src/pdf_processor.py

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


# Default directories (resolved relative to this file)
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"


def ensure_dirs(output_dir: Path = TEMP_DIR) -> None:
    """Create output directories if they don't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def render_page_to_png(page: fitz.Page, output_path: Path, dpi: int = 150) -> Path:
    """
    Render a PDF page to PNG at specified DPI.
    
    Args:
        page: PyMuPDF page object
        output_path: Path where PNG will be saved
        dpi: Resolution for rendering (default 150)
    
    Returns:
        Path to the saved PNG file
    """
    # Calculate scaling matrix for desired DPI (PDF default is 72 DPI)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    
    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Save as PNG
    pix.save(output_path.as_posix())
    
    return output_path


def extract_blocks_with_fonts(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract text blocks with font information from a PDF page.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        List of block dictionaries with bbox and spans
    """
    # Get structured text data
    page_dict = page.get_text("dict")
    
    blocks = []
    
    for block in page_dict.get("blocks", []):
        # Skip image blocks (type 1)
        if block.get("type", 0) != 0:
            continue
        
        # Get block bbox
        bbox = block.get("bbox", [0, 0, 0, 0])
        bbox = [round(coord, 2) for coord in bbox]
        
        # Extract spans from lines
        spans = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                
                # Skip empty spans
                if not text:
                    continue
                
                # Extract font information
                font_size = round(span.get("size", 0), 1)
                font_name = span.get("font", "Unknown")
                
                # Clean up font name (remove subset prefix if present)
                if "+" in font_name:
                    font_name = font_name.split("+")[-1]
                
                spans.append({
                    "text": text,
                    "font_size": font_size,
                    "font_name": font_name
                })
        
        # Only add blocks with actual text content
        if spans:
            blocks.append({
                "bbox": bbox,
                "spans": spans
            })
    
    return blocks


def process_single_page(
    page: fitz.Page,
    page_num: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process a single PDF page: extract blocks and render image.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number (1-based)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with page data
    """
    # Get page dimensions
    rect = page.rect
    width = int(rect.width)
    height = int(rect.height)
    
    # Generate filenames
    image_name = f"page_{page_num:03d}.png"
    json_name = f"page_{page_num:03d}.json"
    
    # Render page to PNG
    image_path = output_dir / image_name
    render_page_to_png(page, image_path, dpi=150)
    
    # Extract blocks with font information
    blocks = extract_blocks_with_fonts(page)
    
    # Create page data structure
    page_data = {
        "page": page_num,
        "image_name": image_name,
        "width": width,
        "height": height,
        "blocks": blocks
    }
    
    # Save individual page JSON
    json_path = output_dir / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    
    return page_data


def extract_pdf(
    pdf_path: str | Path,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Extract blocks with font information from all pages of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save outputs (default: output/temp/)
    
    Returns:
        List of page data dictionaries
    """
    # Convert to Path object and validate
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = TEMP_DIR
    else:
        output_dir = Path(output_dir)
    
    ensure_dirs(output_dir)
    
    # Process PDF
    all_pages_data = []
    
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            
            for page_num in range(1, total_pages + 1):
                # Get page (0-based indexing in PyMuPDF)
                page = doc[page_num - 1]
                
                # Process page
                page_data = process_single_page(page, page_num, output_dir)
                all_pages_data.append(page_data)
                
                # Progress indicator
                print(f"Processed page {page_num}/{total_pages}")
        
        # Save combined JSON for all pages
        combined_json_path = output_dir / "document_all.json"
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(all_pages_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Success! Processed {total_pages} pages.")
        print(f"📁 JSON + images saved in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise
    
    return all_pages_data


def main():
    """CLI entry point for the PDF processor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/pdf_processor.py <PDF_PATH> [OUTPUT_DIR]")
        print("\nExample:")
        print('  python src/pdf_processor.py "document.pdf"')
        print('  python src/pdf_processor.py "document.pdf" "custom/output/path"')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Optional output directory argument
    output_dir = None
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    
    try:
        # Process the PDF
        extract_pdf(pdf_path, output_dir)
        
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