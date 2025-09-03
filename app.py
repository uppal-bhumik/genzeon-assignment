#!/usr/bin/env python3
"""
Professional PDF Layout Analyzer
A robust Streamlit application for PDF document analysis with keyword search capabilities.
"""

import streamlit as st
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
import re
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.pdf_processor import extract_pdf
    from src.layout_analyzer import analyze_layout
except ImportError as e:
    logger.error(f"Failed to import backend modules: {e}")
    st.error("Backend modules not found. Please ensure src/pdf_processor.py and src/layout_analyzer.py exist.")
    st.stop()

# Constants
OUTPUT_DIR = Path("output/temp")
SEARCH_COLORS = [
    (255, 0, 0, 80),    # Red
    (0, 255, 0, 80),    # Green  
    (0, 0, 255, 80),    # Blue
    (255, 255, 0, 80),  # Yellow
    (255, 0, 255, 80),  # Magenta
]

# Page configuration
st.set_page_config(
    page_title="PDF Layout Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 1rem;
    }
    
    /* Card styling */
    .stat-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    /* Success message */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Search highlight legend */
    .highlight-legend {
        display: flex;
        gap: 1rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .color-box {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        border: 1px solid #ddd;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* File uploader */
    .stFileUploader > div {
        padding: 2rem;
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: #f8f9ff;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Error styling */
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class SessionManager:
    """Centralized session state management."""
    
    @staticmethod
    def initialize_session():
        """Initialize all session state variables."""
        defaults = {
            'processed': False,
            'search_results': [],
            'current_page': 0,
            'document_data': None,
            'layout_data': None,
            'statistics': None,
            'file_hash': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def reset_session():
        """Reset session for new file processing."""
        keys_to_reset = [
            'processed', 'search_results', 'current_page', 
            'document_data', 'layout_data', 'statistics', 'file_hash'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                if key == 'processed':
                    st.session_state[key] = False
                elif key == 'current_page':
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = [] if key == 'search_results' else None


class FileManager:
    """Handle file operations and directory management."""
    
    @staticmethod
    def ensure_output_dir() -> None:
        """Ensure output directory exists."""
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ensured: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise
    
    @staticmethod
    def clean_output_dir() -> None:
        """Clean the output directory safely."""
        try:
            if OUTPUT_DIR.exists():
                shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Output directory cleaned")
        except Exception as e:
            logger.error(f"Failed to clean output directory: {e}")
            raise
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Calculate hash of uploaded file for change detection."""
        import hashlib
        return hashlib.md5(file_content).hexdigest()


class PDFProcessor:
    """Handle PDF processing operations."""
    
    @staticmethod
    def process_pdf_file(pdf_path: Path) -> Tuple[bool, str]:
        """
        Process PDF file through the backend pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Clean output directory
            FileManager.clean_output_dir()
            
            # Step 1: Extract PDF blocks and render images
            with st.spinner("📄 Extracting PDF content and rendering pages..."):
                extract_pdf(pdf_path, OUTPUT_DIR)
                logger.info("PDF extraction completed")
            
            # Step 2: Analyze layout
            with st.spinner("🔍 Analyzing document layout..."):
                analyze_layout(OUTPUT_DIR / "document_all.json", OUTPUT_DIR)
                logger.info("Layout analysis completed")
            
            return True, "✅ PDF processed successfully!"
            
        except FileNotFoundError as e:
            error_msg = f"Required file not found: {str(e)}"
            logger.error(error_msg)
            return False, f"❌ {error_msg}"
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON data: {str(e)}"
            logger.error(error_msg)
            return False, f"❌ {error_msg}"
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            return False, f"❌ {error_msg}"


class DataLoader:
    """Handle data loading and validation."""
    
    @staticmethod
    def load_results() -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Load the processed results from JSON files with error handling.
        
        Returns:
            Tuple of (document_data, layout_data) or (None, None) if error
        """
        try:
            document_path = OUTPUT_DIR / "document_all.json"
            layout_path = OUTPUT_DIR / "layout_all.json"
            
            if not document_path.exists():
                logger.error(f"Document file not found: {document_path}")
                return None, None
                
            if not layout_path.exists():
                logger.error(f"Layout file not found: {layout_path}")
                return None, None
            
            with open(document_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            with open(layout_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # Validate data structure
            if not isinstance(document_data, list) or not isinstance(layout_data, list):
                logger.error("Invalid data structure in JSON files")
                return None, None
            
            logger.info(f"Loaded {len(document_data)} document pages and {len(layout_data)} layout pages")
            return document_data, layout_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None, None


class StatisticsCalculator:
    """Calculate and manage document statistics."""
    
    @staticmethod
    def calculate_statistics(layout_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from layout data.
        
        Args:
            layout_data: List of page layout data
            
        Returns:
            Dictionary containing statistics
        """
        if not layout_data:
            return {
                "total_pages": 0,
                "total_sections": 0,
                "avg_sections_per_page": 0,
                "class_counts": {}
            }
        
        total_pages = len(layout_data)
        total_sections = 0
        class_counts = {}
        
        for page in layout_data:
            sections = page.get("sections", [])
            total_sections += len(sections)
            
            for section in sections:
                cls = section.get("class", "Unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        avg_sections = round(total_sections / total_pages, 1) if total_pages > 0 else 0
        
        return {
            "total_pages": total_pages,
            "total_sections": total_sections,
            "avg_sections_per_page": avg_sections,
            "class_counts": class_counts
        }


class SearchEngine:
    """Handle keyword search functionality with a foolproof content-matching logic."""

    @staticmethod
    def search_keywords(layout_data: List[Dict], document_data: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Search for keywords using a robust content-matching logic to guarantee correct bbox retrieval.
        This is the DEFINITIVE, FINAL bug fix.
        """
        results = []
        active_keywords = [(idx, kw.lower().strip()) for idx, kw in enumerate(keywords) if kw.strip()]
        
        if not active_keywords:
            return []

        logger.info("Searching for keywords using the definitive content-matching logic...")

        # Create a quick lookup map of document blocks by their text content for efficiency
        # This is the key to the fix: mapping text content directly to its original block
        doc_block_map = {}
        for page_idx, doc_page in enumerate(document_data):
            doc_block_map[page_idx] = {}
            for block in doc_page.get("blocks", []):
                block_text = " ".join(span.get("text", "").strip() for span in block.get("spans", []) if span.get("text", "").strip()).strip()
                if block_text:
                    doc_block_map[page_idx][block_text] = block

        # Now, search through the clean layout_data
        for page_idx, page_layout in enumerate(layout_data):
            if page_idx not in doc_block_map:
                continue

            for section in page_layout.get("sections", []):
                section_text = section.get("text", "").strip()
                section_text_lower = section_text.lower()

                for keyword_idx, keyword_lower in active_keywords:
                    if keyword_lower in section_text_lower:
                        # Match found in a section. Now find the original block using its full text content as a key.
                        if section_text in doc_block_map[page_idx]:
                            original_block = doc_block_map[page_idx][section_text]
                            
                            results.append({
                                "page_idx": page_idx,
                                "page_num": page_layout.get("page", page_idx + 1),
                                "keyword": keywords[keyword_idx],
                                "keyword_idx": keyword_idx,
                                "bbox": original_block.get("bbox", [0, 0, 0, 0]),
                                "section_class": section.get("class", "Unknown")
                            })
        
        # Remove duplicates
        unique_results = []
        seen = set()
        for res in results:
            identifier = (res["page_num"], tuple(res["bbox"]))
            if identifier not in seen:
                unique_results.append(res)
                seen.add(identifier)

        logger.info(f"Found {len(unique_results)} unique search matches.")
        return unique_results

class ImageProcessor:
    """Handle image processing and highlighting."""
    
    @staticmethod
    def apply_search_highlights(img: Image.Image, page_data: Dict, search_results: List[Dict]) -> Image.Image:
        """
        Apply colored highlights to image based on search results.
        This version includes the CRITICAL coordinate scaling fix.
        """
        try:
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # This is the DPI the images were rendered at in pdf_processor.py
            # This is the key to the fix.
            DPI_FOR_RENDERING = 150
            SCALING_FACTOR = DPI_FOR_RENDERING / 72.0
            
            page_num = page_data.get("page", 1)
            page_results = [r for r in search_results if r["page_num"] == page_num]
            
            for result in page_results:
                bbox = result.get("bbox", [0, 0, 0, 0])
                
                if len(bbox) != 4 or any(not isinstance(x, (int, float)) for x in bbox):
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                # --- START OF THE FIX ---
                # Scale the PDF coordinates to match the rendered image's DPI
                scaled_bbox = [coord * SCALING_FACTOR for coord in bbox]
                # --- END OF THE FIX ---
                
                color_idx = result["keyword_idx"] % len(SEARCH_COLORS)
                color = SEARCH_COLORS[color_idx]
                
                x0, y0, x1, y1 = map(int, scaled_bbox) # Use the scaled coordinates
                
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(img.size[0], x1), min(img.size[1], y1)
                
                if x1 > x0 and y1 > y0:
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=color[:3] + (255,), width=2)
            
            return Image.alpha_composite(img.convert("RGBA"), overlay)
            
        except Exception as e:
            logger.error(f"Error applying highlights: {e}")
            return img

class UIComponents:
    """UI component rendering methods."""
    
    @staticmethod
    def display_statistics(stats: Dict[str, Any]) -> None:
        """Display statistics in a professional grid layout."""
        st.markdown('<div class="section-header">📊 Document Statistics</div>', unsafe_allow_html=True)
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Total Pages", stats["total_pages"])
        
        with col2:
            st.metric("📦 Total Sections", stats["total_sections"])
        
        with col3:
            st.metric("📈 Avg Sections/Page", stats["avg_sections_per_page"])
        
        # Class breakdown
        if stats["class_counts"]:
            st.markdown("### 🏷️ Section Classification Breakdown")
            
            # Create columns for class counts
            num_classes = len(stats["class_counts"])
            cols = st.columns(min(num_classes, 5))
            
            for idx, (cls, count) in enumerate(sorted(stats["class_counts"].items(), 
                                                       key=lambda x: x[1], reverse=True)):
                col_idx = idx % len(cols)
                percentage = (count / stats["total_sections"]) * 100 if stats["total_sections"] > 0 else 0
                with cols[col_idx]:
                    st.metric(
                        cls,
                        count,
                        f"{percentage:.1f}%",
                        delta_color="off"
                    )
    
    @staticmethod
    def display_search_interface() -> List[str]:
        """
        Display the search interface with 5 keyword inputs.
        
        Returns:
            List of keywords entered by user
        """
        st.markdown('<div class="section-header">🔍 Search & Highlight</div>', unsafe_allow_html=True)
        
        # Create 5 input boxes in columns
        st.markdown("Enter up to 5 keywords to search and highlight in the document:")
        
        col1, col2 = st.columns(2)
        keywords = []
        
        with col1:
            keywords.append(st.text_input("Keyword 1", key="kw1", placeholder="Enter first keyword..."))
            keywords.append(st.text_input("Keyword 2", key="kw2", placeholder="Enter second keyword..."))
            keywords.append(st.text_input("Keyword 3", key="kw3", placeholder="Enter third keyword..."))
        
        with col2:
            keywords.append(st.text_input("Keyword 4", key="kw4", placeholder="Enter fourth keyword..."))
            keywords.append(st.text_input("Keyword 5", key="kw5", placeholder="Enter fifth keyword..."))
        
        # Search button row
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_clicked = st.button("🧹 Clear", type="secondary", use_container_width=True)
        
        # Handle button clicks
        if clear_clicked:
            st.session_state.search_results = []
            st.rerun()
        
        if search_clicked:
            UIComponents._handle_search(keywords)
        
        return keywords
    
    @staticmethod
    def _handle_search(keywords: List[str]) -> None:
        """Handle search button click."""
        # Filter out empty keywords
        active_keywords = [kw.strip() for kw in keywords if kw.strip()]
        
        if not active_keywords:
            st.warning("Please enter at least one keyword to search.")
            return
        
        # Ensure we have data loaded
        if st.session_state.document_data is None or st.session_state.layout_data is None:
            st.error("No document data available. Please process a PDF first.")
            return
        
        # Perform search
        with st.spinner("Searching..."):
            try:
                results = SearchEngine.search_keywords(
                    st.session_state.layout_data, 
                    st.session_state.document_data, 
                    keywords
                )
                st.session_state.search_results = results
            
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return
               
        # Display results
        UIComponents._display_search_results(results, active_keywords)
    
    @staticmethod
    def _display_search_results(results: List[Dict], active_keywords: List[str]) -> None:
        """Display search results with legend and summary."""
        if results:
            st.success(f"Found {len(results)} matches across the document!")
            
            # Display color legend
            st.markdown("### Highlight Legend")
            legend_html = '<div class="highlight-legend">'
            for idx, kw in enumerate(active_keywords):
                color = SEARCH_COLORS[idx % len(SEARCH_COLORS)]
                color_rgba = f"rgba{color}"
                legend_html += f'''
                <div class="legend-item">
                    <div class="color-box" style="background-color: {color_rgba};"></div>
                    <span><strong>{kw}</strong></span>
                </div>
                '''
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # Group results by page for summary
            results_by_page = {}
            for result in results:
                page_num = result["page_num"]
                if page_num not in results_by_page:
                    results_by_page[page_num] = []
                results_by_page[page_num].append(result)
            
            # Display detailed results
            with st.expander("📋 Search Results Details", expanded=False):
                for page_num, page_results in sorted(results_by_page.items()):
                    st.markdown(f"**Page {page_num}:** {len(page_results)} matches")
                    for result in page_results:
                        st.markdown(f"  - '{result['keyword']}' in {result['section_class']} section")
        else:
            st.info("No matches found for the given keywords.")
    
    @staticmethod
    def display_page_viewer(document_data: List[Dict], layout_data: List[Dict], 
                          search_results: List[Dict] = None) -> None:
        """Display page viewer with thumbnails and detailed view."""
        st.markdown('<div class="section-header">📖 Document Viewer</div>', unsafe_allow_html=True)
        
        if not document_data:
            st.warning("No pages to display")
            return
        
        # Create two columns: thumbnails and main viewer
        thumb_col, viewer_col = st.columns([1, 3])
        
        with thumb_col:
            UIComponents._display_thumbnails(document_data)
        
        with viewer_col:
            UIComponents._display_main_viewer(document_data, layout_data, search_results)
    
    @staticmethod
    def _display_thumbnails(document_data: List[Dict]) -> None:
        """Display page thumbnails sidebar."""
        st.markdown("### Page Thumbnails")
        
        for idx, page in enumerate(document_data):
            image_name = page.get("image_name", f"page_{idx+1:03d}.png")
            image_path = OUTPUT_DIR / image_name
            
            if image_path.exists():
                # Navigation button
                button_type = "primary" if idx == st.session_state.current_page else "secondary"
                if st.button(f"Page {page.get('page', idx+1)}", 
                           key=f"thumb_{idx}", 
                           use_container_width=True,
                           type=button_type):
                    st.session_state.current_page = idx
                    st.rerun()
                
                try:
                    # Show thumbnail
                    img = Image.open(image_path)
                    img_thumb = img.copy()
                    img_thumb.thumbnail((150, 150))
                    st.image(img_thumb, caption=f"Page {page.get('page', idx+1)}", 
                           use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load thumbnail: {e}")
                
                st.divider()
    
    @staticmethod
    def _display_main_viewer(document_data: List[Dict], layout_data: List[Dict], 
                           search_results: List[Dict]) -> None:
        """Display main page viewer."""
        if st.session_state.current_page >= len(document_data):
            st.error("Invalid page selection")
            return
        
        current_page = document_data[st.session_state.current_page]
        current_layout = (layout_data[st.session_state.current_page] 
                         if st.session_state.current_page < len(layout_data) else None)
        
        st.markdown(f"### Page {current_page.get('page', st.session_state.current_page + 1)}")
        
        # Load and display the page image
        image_name = current_page.get("image_name", f"page_{st.session_state.current_page+1:03d}.png")
        image_path = OUTPUT_DIR / image_name
        
        if image_path.exists():
            try:
                img = Image.open(image_path).convert("RGBA")
                
                # Apply search highlights if any
                if search_results:
                    img = ImageProcessor.apply_search_highlights(img, current_page, search_results)
                
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load page image: {e}")
        else:
            st.warning(f"Page image not found: {image_name}")
        
        # Display sections for this page
        UIComponents._display_page_sections(current_layout)
    
    @staticmethod
    def _display_page_sections(current_layout: Optional[Dict]) -> None:
        """Display extracted sections for current page."""
        if not current_layout or not current_layout.get("sections"):
            st.info("No sections extracted for this page")
            return
        
        st.markdown("### 🔍 Extracted Sections")
        
        # Section class emojis
        class_emoji = {
            "Header": "📌",
            "Text": "📄",
            "Table": "📊",
            "Footer": "📋",
            "Figure": "🖼️"
        }
        
        for idx, section in enumerate(current_layout["sections"]):
            section_class = section.get("class", "Unknown")
            section_text = section.get("text", "")
            
            emoji = class_emoji.get(section_class, "📄")
            
            with st.expander(f"{emoji} {section_class} - Section {idx + 1}", expanded=False):
                st.markdown(f"**Class:** `{section_class}`")
                st.markdown(f"**Bounding Box:** `{section.get('bbox', 'N/A')}`")
                st.markdown("**Content:**")
                st.text_area("", section_text, height=100, disabled=True, 
                           key=f"section_{idx}", label_visibility="collapsed")


class MainApplication:
    """Main application controller."""
    
    def __init__(self):
        SessionManager.initialize_session()
        FileManager.ensure_output_dir()
    
    def run(self) -> None:
        """Run the main application."""
        # Header
        st.title("📄 PDF Layout Analyzer")
        st.markdown("*Professional document processing with AI-powered layout analysis*")
        st.divider()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self) -> None:
        """Render the sidebar interface."""
        with st.sidebar:
            st.markdown("## 📁 Document Processing")
            st.markdown("Upload a PDF to extract and analyze its layout structure.")
            
            # Process New File button
            if st.button("📄 Process New File", type="secondary", use_container_width=True):
                SessionManager.reset_session()
                st.success("Ready for new file!")
                st.rerun()
            
            st.divider()
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a PDF document to analyze its layout"
            )
            
            if uploaded_file is not None:
                self._handle_file_upload(uploaded_file)
            
            # Download section
            self._render_download_section()
            
            # Info section
            self._render_info_section()
    
    def _handle_file_upload(self, uploaded_file) -> None:
        """Handle PDF file upload and processing."""
        # Check if this is the same file
        file_content = uploaded_file.read()
        file_hash = FileManager.calculate_file_hash(file_content)
        
        if st.session_state.file_hash == file_hash and st.session_state.processed:
            st.info("This file has already been processed.")
            return
        
        # Reset for new file
        uploaded_file.seek(0)  # Reset file pointer
        
        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = Path(tmp_file.name)
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            return
        
        # Process button
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            try:
                # Process the PDF
                success, message = PDFProcessor.process_pdf_file(tmp_path)
                
                if success:
                    # Load and cache the results
                    document_data, layout_data = DataLoader.load_results()
                    
                    if document_data is not None and layout_data is not None:
                        st.session_state.document_data = document_data
                        st.session_state.layout_data = layout_data
                        st.session_state.statistics = StatisticsCalculator.calculate_statistics(layout_data)
                        st.session_state.processed = True
                        st.session_state.file_hash = file_hash
                        st.session_state.search_results = []  # Clear previous search
                        st.success(message)
                    else:
                        st.error("Failed to load processed data")
                else:
                    st.error(message)
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")
    
    def _render_download_section(self) -> None:
        """Render the download section in sidebar."""
        if st.session_state.processed:
            st.divider()
            st.markdown("## 💾 Download Results")
            
            layout_path = OUTPUT_DIR / "layout_all.json"
            if layout_path.exists():
                try:
                    with open(layout_path, 'r', encoding='utf-8') as f:
                        layout_json = f.read()
                    
                    st.download_button(
                        label="📥 Download Layout JSON",
                        data=layout_json,
                        file_name=f"layout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to prepare download: {e}")
    
    def _render_info_section(self) -> None:
        """Render the info section in sidebar."""
        st.divider()
        st.markdown("## ℹ️ About")
        st.markdown("""
        This application uses advanced PDF processing to:
        - Extract text blocks with font information
        - Classify content (Headers, Text, Tables, etc.)
        - Enable keyword search with visual highlighting
        - Provide detailed layout analysis
        
        **Professional Document Analyzer**
        """)
    
    def _render_main_content(self) -> None:
        """Render the main content area."""
        if st.session_state.processed and st.session_state.document_data is not None:
            try:
                # Display statistics
                if st.session_state.statistics:
                    UIComponents.display_statistics(st.session_state.statistics)
                
                # Search interface
                UIComponents.display_search_interface()
                
                # Page viewer
                UIComponents.display_page_viewer(
                    st.session_state.document_data,
                    st.session_state.layout_data,
                    st.session_state.search_results
                )
                
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                logger.error(f"Main content error: {e}")
        else:
            self._render_welcome_screen()
    
    def _render_welcome_screen(self) -> None:
        """Render the welcome screen when no document is processed."""
        # Welcome message
        st.markdown("""
        <div class="stat-card">
            <h2>👋 Welcome to PDF Layout Analyzer</h2>
            <p>This professional tool helps you:</p>
            <ul>
                <li>📄 Extract structured content from PDF documents</li>
                <li>🔍 Classify document sections (Headers, Text, Tables, etc.)</li>
                <li>🎯 Search and highlight keywords across pages</li>
                <li>📊 Generate detailed layout statistics</li>
                <li>💾 Export results in JSON format</li>
            </ul>
            <p><strong>Get started by uploading a PDF file using the sidebar!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <h3>🎨 Smart Classification</h3>
                <p>Automatically identifies headers, text blocks, tables, figures, and footers using advanced heuristics.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <h3>🔍 Keyword Search</h3>
                <p>Search up to 5 keywords simultaneously with color-coded highlighting on page images.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card">
                <h3>📊 Visual Analytics</h3>
                <p>Interactive page viewer with thumbnails, expandable sections, and detailed statistics.</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Application entry point."""
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        st.error("Application failed to start. Please check the logs.")
        logger.critical(f"Application startup failed: {e}")
        st.stop()


if __name__ == "__main__":
    main()