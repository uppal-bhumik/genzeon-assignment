# Genzeon AI/ML Internship Assignment: PDF Layout Analyzer

https://genzeon-assignment.streamlit.app

A robust, full-stack web application built for the Genzeon AI/ML Internship assessment. This tool analyzes the layout and content of PDF documents, classifies structural elements, and provides an interactive interface for keyword search and highlighting.

## 🚀 Live Demo

You can access the live, deployed application here:
**[https://genzeon-assignment.streamlit.app](https://genzeon-assignment.streamlit.app)**

## ✨ Key Features

* **Smart Document Processing:** Ingests PDF files and runs a multi-stage backend pipeline to extract detailed structural data.
* **Rule-Based Classification:** Uses a sophisticated heuristic engine to classify document sections into categories like `Header`, `Text`, `Table`, and `Footer`.
* **Interactive Document Viewer:** A dual-column interface with page thumbnails for easy navigation and a main viewer that displays page images and expandable sections of extracted content.
* **Multi-Keyword Search & Highlight:** Search for up to 5 keywords simultaneously with unique, color-coded highlights drawn directly onto the document.
* **Data Export:** Download the final, structured layout analysis as a `layout_all.json` file.

---

## 🧠 Core Methodology & Design Choices

### The Pivot to a Heuristic Engine

The initial approach considered using a pre-trained Deep Learning model (via `LayoutParser` with `Detectron2`/`PaddleDetection`). However, this route presented significant platform-specific installation challenges and dependency conflicts. Given the time constraints of the assignment, a strategic pivot was made to a more robust, reliable, and platform-agnostic **rule-based (heuristic) engine**. This approach not only guarantees cross-platform compatibility but also provides a faster, more lightweight solution with greater transparency and control over the classification logic.

### How the Rule-Based Engine Works

The engine's accuracy is built upon the rich metadata extracted by **PyMuPDF's** `get_text("dict")` method. This provides not just text, but also **font size**, **font name** (including styles like 'Bold'), and precise **coordinates** for every text block. Classification is then performed by applying a cascade of rules that analyze these properties:
* **Headers** are primarily identified by large font sizes and/or bold styling.
* **Footers** are identified by their position in the bottom 10% of the page and common keywords (e.g., 'page', 'confidential').
* **Tables** are identified by a combination of structural clues (e.g., wide page coverage), numeric density, and relevant keywords (e.g., '%', 'total', 'amount').
* **Text** serves as the default classification for standard content blocks.

### Application Architecture

The application is built on a multi-layered architecture to promote **separation of concerns**:
1.  **Data Extraction Layer (`pdf_processor.py`):** Responsible solely for interfacing with the raw PDF, extracting the complete block structure, and rendering page images. This isolates the core `PyMuPDF` dependency.
2.  **Logic/Analysis Layer (`layout_analyzer.py`):** Contains the core heuristic engine. It takes the structured data from the processor and applies the rules to produce the final, classified layout.
3.  **Presentation/UI Layer (`app.py`):** The Streamlit frontend, responsible for user interaction, orchestrating the backend calls, and visualizing the results in an intuitive dashboard.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **PDF Processing:** PyMuPDF
* **Text/Layout Analysis:** Custom Rule-Based Heuristic Engine
* **Core Libraries:** Pillow

---

## ⚙️ Local Setup and Usage

To run this application on your local machine, please follow these steps.

### Prerequisites

* Python 3.9+
* **Tesseract OCR Engine**: This is a dependency for one of the backend modules.
    * [Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    * On macOS: `brew install tesseract`
    * On Linux: `sudo apt-get install tesseract-ocr`

### Installation & Running the App

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/uppal-bhumik/genzeon-assignment.git](https://github.com/uppal-bhumik/genzeon-assignment.git)
    cd genzeon-assignment
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On Mac/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will now be running and accessible in your web browser.
