# 🧾 AI-Powered Receipt Scanner & Data Extractor

### 🎯 The Goal
I built this project to automate the tedious task of manual data entry. The goal was to create a system that can take a raw image of a paper receipt and convert it into **structured digital data (JSON)**.

### ⚙️ How It Works
The system runs on a two-step pipeline:

1.  **Segmentation (YOLOv8n):**
    First, the AI "looks" at the image to locate specific regions of interest—such as the *Total Price*, *Date*, and *Vendor Name*—while ignoring irrelevant background clutter.

2.  **Text Extraction (EasyOCR):**
    The system crops these detected regions and passes them to an Optical Character Recognition (OCR) engine to read the text and output the final values.

### ⚠️ Current Challenges & Next Steps
While the object detection is accurate, the text extraction phase is sensitive to image quality. **"Noisy" images**—those with shadows, poor lighting, or unique fonts—can sometimes lead to character errors (e.g., confusing an `S` with a `5`).

I am currently working on improving the **image pre-processing** to handle these real-world conditions better. I am very open to feedback or suggestions from the community on robust noise-reduction techniques!

The demo is available on [HuggingFace](https://huggingface.co/spaces/Ayoraham/receipt_scanner_demo)
