import numpy 
import gradio as gr
from functions import process_receipt_gradio,get_text_easyocr,get_segments

scanner = gr.Interface(
    fn = process_receipt_gradio,
    inputs = [
        gr.Image(type='numpy',label='Upload Receipt Image'),
        gr.Slider(label='Confidence',value=0.5,minimum=0.1,maximum=0.9,step=0.1)],
    outputs = [
       gr.Image(type='numpy',label='Detected Segments'),
       gr.JSON(label='Extracted Text'),
       gr.JSON(label='Segment Count')
    ],
    title = "Receipt OCR and Object Detection",
    decription = "Upload a receipt image to detect key fields and extract text using YOLO and EasyOCR."
)
scanner.launch(debug=False,
               share=True)
