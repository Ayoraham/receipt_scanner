def get_segments(detections,img):
  """Function to the cropped section of each segment detected
      detections: Output of the yolo model (the detcted segments along with other metrics)
      img: The receipt image which is fed into the yolo model
  """
  segments = {}
  classes = detections[0].names
  segments = [int(segment) for segment in detections[0].boxes.cls.numpy().tolist()]
  segment_count = {classes[segment]:segments.count(segment) for segment in segments}
  bboxes = detections[0].boxes.xyxy.numpy()
  cls = detections[0].boxes.cls.numpy()

  for i in range(len(bboxes)):
    x1,y1,x2,y2 = bboxes[i].tolist()
    name = cls[i]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #Padding
    pad=5
    x1 = max(0, x1 - pad )
    y1 = max(0, y1 - pad) # Corrected: use y1 - pad
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad) # Corrected: use img.shape[0] for height

    segments[f"{classes[name]}"] = img[y1:y2,x1:x2]

  return segments,segment_count

def get_text_easyocr(segments):
  """Function to extract text from the detected segments"""

  import easyocr
  reader = easyocr.Reader(['en'])

  segment_text = {}
  for key, img_segment in segments.items():
    # Ensure the image segment is in a format EasyOCR can process (e.g., RGB if it's grayscale from processing)
    # If the segment is already BGR, readtext can handle it.
    # If it's a single channel (grayscale), easyocr.Reader(['en']) will handle it.
    results = reader.readtext(image=img_segment)
    # Extract the text from each result
    extracted_texts = [res[1] for res in results]
    segment_text[key] = extracted_texts
  return segment_text


def process_receipt_gradio(img_array:numpy.ndarray,confidence:int=0.5):

  from ultralytics import YOLO
  import numpy
  import cv2
  model = YOLO("best.pt")

  # Perform YOLO prediction
  results = model.predict(img_array, conf=confidence)
  predicted_image_plot = results[0].plot() # This returns a numpy array suitable for gradio.Image output

  # Get segments (cropped images)
  segments_dict,segment_count = get_segments(results, img_array)

  # Get text from segments using EasyOCR
  extracted_text = get_text_easyocr(segments_dict)


  return predicted_image_plot, extracted_text, segment_count
