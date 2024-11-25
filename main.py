import streamlit as st
import PIL.Image
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import tempfile
import os

# Initialize PaddleOCR for Chinese
ocr_chinese = PaddleOCR(use_angle_cls=True, lang='ch')  # For Chinese text detection

# Classes for detected objects
classes = [
    "Skirts",
    "Tops",
    "Blazers",
    "Coat",
    "Dress",
    "Jacket",
    "Knitwear",
    "Pant"
]

# Load the YOLO model
model = YOLO('models/best.pt')

# Streamlit UI
st.set_page_config(page_title="Object Detection and Text Extraction", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Single Image Processing", "Multiple Image Processing"]
choice = st.sidebar.radio("Select Mode:", options)

# Function to process an image
def process_image(image_path):
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    labels = results[0].boxes.cls.cpu().numpy().astype(int)  # Class labels

    results_data = []

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)

        # Crop the detected object
        cropped_image = image[y1:y2, x1:x2]
        cropped_pil_image = PIL.Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Use OCR
        result_chinese = ocr_chinese.ocr(np.array(cropped_pil_image), cls=True)

        # Extract fields
        brand_name = None
        product_name = None
        price = None

        if len(result_chinese[0]) > 0:
            brand_name = result_chinese[0][0][1][0]

        if len(result_chinese[0]) > 1:
            second_text = result_chinese[0][1][1][0]
            if "参考价" in second_text:
                price_match = re.search(r'参考价.*?([\d,]+)', second_text)
                if price_match:
                    price = price_match.group(1).replace(',', '')
            else:
                product_name = second_text

        if len(result_chinese[0]) > 2:
            third_text = result_chinese[0][2][1][0]
            if "参考价" in third_text:
                price_match = re.search(r'参考价.*?([\d,]+)', third_text)
                if price_match:
                    price = price_match.group(1).replace(',', '')

        product_type = classes[labels[i]] if labels[i] < len(classes) else None

        results_data.append({
            "image_name": os.path.basename(image_path),
            "type": product_type,
            "brand_name": brand_name,
            "product_name": product_name,
            "price(¥ or €)": price if price else None
        })

    return results_data


if choice == "Single Image Processing":
    st.title("Single Image Processing")

    # Display images from the "Data" folder as test samples
    data_folder = "Data"
    if os.path.exists(data_folder):
        image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_files:
            st.write("### Test Images from Data Folder:")
            selected_image = st.selectbox("Select a test image", image_files)

            # Display the selected image
            image_path = os.path.join(data_folder, selected_image)
            st.image(image_path, caption="Selected Image", use_column_width=True)

            uploaded_file = None
        else:
            st.warning("No images found in the 'Data' folder.")
    else:
        st.error(f"The folder '{data_folder}' does not exist.")

    uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        image_path = temp_file.name

        # Display the image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

    if uploaded_file or selected_image:
        st.markdown("### Processing Image...")
        results_data = process_image(image_path)

        # Display results
        df = pd.DataFrame(results_data)
        st.write("### Detection Results")
        st.dataframe(df)

        # Save to CSV
        csv_file = "single_image_results.csv"
        df.to_csv(csv_file, index=False)

        # Download button
        st.download_button(
            label="Download CSV",
            data=open(csv_file, "rb"),
            file_name="single_image_results.csv",
            mime="text/csv"
        )

        os.remove(temp_file.name) if uploaded_file else None

elif choice == "Multiple Image Processing":
    st.title("Multiple Image Processing")

    # Users can upload multiple images
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        all_results = []

        # Process each uploaded image
        for uploaded_file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            temp_file.close()
            image_path = temp_file.name

            # Process the image
            st.markdown(f"### Processing {uploaded_file.name}...")
            results_data = process_image(image_path)
            all_results.extend(results_data)

            os.remove(temp_file.name)

        # Combine results into a DataFrame
        df = pd.DataFrame(all_results)

        # Display sample results
        st.write("### Sample Detection Results")
        st.dataframe(df.head())

        # Save to CSV
        csv_file = "multiple_images_results.csv"
        df.to_csv(csv_file, index=False)

        # Download button
        st.download_button(
            label="Download Full CSV",
            data=open(csv_file, "rb"),
            file_name="multiple_images_results.csv",
            mime="text/csv"
        )
