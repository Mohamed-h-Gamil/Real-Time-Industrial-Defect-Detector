import urllib.request
import os
import subprocess
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from anomalib.engine import Engine
from anomalib.models import Padim

url = "http://192.168.1.3:8080/shot.jpg"

model = Padim.load_from_checkpoint(
    checkpoint_path="results/myPadim/bottle_cap_dataset/latest/weights/lightning/model.ckpt"
)

# Initialize engine and call predict
engine = Engine()
#cap = cv2.VideoCapture(0)

st.title("Anomaly Detection App")

frame_placeholder = st.empty()

process_button = st.button("Process")
stop_button = st.button("Stop")

frame_placeholder_input = st.empty()
frame_placeholder_output = st.empty()


data_transforms = transforms.Compose([
    transforms.CenterCrop(size=(900, 900)),       # Center crop the image to 256x256
    transforms.Resize(size=(256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),                          # Convert PIL image to PyTorch tensor
])

count = 1

while not stop_button:

    img_arr = np.array(bytearray(urllib.request.urlopen(url).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img, channels="RGB")

    if process_button:
        process_button = False
        img = Image.fromarray(img)
        img_trans = data_transforms(img)
        torchvision.utils.save_image(img_trans, os.path.join("process/input", f"{count}.png"))
        predictions = engine.predict(model=model, data_path=f"process/input/{count}.png")
        out_img = Image.open(f"results/Padim/latest/images/input/{count}.png")
        frame_placeholder_input.image(img, channels="RGB")
        frame_placeholder_output.image(out_img, channels="RGB")
        count += 1
