import os
import glob
import torch
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from infer import load_model, infer_bootstrap

def infer(models_path, image_file, st):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    models_paths = glob.glob(os.path.join(models_path, 'resnext50_fold_*.pth'))
    if not models_paths:
        st.error("No models found in the specified directory.")
        return

    models = [load_model(model_path, device) for model_path in models_paths]

    with st.spinner('Processing image...'):
        mean_score, error = infer_bootstrap(models, image_file, transform, device)
    
    if mean_score is not None:
        st.success(f"The predicted attractiveness score is {mean_score:.3f} ± {error:.3f}")
    else:
        st.error("An error occurred during the prediction.")

if __name__ == '__main__':
    st.title("Attractiveness Prediction")
    st.write("Upload an image to predict its attractiveness score (1-5).")

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    models_path = './models'

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #image = image.resize((250, 250))
        st.image(image, caption='Uploaded Image', use_column_width=False)

        image.save("temp_uploaded_image.jpg")

        with st.spinner('Processing image...'):
            infer(models_path, "temp_uploaded_image.jpg", st)

        # Clean up the temporary file
        os.remove("temp_uploaded_image.jpg")