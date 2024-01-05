import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json

# Streamlit settings
st.set_page_config(
    page_title="Skin Condition Prediction",
    page_icon="🌞",
    layout="centered",
    initial_sidebar_state="auto",
)

# Streamlit sidebar
st.sidebar.title("Skin Condition Prediction")
upload_option = st.sidebar.radio("Choose Image Input Method", ("Upload Image", "Camera"))
st.sidebar.markdown("----")

# Load the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model(model_dir):
    model_path = os.path.realpath(model_dir)
    if not os.path.exists(model_path):
        raise ValueError(f"Exported model folder doesn't exist {model_dir}")
    
    # Load the signature file to get input and output names
    signature_path = os.path.join(model_path, "signature.json")
    with open(signature_path, "r") as f:
        signature = json.load(f)
        inputs = signature.get("inputs")
        outputs = signature.get("outputs")

    # Create a new TensorFlow session and load the model
    session = tf.compat.v1.Session(graph=tf.Graph())
    tf.compat.v1.saved_model.loader.load(sess=session, tags=signature.get("tags"), export_dir=model_path)
    
    return session, inputs, outputs

model_dir = "models"  # Update with the correct path to your model folder
session, inputs, outputs = load_model(model_dir)

if upload_option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            # Read and preprocess the image
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image = np.asarray(image) / 255.0

            # Create the feed dictionary for the model
            feed_dict = {inputs["Image"]["name"]: [image]}

            # List the outputs we want from the model
            fetches = [(key, output["name"]) for key, output in outputs.items()]

            # Run the model and get the results
            outputs = session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)
            results = {}
            for i, (key, _) in enumerate(fetches):
                val = outputs[i].tolist()[0]
                if isinstance(val, bytes):
                    val = val.decode()
                results[key] = val

            # Display prediction results
            st.write(f"Prediction: {results['Prediction']}")
            st.write(f"Confidence: {results['Confidences']:.4f}")

# Start the Streamlit app
if __name__ == "__main__":
    st.title("Skin Condition Prediction")
    st.write("Choose an image input method from the sidebar.")