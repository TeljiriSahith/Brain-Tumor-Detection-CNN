import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import CNN

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to right,
        #0f2027,
        #203a43,
        #2c5364
    );
}

h1,h2,h3,h4,h5,h6,p {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.25);
}

[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():

    model = CNN()

    model.load_state_dict(
        torch.load(
            "brain_tumor_cnn.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()

classes = ["No Tumor", "Tumor"]

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🧠 Model Information")

st.sidebar.info("""
**Model:** CNN

**Framework:** PyTorch

**Accuracy:** 86.27%

**Classes:** 2

**Dataset:** Brain MRI Dataset
""")

# ---------------------------------
# Hero Section
# ---------------------------------
st.markdown("""
<h1 style='text-align:center;'>
🧠 Brain Tumor Detection using Deep Learning
</h1>

<h4 style='text-align:center;'>
Upload MRI scans and get instant AI-powered predictions
</h4>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------
# Metrics Section
# ---------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎯 Accuracy\n\n86.27%")

with col2:
    st.info("📊 Classes\n\n2")

with col3:
    st.info("⚡ Framework\n\nPyTorch")

st.divider()

# ---------------------------------
# Upload Section
# ---------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction Panel")

        if st.button("🔍 Analyze MRI Scan"):

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])

            image_tensor = transform(image)

            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence = (
                    torch.max(probabilities).item()
                    * 100
                )

                _, predicted = torch.max(
                    output,
                    1
                )

            prediction = classes[
                predicted.item()
            ]

            if prediction == "Tumor":

                st.error(
                    "🔴 Tumor Detected"
                )

            else:

                st.success(
                    "🟢 No Tumor Detected"
                )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.progress(
                confidence / 100
            )

            st.info(
                f"Predicted Class: {prediction}"
            )

st.divider()

# ---------------------------------
# Disclaimer
# ---------------------------------
st.warning(
    "⚠️ This application is intended for educational purposes only and should not replace professional medical diagnosis."
)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""
<div style='text-align:center;color:white;'>

Developed using PyTorch & Streamlit 🚀

</div>
""", unsafe_allow_html=True)