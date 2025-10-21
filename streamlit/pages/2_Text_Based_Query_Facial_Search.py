import streamlit as st
import torch
from qdrant_client import QdrantClient
import base64
from PIL import Image
from io import BytesIO

# Load CLIP model and tokenizer (reused from other page)
@st.cache_resource
def load_clip_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model.eval(), preprocess, tokenizer

clip_model, _, clip_tokenizer = load_clip_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# Qdrant client
@st.cache_resource
def get_client():
    return QdrantClient(
        url=st.secrets["qdrant_db_url"],
        api_key=st.secrets["qdrant_api_key"]
    )

client = get_client()

# Interface
st.title("Text-Based Face Search")

query_text = st.text_input("Enter a facial identifier to search by (e.g. glasses, ginger, beard):")
search_button = st.button("Search")

# Turns text input into CLIP embeddings and queries the database.
if search_button and query_text:
    with st.spinner("Searching..."):
        with torch.no_grad():
            text_tokens = clip_tokenizer([query_text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_vector = text_embedding.squeeze().cpu().numpy().tolist()

        results = client.search(
            collection_name="facenet_faces",
            query_vector=("clip", text_vector),
            limit=12
        )

        if not results:
            st.warning("No similar faces found.")
        else:
            for result in results:
                payload = result.payload
                face_base64 = payload["face_base64"]
                image_data = base64.b64decode(face_base64)
                image = Image.open(BytesIO(image_data))
                st.image(image, caption=f"Match from: {payload['source_path']}", use_container_width=True)
