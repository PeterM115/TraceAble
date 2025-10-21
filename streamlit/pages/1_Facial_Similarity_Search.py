# Libraries required for frontend
from qdrant_client import QdrantClient
from io import BytesIO
import streamlit as st
import base64
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
# For drawing boxes on the image around detected faces.
from PIL import ImageDraw
import streamlit.components.v1 as components # Used to tell Streamlit that the html code is raw HTML not markdown. (components.html())
import open_clip # Model for clip embeddings

COLOURS = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow"] # Global to access colour list in draw_boxes_on_image function.

# Define a global to access the qdrant collection we have uploaded to the cluster.
collection_name = "facenet_faces"

# Initialises session state record.
if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None


def set_selected_record(new_record):
    st.session_state.selected_record = new_record


@st.cache_resource # Cache the client so it doesn't have to be remade every time streamlit refreshes.
def get_client():
    return QdrantClient(
        url=st.secrets["qdrant_db_url"],
        api_key=st.secrets["qdrant_api_key"]
    )


@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=True)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, facenet

# Seperate function as there are many returned values, less messy than grouping with load_models imo.
@st.cache_resource
def load_clip_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model.eval(), preprocess, tokenizer


def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))


def tensor_to_image(face_tensor):
    unnorm = face_tensor.detach().cpu() # detach tensor from the computation graph
    img = ((unnorm + 1) / 2).permute(1, 2, 0).clamp(0, 1).numpy()  # rearrange tensor dimensions [-1,1] to [0,1]
    img = (img * 255).astype("uint8") # scaling the pixel values from [0,1] to [0,255]. Converts to 8bit int for display.
    return Image.fromarray(img) # converts array into a PIL image


def draw_boxes_on_image(image, boxes):
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    if boxes is not None:
        for i, box in enumerate(boxes):
            color = COLOURS[i % len(COLOURS)]
            draw.rectangle(box.tolist(), outline=color, width=5)
    return image_with_boxes


# Added for uploading user submitted image to db, duplicated from my embeddings-code.ipynb
def base64_encode_image(pil_image):
    image_data = BytesIO()
    pil_image.save(image_data, format='JPEG') #.save function saves to memory not to disk. Just clarifying here as I initially misunderstood it in the documentation. 
    base64_string = base64.b64encode(image_data.getvalue()).decode('utf-8')
    return base64_string


def base64_encode_tensor(tensor_image):
    # Tensor is in [-1, 1], we need to bring it to [0, 255] for rgb output. 
    unnormalized = ((tensor_image + 1) / 2 * 255).clamp(0, 255).byte()
    pil_image = Image.fromarray(unnormalized.permute(1, 2, 0).cpu().numpy())
    return base64_encode_image(pil_image)



@st.dialog("Source Image")
def show_source_image_popup():
    record = st.session_state.view_source_record
    if record:
        source_img = get_bytes_from_base64(record.payload["source_base64"])
        st.image(source_img, caption="Source Image", use_container_width=True)
        st.write(f"Source Path: {record.payload['source_path']}")


st.title("Facial Similarity Search")
uploaded_file = st.file_uploader("Upload an image to search for similar faces", type=["jpg", "jpeg", "png"])

# 
# db_upload_results = []

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # Previous display uploaded image code: 
    # st.image(image, caption="Uploaded Image", use_container_width=True) 

    mtcnn, facenet = load_models()
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    # new boxes around detected faces on uploaded image code
    boxes, _ = mtcnn.detect(image)  # Get bounding boxes
    image_with_boxes = draw_boxes_on_image(image, boxes)  # Draw them
    st.image(image_with_boxes, caption="Detected Faces", use_container_width=True)
    # Create key for which colours are mapped to which face boxes
    if boxes is not None:
        st.markdown("### Face Legend")

        legend_items = "" # to store each legend item. 
        for i in range(len(boxes)):
            colour = COLOURS[i % len(COLOURS)]
            # adds each colour key to the list 1 by 1
            legend_items += f"""
                <div style='display: flex; align-items: center; width: calc(25% - 12px); margin-bottom: 8px; font-family:  "Source Sans Pro", sans-serif;'>
                    <div style='width: 20px; height: 20px; background-color: {colour}; border-radius: 4px; margin-right: 8px;'></div>
                    <span style='font-size: 14px;'>Face {i + 1}</span>
                </div>
            """
        # Adds all legend_items to a html div for output on frontend. 
        legend_html = f"""
            <div style='display: flex; flex-wrap: wrap; gap: 12px;'>
                {legend_items}
            </div>
        """
        # allows the html to act like normal HTML code. 
        components.html(legend_html, height=120)


    faces = mtcnn(image)  # returns list of face tensors if keep_all=True

    if faces is not None:
    # Convert tensor batch [N, 3, 160, 160] to list of face tensors
        if isinstance(faces, torch.Tensor):
            faces = [face for face in faces]

        st.write("Select a face to search with or mark for upload:")
        face_cols = st.columns(len(faces))

        # Keep track of which faces are selected for upload
        selected_faces_to_upload = []

        for idx, face in enumerate(faces):
            with face_cols[idx]:
                face_img = tensor_to_image(face)
                st.image(face_img, caption=f"Face {idx + 1}")
                
                # Search button
                if st.button(f"Search with Face {idx + 1}", key=f"search_face_{idx}"):
                    st.session_state.selected_face_idx = idx
                    st.rerun()
                
                # Upload checkbox
                checkbox_key = f"upload_face_{idx}"
                if st.checkbox("Upload?", key=checkbox_key):
                    selected_faces_to_upload.append((idx, face))

        # List of faces marked for upload
        if selected_faces_to_upload:
            st.markdown(f"**Faces selected for upload:** {', '.join(str(i+1) for i, _ in selected_faces_to_upload)}")


        # Create a button to trigger the upload
        if st.button("Upload Selected Faces to Database") and selected_faces_to_upload:
            results = []
            client = get_client()

            for idx, face in selected_faces_to_upload:
                if face is None:
                    continue

                with torch.no_grad():
                    embedding = facenet(face.unsqueeze(0)).squeeze().numpy().tolist()

                clip_image = tensor_to_image(face)
                clip_input = clip_preprocess(clip_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_embedding = clip_model.encode_image(clip_input)
                    clip_embedding /= clip_embedding.norm(dim=-1, keepdim=True)
                clip_embedding = clip_embedding.squeeze().cpu().tolist()

                # Duplicate check using Qdrant similarity search
                similar = client.search(
                    collection_name=collection_name,
                    query_vector=("facenet", embedding),
                    limit=1,
                    score_threshold=0.995  # Or tweak if needed
                )

                if similar:
                    st.warning(f"Face {idx + 1} Appears to be identical to an existing entry and will be skipped from uploading to the database (>99.5% similarity rate).")
                    continue

                base64_face = base64_encode_tensor(face)
                base64_original = base64_encode_image(image)

                payload = {
                    "source_path": f"uploaded_by_user_{uploaded_file.name}",
                    "source_base64": base64_original,
                    "face_index": idx,
                    "face_base64": base64_face,
                    "embedding": embedding,
                    "clip_embedding": clip_embedding
                }

                results.append(payload)

            # Send to Qdrant
            from qdrant_client.models import PointStruct
            import uuid

            if not results:
                st.info("No new faces to upload. All selected faces already exist in the database.")
            else:
                # Proceed with upload only if there's something to upload
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "facenet": record["embedding"], 
                            "clip": record["clip_embedding"] # changed to accomodate new clip embedding
                        },
                        payload={
                            "source_path": record["source_path"],
                            "source_base64": record["source_base64"],
                            "face_index": record["face_index"],
                            "face_base64": record["face_base64"]
                        }
                    )
                    for record in results
                ]

                client.upsert(collection_name=collection_name, points=points)
                st.success(f"{len(results)} face(s) uploaded to the database.")
                


        # Show search results if a face was selected
        if "selected_face_idx" in st.session_state:
            selected_idx = st.session_state.selected_face_idx
            selected_face = faces[selected_idx]

            with torch.no_grad():
                embedding = facenet(selected_face.unsqueeze(0)).squeeze().numpy()

            client = get_client()
            results = client.search(
                collection_name=collection_name,
                query_vector=("facenet", embedding),
                limit=12
            )

            st.header("Similar Faces Found:")
            cols = st.columns(4)
            for i, record in enumerate(results):
                col = cols[i % 4]
                with col:
                    st.image(get_bytes_from_base64(record.payload["face_base64"]))
                    st.caption(f"Similarity: {record.score:.2%}")

                    if st.button("View Source", key=f"view_{record.id}"):
                        st.session_state.view_source_record = record
                        show_source_image_popup()
        
                

                
    else:
        st.warning("No face detected in the uploaded image. Please try another.")
