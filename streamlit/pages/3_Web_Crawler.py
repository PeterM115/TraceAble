# Imported libraries
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from PIL import Image, ImageDraw
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid


# Interface elements
st.title("Web Crawler Tool")

url = st.text_input("Enter URL to crawl:")
max_pages = st.slider("Max pages to crawl", 1, 30, 5)
crawl_button = st.button("Start Crawling")

if "image_data" not in st.session_state:
    st.session_state.image_data = []


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


mtcnn, facenet = load_models()

clip_model, clip_preprocess, clip_tokenizer = load_clip_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

    


@st.cache_resource # Cache the client so it doesn't have to be remade every time streamlit refreshes.
def get_client():
    return QdrantClient(
        url=st.secrets["qdrant_db_url"],
        api_key=st.secrets["qdrant_api_key"]
    )

def draw_boxes_on_image(image, boxes):
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline='lime', width=2)
    return image_with_boxes


# Added for uploading user submitted image to db, duplicated from my embeddings-code.ipynb
def base64_encode_image(pil_image):
    image_data = BytesIO()
    pil_image.save(image_data, format='JPEG') #.save function saves to memory not to disk. 
    base64_string = base64.b64encode(image_data.getvalue()).decode('utf-8')
    return base64_string


def base64_encode_tensor(tensor_image):
    # Tensor is in [-1, 1], we need to bring it to [0, 255] for rgb output. 
    unnormalized = ((tensor_image + 1) / 2 * 255).clamp(0, 255).byte()
    pil_image = Image.fromarray(unnormalized.permute(1, 2, 0).cpu().numpy())
    return base64_encode_image(pil_image)


def tensor_to_image(tensor_image):
    # Convert normalized tensor back to a PIL image
    unnormalized = ((tensor_image + 1) / 2 * 255).clamp(0, 255).byte()
    return Image.fromarray(unnormalized.permute(1, 2, 0).cpu().numpy())


# Returns if a link belongs to the same website.
def is_internal_link(base_url, link):
    return urlparse(link).netloc == urlparse(base_url).netloc or urlparse(link).netloc == ""

# st.cache_data caches results so that it doesn't recrawl the same URL unless user requests
@st.cache_data(show_spinner=False)
def crawl_images(start_url, max_pages=5): # just setting max_pages to 5 by default as a safety catch.
    visited = set() # Tracks pages already crawled
    to_visit = [start_url] # Queue of pages to visit
    image_data = [] # Now stores [img_url, boxes, source_base64, faces, embeddings]

    while to_visit and len(visited) < max_pages: # Loops until the max_page limit has been reached or if there are no pages left to crawl before max_page.
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        try:
            response = requests.get(current_url, timeout=5) # Retrieves the page's HTML.
            visited.add(current_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Finds all <img> tags with a src, converts relative paths to full urls.  
                for img_tag in soup.find_all("img", src=True):
                    img_url = urljoin(current_url, img_tag["src"])
                    if any(img_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
                        try:
                            img_response = requests.get(img_url, timeout=5)
                            img = Image.open(BytesIO(img_response.content)).convert("RGB")
                            boxes, _ = mtcnn.detect(img) # calls MTCNN to detect faces and make boxes.
                            if boxes is not None and len(boxes) > 0:
                                aligned_faces = mtcnn(img)  # cropped aligned face tensors
                                if aligned_faces is not None:
                                    with torch.no_grad():
                                        facenet_embeddings = facenet(aligned_faces).detach().cpu()
                                        clip_embeddings = [] # to store clip embeddings
                                        # to generate clip embeddings
                                        for face in aligned_faces:
                                            pil_face = tensor_to_image(face)
                                            clip_input = clip_preprocess(pil_face).unsqueeze(0).to(device)
                                            with torch.no_grad():
                                                clip_embedding = clip_model.encode_image(clip_input)
                                                clip_embedding /= clip_embedding.norm(dim=-1, keepdim=True)
                                            clip_embeddings.append(clip_embedding.squeeze().cpu())

                                    image_data.append({
                                        "img_url": img_url,
                                        "boxes": boxes,
                                        "source_base64": base64_encode_image(img),
                                        "faces": aligned_faces,
                                        "embeddings": facenet_embeddings,
                                        "clip_embeddings": clip_embeddings
                            })

                        except Exception as e:
                            print(f"Failed to check/detect image {img_url}: {e}")

                # Finds all <a href=""> links, converts them to full URLs
                for link_tag in soup.find_all("a", href=True):
                    link = urljoin(current_url, link_tag["href"])
                    if is_internal_link(start_url, link) and link not in visited: # Checks if they're internal and not already visited
                        to_visit.append(link) # Adds them to the crawl list if so.

                time.sleep(5)  # For polite web crawling. 
        except Exception as e:
            st.warning(f"Failed to access {current_url}: {e}")

    return image_data

if crawl_button and url:
    with st.spinner("Crawling for images..."):
        st.session_state.image_data = crawl_images(url, max_pages=max_pages)
    st.success(f"Found {len(st.session_state.image_data)} image(s) with faces")
    


# FOR UPLOAD SELECTED FACES
if "selected_faces" not in st.session_state:
    st.session_state.selected_faces = []

# Button appears before the image results
if st.button("Upload Selected Faces to Database"):
    if not st.session_state.selected_faces:
        st.info("Please select at least one face to upload.")
    else:
        client = get_client() 
        collection_name = "facenet_faces"
        results = []

        for face_data in st.session_state.selected_faces:
            embedding = face_data["embedding"]

            similar = client.search(
                collection_name=collection_name,
                query_vector=("facenet", embedding),
                limit=1,
                score_threshold=0.995
            )

            if similar:
                st.warning(f"Duplicate found for face from {face_data['source_path']}")
                continue

            base64_face = base64_encode_tensor(face_data["face_tensor"])
            clip_embedding = face_data["clip_embedding"]
            payload = {
                "source_path": face_data["source_path"],
                "source_base64": face_data["source_base64"],
                "face_index": face_data["face_index"],
                "face_base64": base64_face,
                "embedding": embedding,
                "clip_embedding": clip_embedding
            }

            results.append(payload)

        if results:
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "facenet": record["embedding"], 
                        "clip": record["clip_embedding"]
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
            st.session_state.selected_faces = []
        else:
            st.info("No new faces were uploaded due to duplication.")

# FOR UPLOAD ALL FACES

if st.button("Upload ALL Faces to Database"):
    client = get_client()
    collection_name = "facenet_faces"
    results = []

    for data in st.session_state.image_data:
        embeddings = data["embeddings"]
        faces = data["faces"]
        source_base64 = data["source_base64"]
        source_path = data["img_url"]
        clip_embeddings = data["clip_embeddings"]

        for i, (embedding_tensor, face_tensor, clip_tensor) in enumerate(zip(embeddings, faces, clip_embeddings)):
            embedding = embedding_tensor.numpy().tolist()
            clip_embedding = clip_tensor.numpy().tolist()

            similar = client.search(
                collection_name=collection_name,
                query_vector=("facenet", embedding),
                limit=1,
                score_threshold=0.995
            )

            if similar:
                st.warning(f"Duplicate found for face {i+1} from {source_path}")
                continue

            base64_face = base64_encode_tensor(face_tensor)

            payload = {
                "source_path": source_path,
                "source_base64": source_base64,
                "face_index": i,
                "face_base64": base64_face,
                "embedding": embedding,
                "clip_embedding": clip_embedding
            }

            results.append(payload)

    if results:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "facenet": record["embedding"], 
                    "clip": record["clip_embedding"]
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
    else:
        st.info("No new faces were uploaded due to duplication.")




for idx, data in enumerate(st.session_state.image_data):
    try:
        img = Image.open(BytesIO(base64.b64decode(data["source_base64"])))
        img_with_boxes = draw_boxes_on_image(img, data["boxes"])
        st.image(img_with_boxes, caption=data["img_url"], use_container_width=True)

        faces = data["faces"]
        embeddings = data["embeddings"]

        face_cols = st.columns(len(faces))
        for i, face_tensor in enumerate(faces):
            with face_cols[i]:
                unnormalized = ((face_tensor + 1) / 2 * 255).clamp(0, 255).byte()
                pil_face = Image.fromarray(unnormalized.permute(1, 2, 0).cpu().numpy())
                st.image(pil_face, caption=f"Face {i+1}")

                face_key = f"{idx}_{i}" # AVOIDS ADDING DUPLICATES EVERY RERUN, ANTI-STREAMLIT CRINGE BEHAVIOUR.
                if st.checkbox(f"Select Face {i+1} from Image {idx+1}", key=f"select_{face_key}"):
                    if not any(f["face_index"] == i and f["source_path"] == data["img_url"] for f in st.session_state.selected_faces):
                        st.session_state.selected_faces.append({
                            "embedding": embeddings[i].tolist(),
                            "clip_embedding": data["clip_embeddings"][i].tolist(),
                            "face_tensor": face_tensor,
                            "source_base64": data["source_base64"],
                            "source_path": data["img_url"],
                            "face_index": i
                        })

    except Exception as e:
        st.warning(f"Failed to display face(s) from {data['img_url']}: {e}")

