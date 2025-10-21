import sys
import os

# Add the project root to the Python path
# Required for the imports from 'logic' directory to be recognised
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import uuid
import base64
from io import BytesIO
from qdrant_client.models import PointStruct

from logic.models import load_models, load_clip_model
from logic.db_client import get_client
from logic.image_utils import draw_boxes_on_image, tensor_to_image, base64_encode_tensor, base64_encode_image, show_source_image
from logic.constants import COLOURS

collection_name = "facenet_faces"

class FacialSearchPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.canvas = tk.Canvas(self, width=300, height=300, bg="gray")
        self.canvas.pack(pady=(10, 0))

        self.select_button = tk.Button(self, text="Select Image", command=self.open_image)
        self.select_button.pack(pady=10)

        self.face_buttons_frame = None
        self.results_frame = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        filename = file_path.split("/")[-1]
        if not file_path:
            return

        # Clean up previous widgets
        if self.face_buttons_frame:
            self.face_buttons_frame.destroy()
        if self.results_frame:
            self.results_frame.destroy()

        image = Image.open(file_path).convert("RGB")
        mtcnn, facenet = load_models()
        clip_model, clip_preprocess, _ = load_clip_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device)

        boxes, _ = mtcnn.detect(image)
        faces = mtcnn(image)
        image_with_boxes = draw_boxes_on_image(image, boxes)

        img_tk = ImageTk.PhotoImage(image_with_boxes.resize((300, 300)))
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

        if not hasattr(self, 'search_label'):
            self.search_label = tk.Label(self, text="Click on a Face to Search:", font=("Arial", 11))
            self.search_label.pack(pady=(10, 0))

        self.face_buttons_frame = tk.Frame(self)
        self.face_buttons_frame.pack()

        self.results_frame = tk.Frame(self)
        self.results_frame.pack(pady=10)

        self.selected_faces = []

        if faces is not None:
            if isinstance(faces, torch.Tensor):
                faces = [face for face in faces]

            for idx, face in enumerate(faces):
                face_img = tensor_to_image(face).resize((80, 80))
                face_img_tk = ImageTk.PhotoImage(face_img)

                face_button = tk.Button(self.face_buttons_frame, image=face_img_tk,
                                        command=lambda f=face, i=idx: self.search_face(
                                            f, i, image, facenet, clip_model, clip_preprocess, device
                                        ))
                face_button.image = face_img_tk
                face_button.grid(row=0, column=idx, padx=5)

                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(self.face_buttons_frame, text=f"Upload Face {idx+1}", variable=var)
                checkbox.var = var
                checkbox.grid(row=1, column=idx)

                self.selected_faces.append((idx, face, checkbox))

            upload_btn = tk.Button(self.face_buttons_frame, text="Upload Selected Faces",
                                   command=lambda: self.upload_faces(
                                       self.selected_faces, image, filename, facenet, clip_model, clip_preprocess, device
                                   ))
            upload_btn.grid(row=2, column=0, columnspan=len(faces), pady=10)

    def search_face(self, face, idx, original_image, facenet, clip_model, clip_preprocess, device):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        with torch.no_grad():
            embedding = facenet(face.unsqueeze(0)).squeeze().numpy()

        client = get_client()
        results = client.search(
            collection_name=collection_name,
            query_vector=("facenet", embedding),
            limit=40
        )

        tk.Label(self.results_frame, text=f"Search Results for Face {idx+1}", font=("Arial", 12, "bold")).pack()

        # Scrollable area
        canvas_wrapper = tk.Canvas(self.results_frame, width=450, height=500)
        scrollbar = tk.Scrollbar(self.results_frame, orient="vertical", command=canvas_wrapper.yview)
        scrollable_frame = tk.Frame(canvas_wrapper)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_wrapper.configure(scrollregion=canvas_wrapper.bbox("all"))
        )

        canvas_wrapper.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_wrapper.configure(yscrollcommand=scrollbar.set)

        canvas_wrapper.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        grid_frame = tk.Frame(scrollable_frame)
        grid_frame.pack()

        for i, record in enumerate(results):
            img_data = base64.b64decode(record.payload["face_base64"])
            result_img = Image.open(BytesIO(img_data)).resize((100, 100))
            result_img_tk = ImageTk.PhotoImage(result_img)

            row = (i // 4) * 3

            img_label = tk.Label(grid_frame, image=result_img_tk)
            img_label.image = result_img_tk
            img_label.grid(row=row, column=i % 4, padx=5, pady=5)

            sim_label = tk.Label(grid_frame, text=f"{record.score:.2%}")
            sim_label.grid(row=row + 1, column=i % 4)

            view_button = tk.Button(grid_frame, text="View Source", command=lambda p=record.payload: show_source_image(p, self))
            view_button.grid(row=row + 2, column=i % 4, pady=(0, 10))

    def upload_faces(self, selected_faces, original_image, original_filename, facenet, clip_model, clip_preprocess, device):
        client = get_client()
        uploaded = []

        for idx, face, checkbox in selected_faces:
            if not checkbox.var.get():
                continue

            with torch.no_grad():
                embedding = facenet(face.unsqueeze(0)).squeeze().numpy().tolist()

            clip_image = tensor_to_image(face)
            clip_input = clip_preprocess(clip_image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_embedding = clip_model.encode_image(clip_input)
                clip_embedding /= clip_embedding.norm(dim=-1, keepdim=True)
            clip_embedding = clip_embedding.squeeze().cpu().tolist()

            similar = client.search(
                collection_name=collection_name,
                query_vector=("facenet", embedding),
                limit=1,
                score_threshold=0.995
            )

            if similar:
                messagebox.showinfo("Duplicate", f"Face {idx + 1} already exists (>99.5% similarity)")
                continue

            payload = {
                "source_path": f"uploaded_by_user/{original_filename}",
                "source_base64": base64_encode_image(original_image),
                "face_index": idx,
                "face_base64": base64_encode_tensor(face),
                "embedding": embedding,
                "clip_embedding": clip_embedding
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"facenet": embedding, "clip": clip_embedding},
                payload=payload
            )

            uploaded.append(point)

        if uploaded:
            client.upsert(collection_name=collection_name, points=uploaded)
            messagebox.showinfo("Success", f"Uploaded {len(uploaded)} face(s) to the database.")
        else:
            messagebox.showinfo("Info", "No new faces to upload.")
