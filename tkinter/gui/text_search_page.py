import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
import base64
from io import BytesIO

from logic.db_client import get_client
from logic.image_utils import show_source_image
from logic.models import load_clip_model


class TextSearchPage(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.clip_tokenizer = load_clip_model()
        self.clip_model = self.clip_model.to(self.device)

        self.search_entry = None
        self.results_frame = None

        self.build_ui()

    def build_ui(self):
        tk.Label(self, text="Text-Based Face Search", font=("Arial", 14, "bold")).pack(pady=10)

        input_frame = tk.Frame(self)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Search term (e.g. glasses, beard, ginger):").pack(side="left", padx=5)
        self.search_entry = tk.Entry(input_frame, width=40)
        self.search_entry.pack(side="left", padx=5)

        search_btn = tk.Button(self, text="Search", command=self.perform_search)
        search_btn.pack(pady=5)

        self.results_frame = tk.Frame(self)
        self.results_frame.pack(pady=10)

    def perform_search(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        query_text = self.search_entry.get().strip()
        if not query_text:
            messagebox.showwarning("Missing Input", "Please enter a description to search for.")
            return

        with torch.no_grad():
            text_tokens = self.clip_tokenizer([query_text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_vector = text_embedding.squeeze().cpu().numpy().tolist()

        client = get_client()
        results = client.search(
            collection_name="facenet_faces",
            query_vector=("clip", text_vector),
            limit=40
        )

        if not results:
            tk.Label(self.results_frame, text="No matching faces found.", font=("Arial", 11)).pack()
            return

        tk.Label(self.results_frame, text=f"Search Results for: \"{query_text}\"", font=("Arial", 12, "bold")).pack()

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

        grid_frame = tk.Frame(self.results_frame)
        grid_frame.pack(pady=10)

    
