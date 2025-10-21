import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import base64
import uuid
from qdrant_client.models import PointStruct
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import threading
from playwright.sync_api import sync_playwright
import time

from logic.models import load_models, load_clip_model
from logic.image_utils import base64_encode_image, base64_encode_tensor, draw_boxes_on_image, tensor_to_image
from logic.db_client import get_client


class WebCrawlerPage(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        self.mtcnn, self.facenet = load_models()
        self.clip_model, self.clip_preprocess, _ = load_clip_model()
        self.client = get_client()

        self.image_data = []
        self.selected_faces = []

        self.build_ui()

    def build_ui(self):
        self.url_entry = tk.Entry(self, width=50)
        self.url_entry.pack(pady=10)

        self.pages_slider = tk.Scale(self, from_=1, to=30, orient="horizontal", label="Max pages to crawl", length=300)
        self.pages_slider.set(5)
        self.pages_slider.pack(pady=10)

        self.crawl_button = tk.Button(self, text="Start Crawling", command=self.start_crawling)
        self.crawl_button.pack(pady=10)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

        canvas_frame = tk.Frame(self)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, height=500)
        self.scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.image_frame = tk.Frame(self.canvas)

        self.image_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        tk.Button(self, text="Upload Selected Faces", command=self.upload_faces).pack(pady=10)
        tk.Button(self, text="Upload All Faces", command=self.upload_all_faces).pack(pady=5)


    def start_crawling(self):
        url = self.url_entry.get()
        max_pages = int(self.pages_slider.get())

        if not url:
            messagebox.showwarning("Input Error", "Please enter a valid URL.")
            return

        self.crawl_button.config(state="disabled")
        self.status_label.config(text="Crawling... Please wait.")
        self.image_data.clear()
        self.update_image_display()

        threading.Thread(target=lambda: self.crawl_images_and_update_threadsafe(url, max_pages)).start()


    def crawl_images_and_update_threadsafe(self, url, max_pages):
        results = self.crawl_images(url, max_pages)
        self.image_data.clear()
        self.image_data.extend(results)

        self.after(0, lambda: (
            self.status_label.config(text="Crawling Complete"),
            self.update_image_display(),
            self.crawl_button.config(state="normal")
        ))


    # New playwright scrolling
    def get_page_html_with_infinite_scroll(self, url, max_scrolls=30, scroll_pause=2):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            last_height = page.evaluate("document.body.scrollHeight")
            scrolls = 0

            while scrolls < max_scrolls:
                page.keyboard.press("End")
                time.sleep(scroll_pause)

                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    print("Reached end of page (no more new content).")
                    break

                last_height = new_height
                scrolls += 1

            html = page.content()
            browser.close()
            return html



    # New process image function as it will be used under 2 conditionals now. 
    def process_image_url(self, img_url, image_data):
        if any(img_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            try:
                img_response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")
                boxes, _ = self.mtcnn.detect(img)

                if boxes is not None and len(boxes) > 0:
                    aligned_faces = self.mtcnn(img)
                    if aligned_faces is not None:
                        with torch.no_grad():
                            facenet_embeddings = self.facenet(aligned_faces).detach().cpu()
                            clip_embeddings = []
                            for face_tensor in aligned_faces:
                                pil_face = tensor_to_image(face_tensor)
                                processed_face = self.clip_preprocess(pil_face).unsqueeze(0)
                                clip_embed = self.clip_model.encode_image(processed_face).cpu()
                                clip_embeddings.append(clip_embed.squeeze(0))

                        image_data.append({
                            "img_url": img_url,
                            "boxes": boxes,
                            "source_base64": base64_encode_image(img),
                            "faces": aligned_faces,
                            "embeddings": facenet_embeddings,
                            "clip_embeddings": clip_embeddings
                        })

            except Exception as e:
                print(f"Failed to process image {img_url}: {e}")



    def crawl_images(self, start_url, max_pages=5):
        visited = set()
        to_visit = [start_url]
        image_data = []
        
        # Start with the first page
        current_page = 1
        
        while current_page <= max_pages:
            # Build the URL for the current page
            current_url = f"{start_url}?page={current_page}"
            
            # Skip if we've already visited this page
            if current_url in visited:
                continue

            visited.add(current_url)

            try:
                # Get HTML content for the page
                html = self.get_page_html_with_infinite_scroll(current_url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")

                    # Find and process <img> tags
                    for img_tag in soup.find_all("img", src=True):
                        img_url = urljoin(current_url, img_tag["src"])
                        self.process_image_url(img_url, image_data)

                    # Find and process <image> tags (e.g. SVG images)
                    for image_tag in soup.find_all("image", {"xlink:href": True}):
                        img_url = urljoin(current_url, image_tag["xlink:href"])
                        self.process_image_url(img_url, image_data)

                    # Follow internal links on the page to crawl more
                    for link_tag in soup.find_all("a", href=True):
                        link = urljoin(current_url, link_tag["href"])
                        if self.is_internal_link(start_url, link) and link not in visited:
                            to_visit.append(link)

                time.sleep(2)  # Add a delay between page fetches
                current_page += 1  # Move to the next page

            except Exception as e:
                print(f"Failed to access {current_url}: {e}")

        return image_data


        return image_data


    def is_internal_link(self, base_url, link):
        return urlparse(link).netloc == urlparse(base_url).netloc or urlparse(link).netloc == ""


    def update_image_display(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        self.selected_faces.clear()

        for idx, data in enumerate(self.image_data):
            img = Image.open(BytesIO(base64.b64decode(data["source_base64"])))
            img_with_boxes = draw_boxes_on_image(img, data["boxes"])
            img_with_boxes = img_with_boxes.resize((400, int(img_with_boxes.height * (400 / img_with_boxes.width))))
            img_tk = ImageTk.PhotoImage(img_with_boxes)

            panel = tk.Label(self.image_frame, image=img_tk)
            panel.image = img_tk
            panel.grid(row=idx, column=0)

            for i, face_tensor in enumerate(data["faces"]):
                face_img = tensor_to_image(face_tensor).resize((75, 75))
                face_img_tk = ImageTk.PhotoImage(face_img)

                var = tk.BooleanVar()
                face_button = tk.Checkbutton(self.image_frame, image=face_img_tk, variable=var)
                face_button.image = face_img_tk
                face_button.grid(row=idx, column=i+1)

                self.selected_faces.append({
                    "checkbox": var,
                    "face_tensor": face_tensor,
                    "embedding": data["embeddings"][i],
                    "clip_embedding": data["clip_embeddings"][i],
                    "source_base64": data["source_base64"],
                    "img_url": data["img_url"],
                    "face_index": i,
                })


    def upload_faces(self):
        self._upload_faces(only_selected=True)


    def upload_all_faces(self):
        self._upload_faces(only_selected=False)


    def _upload_faces(self, only_selected):
        uploaded = []
        for face in self.selected_faces:
            if only_selected and not face["checkbox"].get():
                continue

            embedding = face["embedding"].numpy().tolist()
            clip_embedding = face["clip_embedding"].numpy().tolist()

            similar = self.client.search(
                collection_name="facenet_faces",
                query_vector=("facenet", embedding),
                limit=1,
                score_threshold=0.995
            )

            if similar:
                continue

            payload = {
                "source_path": face["img_url"],
                "source_base64": face["source_base64"],
                "face_index": face["face_index"],
                "face_base64": base64_encode_tensor(face["face_tensor"]),
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
            self.client.upsert(collection_name="facenet_faces", points=uploaded)
            messagebox.showinfo("Success", f"Uploaded {len(uploaded)} face(s) to the database.")
        else:
            messagebox.showinfo("Info", "No new faces to upload.")
