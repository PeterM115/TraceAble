
from io import BytesIO
import base64
from PIL import Image
from PIL import ImageDraw
import tkinter as tk
from PIL import Image, ImageTk
from logic.constants import COLOURS

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

def show_source_image(payload, root):
    top = tk.Toplevel(root)
    top.title("Source Image")

    img_data = base64.b64decode(payload["source_base64"])
    img = Image.open(BytesIO(img_data))
    img.thumbnail((500, 500))
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(top, image=img_tk)
    label.image = img_tk
    label.pack()

    source_path = payload.get("source_path", "Unknown")
    tk.Label(top, text=f"Source Path: {source_path}").pack(pady=5)

    # Center the popup relative to root
    top.update_idletasks()
    window_width = top.winfo_width()
    window_height = top.winfo_height()

    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    root_height = root.winfo_height()

    x = root_x + (root_width // 2) - (window_width // 2)
    y = root_y + (root_height // 2) - (window_height // 2)

    top.geometry(f"+{x}+{y}")