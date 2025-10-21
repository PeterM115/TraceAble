import tkinter as tk
from tkinter import ttk

class HomePage(ttk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.configure_gui()
        self.build_ui()

    def configure_gui(self):
        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def build_ui(self):
        # Outer wrapper that fills and centers the inner frame
        wrapper = ttk.Frame(self)
        wrapper.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Inner content frame centered inside wrapper
        content_frame = ttk.Frame(wrapper)
        content_frame.grid(row=0, column=0)
        wrapper.columnconfigure(0, weight=1)
        
        # Title
        ttk.Label(content_frame, text="TraceAble", font=("Helvetica", 24, "bold")).grid(
            row=0, column=0, pady=(30, 10), sticky="n", columnspan=2
        )

        # Subtitle
        ttk.Label(content_frame, text="An OSINT Project for Identity Matching", font=("Helvetica", 14)).grid(
            row=1, column=0, pady=(0, 20), sticky="n", columnspan=2
        )

        # Intro
        intro_text = (
            "TraceAble is a toolset purpose-built to aid investigations linking online and real-world identities. "
            "Created by Peter McGee (me) as a dissertation project for my Bachelors degree in Computer Science and Cybersecurity."
        )
        ttk.Label(content_frame, text=intro_text, wraplength=700, justify="left").grid(
            row=2, column=0, padx=40, pady=(0, 20), sticky="w", columnspan=2
        )

        # Tools Header
        ttk.Label(content_frame, text="Tools Included", font=("Helvetica", 12, "bold")).grid(
            row=3, column=0, padx=40, sticky="w", columnspan=2
        )

        # Tools list
        tools = [
            "- Facial similarity search using embeddings from FaceNet InceptionResnetV1 and a vector database.",
            "- Text-based search by specific attributes/identifiers.",
            "- Web crawler tool to add images from links into the database."
        ]
        for i, tool in enumerate(tools):
            ttk.Label(content_frame, text=tool, wraplength=700, justify="left").grid(
                row=4 + i, column=0, padx=60, sticky="w", columnspan=2
            )

        # Disclaimer Header
        ttk.Label(content_frame, text="DISCLAIMER", font=("Helvetica", 12, "bold")).grid(
            row=7, column=0, padx=40, pady=(30, 5), sticky="w", columnspan=2
        )

        # Disclaimer Text
        disclaimer_text = (
            "The web crawler tool was initially added as a live proof-of-concept to demonstrate how facial "
            "recognition via embeddings can be an alternative to biometrics.\n\n"
            "Crawling social media sites is often against their ToS and may exist in a legal grey area.\n\n"
            "To protect the academic integrity of this project, it must be stated that the ethical and lawful use "
            "of these tools is the responsibility of the user."
        )
        ttk.Label(content_frame, text=disclaimer_text, wraplength=700, justify="left", foreground="red").grid(
            row=8, column=0, padx=40, pady=(0, 30), sticky="w", columnspan=2
        )
