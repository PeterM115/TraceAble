import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk

from gui.home import HomePage
from gui.facial_search_page import FacialSearchPage
from gui.text_search_page import TextSearchPage
from gui.web_crawler import WebCrawlerPage

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TraceAble")

        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Resize the window to full screen while keeping the title bar
        self.geometry(f"{screen_width}x{screen_height}+0+0")

        # Overall grid layout
        self.grid_columnconfigure(0, weight=0)  # Sidebar (fixed width)
        self.grid_columnconfigure(1, weight=1)  # Content (expands)

        self.grid_rowconfigure(0, weight=1)  # Center content vertically

        # Sidebar (left)
        sidebar = tk.Frame(self, width=200, bg="#2c3e50")
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        tk.Label(sidebar, text="TraceAble", bg="#2c3e50", fg="white", font=("Arial", 16)).pack(pady=20)

        # Navigation buttons - Center them in the sidebar
        buttons = [
            ("Home", lambda: self.show_frame("Home")),
            ("Facial Search", lambda: self.show_frame("FacialSearch")),
            ("Text Search", lambda: self.show_frame("TextSearch")),
            ("Web Crawler", lambda: self.show_frame("WebCrawler")),
        ]
        for text, command in buttons:
            btn = tk.Button(sidebar, text=text, command=command, bg="#34495e", fg="white", relief="flat")
            btn.pack(fill="x", pady=5)

        # Main content container (right)
        container = tk.Frame(self)
        container.grid(row=0, column=1, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.frames = {}

        for F, name in zip(
            [HomePage, FacialSearchPage, TextSearchPage, WebCrawlerPage],
            ["Home", "FacialSearch", "TextSearch", "WebCrawler"]
        ):
            frame = F(container, self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[name] = frame

        self.show_frame("Home")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
