# TraceAble
Three different tools grouped together under the umbrella suite TraceAble: facial similarity searching, text-based similarity searching, and a web crawler tool to extract images of people from webpages. 

Image search embeddings are generated using InceptionResnetV1 (Facenet).

Text search embeddings are generated using OpenAI's CLIP model.

The web crawler only gathers and stores images of faces, running facial detection on all images found in a page with an MTCNN model. Utilises a vector database to store the face images as embeddings generated using pre-trained AI models, along with their metadata. Has both web (Streamlit) and desktop (Tkinter) implementations of the application. 

Utilising a Qdrant cluster for the vector database. URL and API key stored in a .env file for Tkinter desktop implementation, secrets.toml file for Streamlit web implementation.
