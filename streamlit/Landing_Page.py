import streamlit as st

st.set_page_config(
    page_title="TraceAble",
)

st.write("# Welcome to TraceAble: An OSINT Project for Identity Matching")

st.sidebar.success("Please choose a tool from the list above.")

st.markdown(
    """
    TraceAble is a toolset purpose built to aid investigations linking online and real-world identities. Created by Peter McGee (me) as a dissertation project for my bachelors degree in Computer Science and Cybersecurity.
    
    ### Tools included
    - Facial similarity search that uses facial embeddings generated leveraging the FaceNet InceptionResnetV1 model powered by a vector database to find similarity between embeddings. 
    - Text based search to query the database of face images by specific attributes/identifiers. 
    - Web crawler tool to allow the user to add images to the vector database through pasting in a hyperlink for a website to crawl.

    ### DISCLAIMER
    The web crawler tool was initially added as a feature to demonstrate a live proof of concept of the capabilities of vector database-powered facial recognition searching as an alternative to biometric. 

    It is against ToS to crawl many of the social media sites that I initially had in mind at the start of this project, and it is a grey area as to whether doing so is a civil or legal dispute.
    To not jeopardise the sanctity of this work's academic value, I must clarify that **the onus of using these tools ethically, responsibly and within legal boundaries falls on the user.**
    
"""
)