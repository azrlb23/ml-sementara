# Deployment Guide - SegmentIQ

This guide outlines how to deploy the SegmentIQ Streamlit application to production.

---

## Option 1: Streamlit Community Cloud (Recommended & Easiest)
Streamlit offers free hosting for public repositories on GitHub.

### Steps:
1. **Push your code to GitHub**:
   Ensure the following structure is committed to your repository:
   ```text
   segmentation-web/
   ├── app.py
   ├── views/
   ├── utils/
   ├── data/
   │   ├── processed/
   │   └── Labeled/
   ├── models/
   └── requirements.txt
   ```
2. **Log in to Streamlit Community Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. **Deploy App**:
   - Click **"Create app"** (or **"New app"**).
   - Select your Repository, Branch (e.g. `kayis`), and Main file path (`app.py`).
   - Click **"Deploy!"**.
   - Your app will be live on a public URL in a few minutes.

---

## Option 2: Hugging Face Spaces (Great for ML/Data Apps)
Hugging Face Spaces provides a free hosting tier specifically built for Python ML web applications.

### Steps:
1. Log in to [Hugging Face](https://huggingface.co/).
2. Go to **Spaces** -> **"Create new Space"**.
3. Set your Space name, select **Streamlit** as the SDK, and choose the free CPU tier.
4. Clone the space repository locally or upload your files directly through the Hugging Face web interface.
5. Commit and push your project files (including `requirements.txt`). Hugging Face will automatically build and launch the Streamlit server.

---

## Option 3: Docker Container (Local or VPS Deployment)
If you want to host on a private Virtual Private Server (VPS) or cloud providers like AWS, GCP, Render, or Railway, you can use the provided [Dockerfile](file:///d:/Machine%20Learning/segmentation-web/Dockerfile).

### Build & Run locally:
1. **Build the image**:
   ```bash
   docker build -t segmentiq .
   ```
2. **Run the container**:
   ```bash
   docker run -p 8501:8501 segmentiq
   ```
   Open `http://localhost:8501` to view the application.

---

## Option 4: Render / Railway (PaaS)
PaaS providers automatically build and deploy your application directly from your GitHub repository using the `Dockerfile`.

### Steps (Render):
1. Sign in to [Render](https://render.com/).
2. Click **"New +"** -> **"Web Service"**.
3. Connect your GitHub repository.
4. Render will automatically detect the `Dockerfile` and select **Docker** as the Environment.
5. Choose the **Free** instance type.
6. Click **"Deploy Web Service"**.
