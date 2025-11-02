# 🌐 TalkBuddy — Multilingual Translator App

**TalkBuddy** is an AI-powered translation web app built using **Gradio** and **Hugging Face Transformers**.  
It enables smooth translation across **50 languages** with a clean, elegant interface and real-time switching.

---

## 🚀 Live Demo
👉 [Open TalkBuddy on Hugging Face](https://huggingface.co/spaces/YaminiVatluri/TalkBuddy)

---

## 💡 Features

- 🌍 Supports **50 languages** including English, Hindi, French, Chinese, Telugu, Tamil, and more.  
- ⚡ Uses **MBART-50 (facebook/mbart-large-50-many-to-many-mmt)** for high-quality translation.  
- 🎨 Built with **Gradio Blocks** for a modern, minimal UI.  
- 🔁 Instant **language swap** functionality.  
- 💬 Inspirational quote & stylish design with subtle gradients.  

---

## 🧠 Tech Stack

| Component | Description |
|------------|--------------|
| **Frontend** | Gradio UI |
| **Backend** | MBART model from Hugging Face Transformers |
| **Framework** | Python (Transformers, Torch) |
| **Deployment** | Hugging Face Spaces |

---

## 🛠️ Installation (For Local Setup)

```bash
# Clone the repository
git clone https://github.com/Yamini557/TalkBuddy.git
cd TalkBuddy

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
