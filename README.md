# APIs
A unified collection of APIs for speech and language processing, including Speech-to-Text, Chatbot, Speech Denoising, and Text Summarization. Built in Python with FastAPI endpoints for real-time, and product applications in audio and text intelligence.

# peech-to-Text (STT) API
The STT module performs ***automatic speech recognition*** using ***Whisper*** model
- Converts audio input (WAV) to accurate text.
- Supports ***batch or streaming transcription***.
- Ready for ***medical, educational, and assistive applications***.

# Chatbot API
A context-aware conversational engine powered by ***large language models (LLMs)*** such as GPT-turbo or custom fine-tuned dialogue models.
- Handles ***multi-turn dialogues*** and context tracking.
- Supports ***domain-specific knowledge integration***.
- ***Combined with STT for voice-to-voice assistants***.

# Speech Denoising API
Applies ***transformer-based speech enhancement*** to remove background noise, reverb, and distortion.
- Utilizes models like ***DNN, UNet, or ConFormer***.
- Provides ***clean, high-quality audio output***.
- Suitable for ***telehealth, hearing-aid research, and voice analytic***s.

# Docker Deployment
Each module includes a dedicated ***Dockerfile*** with pre-configured dependencies for quick setup.
Run any API locally or on cloud with a single command:
- docker build -t speech_ai_api .
- docker run -p 8000:8000 speech_ai_api

Fully compatible with ***AWS, GCP, Azure, and local servers***.
Supports ***scalable deployment, load balancing***, and ***microservice orchestration*** via Docker Compose or Kubernetes.
