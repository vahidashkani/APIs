#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:50:34 2025

@author: nca
"""

# main.py
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from stt_service import convert_speech_to_text

# LangChain / RAG bits
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

# --------- API KEY (you hardcoded earlier; keep as-is or use env) ----------
GROQ_API_KEY = ""

# Shared LLM + embeddings
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

prompt = ChatPromptTemplate.from_template(
    "Use the context (transcript) to answer.\n<context>\n{context}\n</context>\n\nQuestion: {input}"
)

app = FastAPI(title="Whisper STT + RAG Chatbot")

BASE_DIR = Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
SESSIONS_DIR = BASE_DIR / "sessions"
UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

def _session_dir(session_id: str) -> Path:
    d = SESSIONS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_transcript(session_id: str, text: str) -> Path:
    d = _session_dir(session_id)
    p = d / "transcript.txt"
    p.write_text(text, encoding="utf-8")
    return p

def _build_and_save_faiss(session_id: str, text: str) -> None:
    docs = splitter.create_documents([text])
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(str(_session_dir(session_id) / "faiss"))

def _load_faiss(session_id: str) -> FAISS:
    d = _session_dir(session_id) / "faiss"
    if not d.exists():
        raise HTTPException(status_code=404, detail="Session FAISS not found")
    return FAISS.load_local(str(d), embeddings, allow_dangerous_deserialization=True)

def _answer_with_rag(session_id: str, question: str) -> str:
    db = _load_faiss(session_id)
    retriever = db.as_retriever()
    doc_chain = create_stuff_documents_chain(llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)
    return chain.invoke({"input": question})["answer"]

@app.post("/start-session")
async def start_session(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        wav_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(wav_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        transcript = convert_speech_to_text(str(wav_path))
        _save_transcript(session_id, transcript)
        _build_and_save_faiss(session_id, transcript)

        return JSONResponse({"session_id": session_id, "transcription": transcript})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT/Indexing failed: {e}")

@app.post("/append-audio")
async def append_audio(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        wav_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(wav_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        new_text = convert_speech_to_text(str(wav_path))
        txt_path = _session_dir(session_id) / "transcript.txt"
        all_text = (txt_path.read_text(encoding="utf-8") + "\n" if txt_path.exists() else "") + new_text
        _save_transcript(session_id, all_text)
        _build_and_save_faiss(session_id, all_text)

        return {"session_id": session_id, "appended_chars": len(new_text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Append failed: {e}")

@app.post("/chat")
async def chat(session_id: str = Form(...), question: str = Form(...)):
    try:
        answer = _answer_with_rag(session_id, question)
        return {"session_id": session_id, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
