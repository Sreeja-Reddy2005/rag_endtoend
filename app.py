from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
import os


from auth import login, register
from chat_db import *

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from PIL import Image
import io
import base64
import os
import re

DOC_CACHE = {}


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="my_token"
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_tables()


class ChatRequest(BaseModel):
    prompt: str
    conversation_id: int | None = None
    user_id: int | None = None


def chat_llm(prompt):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt[:4000]}],
    )
    return res.choices[0].message.content


def summarize_conversation(chat_history, batch_size=6):
    if not chat_history:
        return "No conversation to summarize."

    summaries = []
    batch = []
    i = 0

    while i < len(chat_history):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]["content"][:200]

            bot_msg = ""
            if i + 1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                bot_msg = chat_history[i+1]["content"][:400]

            batch.append(f"Q: {user_msg}\nA: {bot_msg}")

            if len(batch) == batch_size:
                combined = "\n\n".join(
                    [f"{idx+1}. {item}" for idx, item in enumerate(batch)]
                )

                summary = chat_llm(f"""
Summarize EACH Q&A into ONLY 1-2 lines.
Include ALL questions.
Do NOT skip any.
Keep it concise and structured.

{combined}
""")

                summaries.append(summary)
                batch = []

        i += 1

    if batch:
        combined = "\n\n".join(
            [f"{idx+1}. {item}" for idx, item in enumerate(batch)]
        )

        summary = chat_llm(f"""
Summarize EACH Q&A into ONLY 1-2 lines.
Include ALL questions.
Do NOT skip any.
Keep it concise and structured.

{combined}
""")

        summaries.append(summary)

    return "\n\n".join(summaries)


def process_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def retrieve_relevant_chunks(query, documents, top_k=5):

    tokenized_docs = [
        doc.page_content.lower().split() for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)


    boosted_scores = []
    for score, doc in zip(scores, documents):
        text = doc.page_content

        bonus = 0


        if any(char.isdigit() for char in text):
            bonus += 5


        if "+" in text or "-" in text:
            bonus += 5

        boosted_scores.append(score + bonus)

    ranked = sorted(
        zip(boosted_scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]



def expand_query(prompt):
    words = re.findall(r'\w+', prompt.lower())
    expanded = set(words)

    for word in words:
        if not word.endswith("s"):
            expanded.add(word + "s")

    return " ".join(expanded)


def smart_rag_response(prompt, rag_context, is_general_query):

    if not rag_context:
        return chat_llm(prompt)

    prompt_lower = prompt.lower()


    if any(word in prompt_lower for word in ["analyse", "analyze", "summary", "overview"]):
        rag_prompt = f"""
Give a detailed explanation based ONLY on the context.

Do NOT extract values.
Explain clearly and completely.

Context:
{rag_context}

Question:
{prompt}
"""


    elif any(word in prompt_lower for word in ["points", "values", "score", "reward"]):
        rag_prompt = f"""
Extract all event-value pairs from the context.

Rules:
- Look for numbers (+ or - values)
- Each number corresponds to an event
- Return ALL pairs clearly

Format:
Event → Value

Context:
{rag_context}

Question:
{prompt}
"""

  
    else:
        rag_prompt = f"""
Answer the question based ONLY on the context.

Context:
{rag_context}

Question:
{prompt}
"""

    return chat_llm(rag_prompt)



@app.post("/register")
def register_api(username: str, password: str):
    success = register(username, password)

    if not success:
        raise HTTPException(status_code=400, detail="User already exists")

    return {"message": "User registered successfully"}

@app.post("/login")
def login_api(username: str, password: str):
    user = login(username, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login successful",
        "user_id": user.id
    }


@app.post("/new-chat")
def new_chat(user_id: int, title: str = "New Chat"):
    cid = create_conversation(user_id, title)
    return {
        "message": "New chat created",
        "conversation_id": cid
    }





def detect_source(prompt, has_image, has_pdf, last_source):
    prompt = prompt.lower()

    if "image" in prompt:
        return "image"

    if "doc" in prompt or "pdf" in prompt or "document" in prompt:
        return "pdf"


    if last_source:
        return last_source


    if has_pdf:
        return "pdf"

    if has_image:
        return "image"

    return "general"

@app.post("/chat")
def chat_api(request: ChatRequest):

    try:
        print("INPUT:", request)

        prompt = request.prompt
        conversation_id = request.conversation_id
        user_id = request.user_id

        if len(prompt.split()) <= 3:
            prompt = f"Explain in detail: {prompt}"

        is_general_query = any(word in prompt.lower() for word in [
            "analyse", "analyze", "summary", "overview"
        ])

        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id required")


        file_path = load_document_path(conversation_id)
        documents = None

        if file_path:
            if file_path in DOC_CACHE:
                documents = DOC_CACHE[file_path]
            else:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                documents = splitter.split_documents(docs)
                DOC_CACHE[file_path] = documents


        img_base64 = load_image(conversation_id)
        has_image = bool(img_base64)
        has_pdf = bool(file_path)

        last_source = get_last_source(conversation_id)

        source = detect_source(prompt, has_image, has_pdf, last_source)

        print("SOURCE:", source)


        if source == "image" and img_base64:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt[:800]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }],
            )
            res = response.choices[0].message.content

  
        elif source == "pdf" and documents:

            query = expand_query(prompt)

 
            retrieved = retrieve_relevant_chunks(query, documents, top_k=7)

            print("\n--- RETRIEVED CHUNKS ---")
            for score, doc in retrieved:
                print(score, doc.page_content[:200])

            if retrieved:

                raw_context = "\n\n".join([doc.page_content for _, doc in retrieved])

     
                clean_context = re.sub(r'\+\s*(\d)\s*(\d)', r'+\1.\2', raw_context)
                clean_context = re.sub(r'-\s*(\d)\s*(\d)', r'-\1.\2', clean_context)

                rag_context = clean_context

    
                print("\n--- FINAL CONTEXT ---\n", rag_context[:500])

            else:
                rag_context = ""

            res = smart_rag_response(
                prompt[:800],
                rag_context[:1500],
                is_general_query
            )


        else:
            res = chat_llm(prompt[:1000])


        save_message(conversation_id, "user", prompt)
        save_message(conversation_id, "assistant", res)


        save_last_source(conversation_id, source)

        return {
            "response": res,
            "conversation_id": conversation_id
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))  


@app.post("/upload-image")
async def upload_image(conversation_id: int = Form(...), file: UploadFile = File(...)):

    try:
  
        contents = await file.read()
        img_base64 = base64.b64encode(contents).decode("utf-8")

    
        save_image(conversation_id, img_base64)
        save_last_source(conversation_id, "image")

   
        

        return {
    "message": "Image uploaded successfully",
    "conversation_id": conversation_id
}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf")
async def upload_pdf(conversation_id: int = Form(...), file: UploadFile = File(...)):

    try:
  
        file_path = f"temp_{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())
        save_document(conversation_id, file_path)
        save_last_source(conversation_id, "pdf")

        return {
    "message": "PDF uploaded successfully",
    "conversation_id": conversation_id
}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{user_id}")
def get_user_conversations(user_id: int):
    return get_conversations(user_id)



@app.get("/messages/{conversation_id}")
def get_chat(conversation_id: int):
    return get_messages(conversation_id)


@app.get("/summarize/{conversation_id}")
def summarize(conversation_id: int):
    chat = get_messages(conversation_id)
    summary = summarize_conversation(chat)
    return {"summary": summary}


@app.get("/")
def health():
    return {"status": "API running"}