from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from anthropic import Anthropic
import google.generativeai as genai
import openai
import asyncio
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import uuid
from googlesearch import search
from bs4 import BeautifulSoup
# from urllib.parse import urlparse
import json
from typing import Optional, List, Dict

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize models
llm_model = OllamaLLM(model="llama3.2", temperature=0.7, max_tokens=512)

# Gemini initialization
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai_model = genai.GenerativeModel('gemini-pro')

# Claude initialization
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# OpenAI initialization
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models
class LLMInput(BaseModel):
    input_text: str
    conversation_id: Optional[str] = None
    model: str

class SessionInfo(BaseModel):
    title: str
    created_at: datetime
    last_updated: datetime
    message_count: int
    model: str
    context: Dict[str, List]  # Store context for each model separately

# Storage structures
session_histories: Dict[str, List[Dict]] = {}
session_metadata: Dict[str, SessionInfo] = {}

# System prompts
SYSTEM_PROMPTS = {
    "llama": """You are a professional and friendly assistant. Always provide clear, concise, and precise answers. Maintain context of our conversation.""",
    "claude": """You are Claude, a helpful AI assistant created by Anthropic. Provide thoughtful, nuanced responses while maintaining conversation context.""",
    "gemini": """You are a helpful AI assistant powered by Google. Provide accurate, well-researched responses while maintaining conversation context.""",
    "openai": """You are a helpful and professional assistant. Respond in a concise and structured manner while maintaining conversation context.""",
    "google": """You are a search engine assistant that provides top search results for user queries."""
}

def init_session_context(session_id: str):
    """Initialize context for all models in a session"""
    if session_id not in session_metadata:
        session_metadata[session_id] = SessionInfo(
            title="New Conversation",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            message_count=0,
            model="",
            context={
                "llama": [],
                "claude": [],
                "openai": [],
                "gemini": [],
                "google": []
            }
        )

async def generate_response(model: str, input_text: str, session_id: str) -> str:
    """Generate response using the specified model with session-specific context"""
    try:
        init_session_context(session_id)
        session_info = session_metadata[session_id]
        shared_context = session_info.context  # Access shared context
        
        model_context = shared_context[model]

        if model == "llama":
            context_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                                    for msg in model_context])
            full_prompt = f"{SYSTEM_PROMPTS['llama']}\n\n{context_text}\nUser: {input_text}"
            response = await asyncio.to_thread(llm_model.invoke, full_prompt)
        
        elif model == "claude":
            messages = [{"role": "user", "content": msg["user"]} for msg in model_context]
            messages.extend([{"role": "assistant", "content": msg["assistant"]} for msg in model_context])
            messages.append({"role": "user", "content": input_text})
            
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.7,
                system=SYSTEM_PROMPTS["claude"],
                messages=messages
            )
            response = response.content[0].text
        
        elif model == "openai":
            messages = [{"role": "system", "content": SYSTEM_PROMPTS["openai"]}]
            for msg in model_context:
                messages.extend([
                    {"role": "user", "content": msg["user"]},
                    {"role": "assistant", "content": msg["assistant"]}
                ])
            messages.append({"role": "user", "content": input_text})
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            response = response.choices[0].message.content
        
        elif model == "gemini":
            messages = [{"role": "system", "content": SYSTEM_PROMPTS["gemini"]}]
            for msg in model_context:
                messages.extend([
                    {"role": "user", "content": msg["user"]},
                    {"role": "assistant", "content": msg["assistant"]}
                ])
            messages.append({"role": "user", "content": input_text})
            
            # Convert messages to a single prompt string
            full_prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
            
            response = await asyncio.to_thread(
                genai_model.generate_content,
                full_prompt,
                generation_config={
                    "temperature": 0.7, 
                    "max_output_tokens": 1024
                }
            )
            response = response.text
        
        elif model == "google":
            # Perform a Google search and fetch top 5 results
            search_results = search(input_text, num_results=5)
            formatted_results = []
            for result in search_results:
                try:
                    # Fetch page title and description using BeautifulSoup
                    response = requests.get(result, timeout=50)
                    soup = BeautifulSoup(response.content, "html.parser")

                    # Extract title
                    title = soup.title.string if soup.title else "No Title"

                    # Extract meta description
                    meta_description = soup.find("meta", {"name": "description"})
                    description = meta_description["content"] if meta_description else "No description available."

                    # Format result
                    formatted_results.append(
                        f"<div style='margin-bottom: 10px;'>"
                        f"<a href='{result}' target='_blank' style='font-size: 16px; color: blue;'>{title}</a><br>"
                        f"<span style='font-size: 14px; color: gray;'>{description}</span><br>"
                        f"<span style='font-size: 12px; color: green;'>{result}</span>"
                        f"</div>"
                    )
                except Exception as e:
                    # If fetching metadata fails, fallback to showing the link only
                    formatted_results.append(f"<a href='{result}' target='_blank'>{result}</a>")

            return "<br>".join(formatted_results)
        
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Update context for all models
        for key in shared_context.keys():
            shared_context[key].append({"user": input_text, "assistant": response})
            shared_context[key] = shared_context[key][-10:]  # Limit to last 10 exchanges

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


def generate_chat_title(session_context: Dict[str, List]) -> str:
    """
    Generate a simple title based on the first 5 interactions of the conversation.
    Defaults to 'New Chat' if no meaningful title can be generated.
    """
    try:
        # Extract the first 5 interactions (user and assistant messages)
        conversation_summary = []
        interaction_count = 0
        for model, messages in session_context.items():
            for message in messages:
                if interaction_count >= 5:
                    break
                if "user" in message:
                    conversation_summary.append(message["user"])
                    interaction_count += 1
                if "assistant" in message:
                    conversation_summary.append(message["assistant"])
                    interaction_count += 1
            if interaction_count >= 5:
                break

        # Join the summary into a single string
        full_conversation = " ".join(conversation_summary).strip()

        # If there is meaningful content, generate a title
        if full_conversation:
            title_prompt = (
                f"Generate a concise and descriptive title for this conversation: {full_conversation}. "
                f"Keep it simple and creative."
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Faster and efficient for simple tasks
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in summarizing conversations."},
                    {"role": "user", "content": title_prompt}
                ],
                max_tokens=10,  # Keep the title short
                temperature=0.5  # More deterministic
            )
            title = response.choices[0].message.content.strip()
            return title.title() if title else "New Chat"

        # Fallback to default title if no content is available
        return "New Chat"
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the HTML page for the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_input(data: LLMInput):
    """Process input text with the selected model."""
    try:
        session_id = data.conversation_id or str(uuid.uuid4())
        is_new_session = session_id not in session_histories

        if is_new_session:
            session_histories[session_id] = []
            init_session_context(session_id)

        # Generate response with shared context
        response_text = await generate_response(data.model, data.input_text, session_id)

        # Update session metadata
        session_metadata[session_id].last_updated = datetime.now()
        session_metadata[session_id].message_count += 1
        session_metadata[session_id].model = data.model

        # Generate a dynamic title
        session_metadata[session_id].title = generate_chat_title(session_metadata[session_id].context)

        # Add to session history
        message = {
            "model": data.model,
            "input_text": data.input_text,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        session_histories[session_id].append(message)

        # Save session data
        await save_session_data(session_id)

        return {
            "conversation_id": session_id,
            "input": data.input_text,
            "response": response_text,
            "title": session_metadata[session_id].title,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_sessions():
    """Retrieve all session titles and metadata"""
    try:
        sessions = []
        for session_id, metadata in session_metadata.items():
            sessions.append({
                "session_id": session_id,
                "title": metadata.title,
                "created_at": metadata.created_at,
                "last_updated": metadata.last_updated,
                "message_count": metadata.message_count,
                "model": metadata.model
            })
        return {"sessions": sorted(sessions, key=lambda x: x["last_updated"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {e}")

@app.get("/session-history/{session_id}")
async def get_session_history(session_id: str):
    """Retrieve session history"""
    try:
        if session_id in session_histories:
            return {
                "session_id": session_id,
                "metadata": session_metadata[session_id],
                "history": session_histories[session_id]
            }

        # Try to load from file if not in memory
        try:
            with open(f"sessions/{session_id}.json", "r") as f:
                data = json.load(f)
                session_histories[session_id] = data["history"]
                metadata = data["metadata"]
                metadata["context"] = metadata.get("context", {model: [] for model in SYSTEM_PROMPTS.keys()})
                session_metadata[session_id] = SessionInfo(**metadata)
                return {
                    "session_id": session_id,
                    "metadata": session_metadata[session_id],
                    "history": session_histories[session_id]
                }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {e}")

async def save_session_data(session_id: str):
    """Save session data to file"""
    try:
        os.makedirs("sessions", exist_ok=True)
        session_data = {
            "metadata": {
                **session_metadata[session_id].dict(),
                "context": session_metadata[session_id].context
            },
            "history": session_histories[session_id]
        }
        async with asyncio.Lock():
            with open(f"sessions/{session_id}.json", "w") as f:
                json.dump(session_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving session data: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
