import os
from typing import List, Dict
from anthropic import Anthropic
import google.generativeai as genai
import openai
from googlesearch import search
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
import requests
import environ
import time


env = environ.Env()
environ.Env.read_env() 


genai.configure(api_key=env("GEMINI_API_KEY"))
genai_model = genai.GenerativeModel('gemini-2.0-flash')

anthropic_client = Anthropic(api_key=env("CLAUDE_API_KEY"))

openai.api_key = env("OPENAI_API_KEY")


# System prompts
SYSTEM_PROMPTS = {

    "claude": """You are Claude, a helpful AI assistant created by Anthropic. Provide thoughtful, nuanced responses while maintaining conversation context.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.""",

    "gemini": """You are a helpful AI assistant powered by Google. Provide accurate, well-researched responses while maintaining conversation context.Never mention you are AI or Assistent in the response.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.""",

    "chatgpt": """You are a helpful and professional assistant. Respond in a concise and structured manner while maintaining conversation context.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate.""",


    "google": """You are a search engine assistant that provides top search results for user queries.

    RESPONSE STRUCTURE

    - Use consistent markdown formatting with proper spacing and paragraph breaks.
    - Organize complex responses using clear headers and logical sections.
    - Scale response length to match query complexity.
    - Include relevant code blocks, examples, or reference materials when appropriate."""
     

}


def generate_title_from_message(message: str, model: str) -> str:
    """
    Generate a concise title from the first message using the specified model.
    """
    try:
        title_prompt = f"Generate a very short, concise title (max 6 words) Which summarizes this message: {message}"
        
        if model == "claude":
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,  # Short response for title
                temperature=0.3,  # Lower temperature for more focused output
                messages=[{"role": "user", "content": title_prompt}]
            )
            title = response.content[0].text if response.content else "New Conversation"
            
        elif model == "chatgpt":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": title_prompt}],
                max_tokens=50,
                temperature=0.3
            )
            title = response['choices'][0]['message']['content'] if 'choices' in response else "New Conversation"
            
        elif model == "gemini":
            response = genai_model.generate_content(
                title_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 50
                }
            )
            title = response.text if response else "New Conversation"

        elif model == "google":
            from googlesearch import search
            import requests
            from bs4 import BeautifulSoup

            try:
                search_results = list(search(message, num=1))  # Get top search result
                if search_results:
                    url = search_results[0]
                    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(response.content, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else "Search Result"
                else:
                    title = "Google Search"
            except Exception as e:
                title = "Google Search"
            
        else:
            title = "New Conversation"
            
        # Clean up the title
        title = title.strip().strip('"').strip("'")
        return title[:50]  # Limit title length
        
    except Exception as e:
        return "New Conversation"


def generate_response(model: str, input_text: str, chat_history: List[Dict]) -> str:
    try:
        conversation_title = (
            generate_title_from_message(input_text, model) 
            if not chat_history 
            else chat_history[0].get('title', "New Conversation")
        )

        messages = []
        for msg in chat_history:
            if "user" in msg:
                messages.append({"role": "user", "content": msg["user"]})
            if "assistant" in msg:
                messages.append({"role": "assistant", "content": msg["assistant"]})
        messages.append({"role": "user", "content": input_text})


        if model == "claude":
            messages = [{"role": "user", "content": msg["user"]} for msg in chat_history]
            messages.extend([{"role": "assistant", "content": msg["assistant"]} for msg in chat_history])
            messages.append({"role": "user", "content": input_text})
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.7,
                system=SYSTEM_PROMPTS["claude"],
                messages=messages
            )
            response = response.content[0].text if response.content else "No response generated."

        elif model == "chatgpt":
            messages = [{"role": "system", "content": SYSTEM_PROMPTS["chatgpt"]}]
            for msg in chat_history:
                messages.extend([
                    {"role": "user", "content": msg["user"]},
                    {"role": "assistant", "content": msg["assistant"]}
                ])
            messages.append({"role": "user", "content": input_text})

            # Direct synchronous call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            # Correctly access the text response
            response = response['choices'][0]['message']['content'] if 'choices' in response and len(response['choices']) > 0 else "No response generated."

        elif model == "gemini":
            messages = [{"role": "system", "content": SYSTEM_PROMPTS["gemini"]}]
            for msg in chat_history:
                messages.extend([
                    {"role": "user", "content": msg["user"]},
                    {"role": "assistant", "content": msg["assistant"]}
                ])
            messages.append({"role": "user", "content": input_text})
            full_prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
            response = genai_model.generate_content(
                full_prompt, 
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024
                }
            ) 
            response = response.text if response else "No response generated."

        elif model == "google":
            # Perform a Google search and fetch top 5 results
            search_results = search(input_text, num=4)
            formatted_results = ""
            counter = 0
            for result in search_results:
                if counter > 4:
                  break

                try:
                    # Fetch page title and description using BeautifulSoup
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(result, timeout=50, headers= headers)
                    time.sleep(3)
                    if response.status_code == 200:
                        counter += 1
                        soup = BeautifulSoup(response.content, "html.parser")
                    else:
                        continue

                    # Extract title
                    title_URl = soup.title.string if soup.title else "No Title"
                    meta_description = soup.find("meta", {"name": "description"})
                    description = meta_description["content"] if meta_description else "No description available."


                    # Format result
                    
                    formatted_results += f"""<div style='margin-bottom: 10px;'>
                        <a href='{result}' target='_blank' style='font-size: 14px; color: blue;'>{title_URl}</a><br>
                        <span style='font-size: 14px; color: gray;'>{description}</span><br>
                        <span style='font-size: 12px; color: green;'>{result}</span>
                        </div>"""
                    
                except Exception as e:
                    # If fetching metadata fails, fallback to showing the link only
                    formatted_results.append(f"<a href='{result}' target='_blank'>{result}</a>")

            print(formatted_results)
            # return "<br>".join(formatted_results)
            response = formatted_results.replace('\n', '')

        
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Append new interaction to the chat history
        chat_history.append({"user": input_text, "assistant": response})

        return {"content": response, "title": conversation_title}

    except Exception as e:
        return {"content": f"Error generating response: {str(e)}", "title": "Error"}

