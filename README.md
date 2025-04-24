# 🤖🧠 Multi-Chatbot Fusion AI – GPT, Claude, Copilot, LLaMA + Google Enhanced Search

This project is a powerful **multi-chatbot integration system** that enables seamless interaction across different AI models like **OpenAI GPT, Anthropic Claude, GitHub Copilot, Local LLaMA**, and even **Google Search**, all within a **shared memory conversation history**.

Users can query all agents at once or separately, and responses are generated **in context** of the same unified conversation. It also features **dynamic conversation titling** and **web-scraped real-time context enrichment** via Google.

---

## 🔥 Key Features

- 💬 **Unified Chat Memory**
  - All agents (GPT, Claude, Copilot, Local LLaMA) access and respond using a shared conversation history

- 🧠 **Multi-Model Switching or Parallel Reply**
  - Send a query to any one or multiple models simultaneously
  - View comparative answers and consolidate insights

- 🌐 **Integrated Google Search Context**
  - Scrapes latest results for real-time info and appends it to the memory for more relevant replies

- 🧲 **Local LLaMA Model Integration**
  - Runs offline with your trained or fine-tuned LLaMA model (e.g., llama.cpp / Hugging Face)

- 📝 **Dynamic Conversation Title Generator**
  - Auto-generates titles from the conversation summary using prompt engineering
 


## ⚙️ Tech Stack

  Component | Tech / API
  LLMs | OpenAI GPT, Claude, Copilot, LLaMA
  Search | Google Search + Scraping (SerpAPI, BS4)
  Local Models | llama.cpp, Hugging Face Transformers
  Backend | Python, FastAPI or Flask
  Frontend (opt) | Gradio or custom HTML interface




## 🚀 Getting Started


1. Clone the Repo


        git clone https://github.com/yourusername/multi-chatbot-shared-memory.git
        cd multi-chatbot-shared-memory



2. Create Environment & Install Requirements

        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt


3. Setup API Key

   
       cp .env.example .env
      



## 🔍 How It Works


**Shared Memory System**

- All agents read from and write to a central memory structure managed in memory_manager.py
- This allows consistent context sharing across different model types

**Google Search Injection**

- Searches Google using your query (via SerpAPI or custom curl scraper)
- Parses the result and injects it as memory for model reference

**Local Model Support**

- You can run LLaMA or any other compatible model offline with:
- llama.cpp integration
- Hugging Face pipeline

**Dynamic Titling**

- At any point, the app can auto-generate a title from recent chat messages for easy logging
     
   
## 🔮 Future Improvements

 
 - Web UI to toggle model combinations
 - Automatic summary of all model outputs
 - Session history with export
 - Model performance tracking
 - Speech-to-text + voice response


## 🔐 Environment Setup (.env)

    OPENAI_API_KEY=your_openai_key
    CLAUDE_API_KEY=your_claude_key
    COPILOT_API_KEY=your_copilot_key
    SERPAPI_KEY=your_google_search_key




## 📜 License

MIT License – use and customize freely.

## 🤝 Contributing

Want to add new agents or improve the memory engine? PRs welcome! Open issues or contact for collaboration.
