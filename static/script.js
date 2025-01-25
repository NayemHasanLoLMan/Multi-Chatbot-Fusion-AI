const sendButton = document.getElementById("send-button");
const userInput = document.getElementById("user-input");
const chatbotSelector = document.getElementById("chatbot-selector");
const chatWindow = document.getElementById("chat-window");

// Helper function to append a message to the chat window
function appendMessage(content, isUser) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    messageDiv.classList.add(isUser ? "user-message" : "bot-message");
    messageDiv.textContent = content;
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to the bottom
}

// Handle sending a message
sendButton.addEventListener("click", async () => {
    const inputText = userInput.value.trim();
    const chatbot = chatbotSelector.value;

    if (!inputText) {
        alert("Please enter a message.");
        return;
    }

    // Append user message to the chat window
    appendMessage(inputText, true);
    userInput.value = "";

    try {
        // Choose endpoint based on chatbot selection
        const endpoint = chatbot === "llm" ? "/process-with-llm/" : "/process-with-openai/";

        // Send a POST request to the selected endpoint
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ input_text: inputText }),
        });

        const data = await response.json();

        if (response.ok) {
            // Append bot response to the chat window
            appendMessage(data.response, false);
        } else {
            throw new Error(data.detail || "An error occurred.");
        }
    } catch (error) {
        console.error("Error:", error);
        appendMessage("An error occurred while processing your message.", false);
    }
});
