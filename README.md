# SmolMCP: A Hugging Face Agent

This Hugging Face Space hosts SmolMCP, a powerful AI agent capable of using other Hugging Face Spaces and tools to complete complex tasks.

## How to Use

1.  **Provide Your Access Tokens:**
    To use this Space, you need to provide API keys for the services you want the agent to use. Go to the "Settings" tab of this Space and add the following secrets:
    *   `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face Hub token. This is required for the agent to interact with the Hugging Face Hub and use the Inference API.

2.  **Enter Your Prompt:**
    In the Gradio interface, type your request for the agent in the text box.

3.  **Run the Agent:**
    Click the "Run" button to start the agent. The agent will then generate and execute code to fulfill your request.

## Technical Details

*   **Framework:** Gradio
*   **Agent:** Hugging Face SmolAgent (CodeAgent)
*   **Model:** GLM4.6 (or other models supported by LiteLLM)
*   **Infrastructure:** Hugging Face ZeroGPU

This project is built by Jules, your AI software engineer.