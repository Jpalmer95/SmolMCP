import gradio as gr
import os
import spaces
import logging
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel
from hf_tools import SearchHfSpacesTool, CallHfSpaceApiTool

# Set up logging
logging.basicConfig(filename='app_detailed.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application starting...")

try:
    # --- Agent Configuration ---

# 1. Set up the model
# We will use the Hugging Face Inference API.
# The user needs to set the HUGGING_FACE_HUB_TOKEN in the Space's secrets.
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
)

# 2. Define the tools
# The agent will have access to a web search tool and Hugging Face tools.
tools = [WebSearchTool(), SearchHfSpacesTool(), CallHfSpaceApiTool()]

# 3. Instantiate the agent
agent = CodeAgent(
    tools=tools,
    model=model,
    # stream_outputs=True, # Streaming is not yet supported in this Gradio setup
)

# --- Gradio Interface ---

@spaces.GPU
def run_agent(prompt):
    """
    This function runs the SmolAgent with the user's prompt.
    The @spaces.GPU decorator ensures that this function runs on a GPU.
    """
    try:
        # The agent.run() method returns a string with the final answer.
        answer = agent.run(prompt)
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface
iface = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(lines=4, label="Your Prompt", placeholder="Enter your request for the agent..."),
    outputs=gr.Markdown(label="Agent's Response"),
    title="SmolMCP: A Hugging Face Agent",
    description="This agent can use tools to answer your questions. Enter your Hugging Face Hub token in the settings to allow the agent to use Hugging Face tools.",
)

# Launch the Gradio app
    logging.info("Launching Gradio app...")
    iface.launch()
    logging.info("Gradio app launched.")

except Exception as e:
    logging.error(f"An unhandled exception occurred during setup: {e}", exc_info=True)
    # Also print to stderr so it's visible if the app is run in the foreground
    print(f"An unhandled exception occurred during setup: {e}")