from huggingface_hub import HfApi, InferenceClient
from smolagents import Tool

class SearchHfSpacesTool(Tool):
    name = "search_hf_spaces"
    description = "Searches for Hugging Face Spaces that match the given query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query.",
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """
        Searches for Hugging Face Spaces that match the given query.

        Args:
            query: The search query.

        Returns:
            A string containing a list of matching Spaces, including their ID and description.
        """
        api = HfApi()
        spaces = api.list_spaces(search=query)
        return "\n".join([f"- {space.id}: {space.card_data.get('title', '')}" for space in spaces])

class CallHfSpaceApiTool(Tool):
    name = "call_hf_space_api"
    description = "Calls the API of a Hugging Face Space."
    inputs = {
        "space_id": {
            "type": "string",
            "description": "The ID of the Space to call.",
        },
        "args": {
            "type": "array",
            "description": "Positional arguments to pass to the Space's API.",
        },
        "kwargs": {
            "type": "object",
            "description": "Keyword arguments to pass to the Space's API.",
        },
    }
    output_type = "string"

    def forward(self, space_id: str, *args, **kwargs) -> str:
        """
        Calls the API of a Hugging Face Space.

        Args:
            space_id: The ID of the Space to call.
            *args: Positional arguments to pass to the Space's API.
            **kwargs: Keyword arguments to pass to the Space's API.

        Returns:
            The result of the API call.
        """
        client = InferenceClient()
        try:
            # This is a generic way to call a Gradio API.
            # The exact parameters will depend on the specific Space.
            result = client.predict(repo_id=space_id, *args, **kwargs)
            return str(result)
        except Exception as e:
            return f"Error calling Space API: {e}"