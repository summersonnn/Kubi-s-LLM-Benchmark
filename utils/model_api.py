"""
Model-agnostic API caller for OpenRouter integration.
Supports model selection by index and detailed reasoning retrieval.
Manages API configuration via environment variables and local files.
"""

import asyncio
import os
import time
from typing import Any, List, Optional
from types import SimpleNamespace
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from dotenv import load_dotenv
from utils.utils import setup_logging

logger = setup_logging(__name__)

class ModelAPI:
    """
    A class to handle interactions with OpenRouter APIs using the OpenAI client.
    Configuration is loaded from environment variables and a models.txt file.
    """

    def __init__(self, env_path: Optional[str] = None, models_path: str = "config/models.txt") -> None:
        """
        Initializes the ModelAPI with configuration from environment variables
        and a list of models from a file.
        
        Args:
            env_path: Optional path to the .env file.
            models_path: Path to the models.txt file.
        """
        # Load environment variables from .env file
        load_dotenv(dotenv_path=env_path)
        
        # OpenRouter configuration
        self.base_url = os.getenv("MODEL_API_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("MODEL_API_KEY")
        
        # Load models from file
        self.models: List[str] = []
        if os.path.exists(models_path):
            with open(models_path, "r") as f:
                self.models = [
                    line.strip() for line in f 
                    if line.strip() and not line.strip().startswith("!")
                ]
        
        # Common parameters
        try:
            self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "8196"))
        except (ValueError, TypeError):
            self.max_tokens = 8196
            
        try:
            self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        except (ValueError, TypeError):
            self.temperature = 0.7

        # Timeout configuration
        try:
            self.timeout = float(os.getenv("MODEL_API_TIMEOUT", "90.0"))
        except (ValueError, TypeError):
            self.timeout = 90.0

        # Validate required fields
        if not self.api_key:
            raise ValueError("Missing required environment variable: MODEL_API_KEY")
        
        # Initialize AsyncOpenAI client
        # Clean up base_url if it ends with /api/v (some users might forget the 1)
        if self.base_url.endswith("/api/v"):
            self.base_url += "1"
            
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    async def call(
        self, 
        prompt: str, 
        model_index: Optional[int] = None, 
        model_name: Optional[str] = None,
        reasoning: bool = True, 
        **kwargs: Any
    ) -> Any:
        """
        Sends a completion request to the selected model.
        
        Args:
            prompt: The user prompt.
            model_index: Index of the model in models.txt (0-based).
            reasoning: Whether to enable reasoning via extra_body.
            **kwargs: Additional parameters to override defaults.
            
        Returns:
            The ChatCompletion response object from the API.
        """
        if not self.models:
            raise ValueError("No models loaded from models.txt")
        
        if model_name:
            selected_model = model_name
        else:
            if model_index is None:
                model_index = 0
            if model_index < 0 or model_index >= len(self.models):
                raise IndexError(f"Model index {model_index} out of range (0-{len(self.models)-1})")
            selected_model = self.models[model_index]
        
        # Build messages
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        
        # Handle extra_body for reasoning
        extra_body = kwargs.pop("extra_body", {})
        if reasoning:
            if "reasoning" not in extra_body:
                extra_body["reasoning"] = {"effort": "high"}
        
        # Determine parameters to avoid duplicates in **kwargs
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        timeout = kwargs.pop("timeout", self.timeout)

        # Call the API with asyncio.wait_for to ensure strict timeout cancellation
        
        # We need to ensure stream=False is used
        kwargs["stream"] = False
        
        try:
             response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=None, # We handle timeout with asyncio.wait_for
                    extra_body=extra_body,
                    **kwargs
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Model call timed out after {timeout} seconds")

        # Simplified handling for non-streaming response
        # The response object is directly compatible with what main.py expects
        # We just need to attach our custom 'reasoning_details' attribute to the message
        
        choice = response.choices[0]
        message = choice.message
        
        # Try to extract reasoning if available (provider dependent)
        reasoning_text = None
        if hasattr(message, "reasoning"):
             reasoning_text = message.reasoning
        elif hasattr(message, "reasoning_content"):
             reasoning_text = message.reasoning_content
             
        # Monkey-patch the message object to include reasoning_details
        # Use setattr to avoid type checking issues or read-only errors if it's a Pydantic model
        # But ChatCompletionMessage is usually Pydantic. Safer to just return the response
        # and letting the caller handle it, BUT main.py expects `message.reasoning_details`.
        # Since response objects might be immutable, let's wrap or modify carefully.
        # Actually, main.py expects:
        # response.choices[0].message.content
        # response.choices[0].message.reasoning_details
        
        # Let's create a SimpleNamespace wrapper if needed, or just set the attribute if allowed.
        # OpenRouter/OpenAI Pydantic models might effectively be immutable.
        # Safer approach: Create the mock structure we used before, but populated from the full response.
        
        content = message.content
        usage = response.usage
        
        if not usage:
            usage = SimpleNamespace(completion_tokens=0, cost=0.0)

        mock_message = SimpleNamespace(
            content=content,
            reasoning_details=reasoning_text
        )
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            usage=usage
        )
        
        return mock_response



if __name__ == "__main__":
    try:
        api = ModelAPI()
        logger.info("Loaded %d models.", len(api.models))
        for i, m in enumerate(api.models):
            logger.info("[%d] %s", i, m)
    except Exception as e:
        logger.error("Initialization error: %s", e)
