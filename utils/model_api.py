"""
Model-agnostic API caller for OpenRouter integration.
Supports model selection by index and detailed reasoning retrieval.
Manages API configuration via environment variables and local files.
"""

import os
from typing import Any, List, Optional
from openai import OpenAI
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

        # Validate required fields
        if not self.api_key:
            raise ValueError("Missing required environment variable: MODEL_API_KEY")
        
        # Initialize OpenAI client
        # Clean up base_url if it ends with /api/v (some users might forget the 1)
        if self.base_url.endswith("/api/v"):
            self.base_url += "1"
            
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def call(
        self, 
        prompt: str, 
        model_index: int = 0, 
        reasoning: bool = True, 
        **kwargs: Any
    ) -> ChatCompletion:
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
        
        # Call the API
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
            **kwargs
        )
        
        return response

if __name__ == "__main__":
    try:
        api = ModelAPI()
        logger.info("Loaded %d models.", len(api.models))
        for i, m in enumerate(api.models):
            logger.info("[%d] %s", i, m)
    except Exception as e:
        logger.error("Initialization error: %s", e)
