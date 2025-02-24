import ollama
import subprocess
import time
import pandas as pd
import re
import sys
import os
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

# Ensure Utilities is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *

class BaseAgent:
    def __init__(self, name: str, logger_name: str ="Agent", model: str = "deepseek-r1:1.5b", system_prompt: str = ""):
        """
        Base class for an LLM-based agent.

        :param name: Name of the agent.
        :param model: The LLM model to use (default: "deepseek-r1:1.5b").
        :param system_prompt: The system prompt to initialize the agent with.
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.chat_history = []
        self.log = logger(name="Agent", log_file=f"Logs/{logger_name}.log")

        self.log.info(f"Initialized LLMAgent '{self.name}' with model {self.model}")

    def start_ollama_server(self):
        """Start Ollama server in the background if not already running."""
        self.log.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give some time for the server to start
        self.log.info("Ollama server started successfully.")

    def response(self, input_prompt: str) -> tuple[str, str]:
        """
        Handles chat interaction with the LLM, returning the raw response.

        :param input_prompt: The user's input prompt.
        :return: A tuple containing the raw response and a status message.
        """
        # Update chat history
        self.chat_history.append({"role": "system", "content": self.system_prompt})
        self.chat_history.append({"role": "user", "content": input_prompt})

        # Query the model
        self.log.info(f"Sending request to {self.name}...")

        try:
            response = ollama.chat(model=self.model, messages=self.chat_history)

            # Check if response is empty (potential token limit issue)
            if not response or "message" not in response or "content" not in response["message"]:
                self.log.error("Invalid response format from LLM. Possible input token limit exceeded.")
                return "", "Invalid response format. Input may be too long."

            response_content = response["message"]["content"]

            # Store assistant's response
            self.chat_history.append({"role": "assistant", "content": response_content})

            return response_content, "Success"

        except ollama.OllamaError as e:
            self.log.critical(f"Ollama model error: {e}. Ensure the model '{self.model}' is pulled and available.")
            return "", "Ollama model not found or not running."

        except ValueError as e:
            self.log.error(f"Input exceeded token limit: {e}")
            return "", "Input too long. Reduce the size of your query."

        except Exception as e:
            self.log.error(f"Unexpected error during LLM response processing: {e}")
            return "", "An unexpected error occurred."