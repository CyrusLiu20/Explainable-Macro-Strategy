import ollama
import subprocess
import time
import pandas as pd
import requests
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
    def __init__(self, name: str, logger_name: str = "base_agent", model: str = "deepseek-r1:1.5b", system_prompt: str = ""):
        """
        Base class for an LLM-based agent.

        :param name: Name of the agent.
        :param logger_name: Name of the logger (default: "Agent").
        :param model: The LLM model to use (default: "deepseek-r1:1.5b").
        :param system_prompt: The system prompt to initialize the agent with.
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.chat_history = []
        self.log = logger(name=logger_name, log_file=f"Logs/{logger_name}.log")

        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        }

        self.log.info(f"Initialized LLMAgent '{self.name}' with model {self.model}")

    def _prepare_prompt(self, input_prompt: str, has_system_prompt: bool) -> tuple[list[dict[str, str]], str]:
        """
        Prepares the chat history and input prompt based on whether the model supports system prompts.

        :param input_prompt: The user's input prompt.
        :param has_system_prompt: Whether the model supports system prompts.
        :return: A tuple containing the updated chat history and the final input prompt.
        """
        if has_system_prompt:
            # If the model supports system prompts, add the system prompt to the chat history as a system message
            self.chat_history.append({"role": "system", "content": self.system_prompt})
            final_input_prompt = input_prompt
        else:
            # If the model does not support system prompts, add the system prompt as a user message
            self.chat_history.append({"role": "user", "content": self.system_prompt})
            final_input_prompt = input_prompt

        # # Add the user's input prompt to the chat history
        # self.log.info(self.system_prompt)
        # self.log.info(input_prompt)

        self.chat_history.append({"role": "user", "content": final_input_prompt})

        return self.chat_history, final_input_prompt

    def start_ollama_server(self):
        """Start Ollama server in the background if not already running."""
        self.log.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give some time for the server to start
        self.log.info("Ollama server started successfully.")

    def response(self, input_prompt: str, has_system_prompt: bool) -> tuple[str, str]:
        """
        Handles chat interaction with the LLM, returning the raw response.

        :param input_prompt: The user's input prompt.
        :param has_system_prompt: Whether the model supports system prompts.
        :return: A tuple containing the raw response and a status message.
        """
        # Prepare the chat history and input prompt based on model capabilities
        chat_history, final_input_prompt = self._prepare_prompt(input_prompt, has_system_prompt)

        # Query the model
        self.log.info(f"Sending request to {self.name}...")
        # self.log.info(chat_history)

        try:
            response = ollama.chat(model=self.model, messages=chat_history)

            # Check if response is empty (potential token limit issue)
            if not response or "message" not in response or "content" not in response["message"]:
                self.log.error("Invalid response format from LLM. Possible input token limit exceeded.")
                return "", "Invalid response format. Input may be too long."

            response_content = response["message"]["content"]

            # Store assistant's response
            self.chat_history.append({"role": "assistant", "content": response_content})

            return response_content, "Success"

        except ValueError as e:
            self.log.error(f"Input exceeded token limit: {e}")
            return "", "Input too long. Reduce the size of your query."

        except Exception as e:
            self.log.error(f"Unexpected error during LLM response processing: {e}")
            return "", "An unexpected error occurred."


    def response_chat(self, input_prompt: str, has_system_prompt: bool) -> tuple[str, str]:
        """
        Handles chat interaction with the DeepSeek API, returning the raw response.

        :param input_prompt: The user's input prompt.
        :param has_system_prompt: Whether the model supports system prompts.
        :return: A tuple containing the raw response and a status message.
        """
        # Prepare the chat history and input prompt based on model capabilities
        chat_history, final_input_prompt = self._prepare_prompt(input_prompt, has_system_prompt)

        # Prepare messages in DeepSeek format
        messages = chat_history  # Assuming chat_history is already in [{"role": ..., "content": ...}] format

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,  # adjust if you want
        }

        self.log.info(f"Sending request to {self.name}...")

        try:
            response = requests.post(self.url, headers=self.headers, json=payload)

            if response.status_code != 200:
                self.log.error(f"API Error {response.status_code}: {response.text}")
                return "", f"API Error {response.status_code}"

            response_json = response.json()

            if "choices" not in response_json or not response_json["choices"]:
                self.log.error("Invalid response format from DeepSeek API.")
                return "", "Invalid response format."

            response_content = response_json["choices"][0]["message"]["content"]

            # Store assistant's response
            self.chat_history.append({"role": "assistant", "content": response_content})

            return response_content, "Success"

        except requests.exceptions.RequestException as e:
            self.log.error(f"Request error: {e}")
            return "", "Network error or API unreachable."

        except Exception as e:
            self.log.error(f"Unexpected error during LLM response processing: {e}")
            return "", "An unexpected error occurred."
