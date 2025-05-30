import ollama
import subprocess
import time
import pandas as pd
import requests
import json
import sys
import os

from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

# Ensure Utilities is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *

class BaseAgent:
    def __init__(self, name: str, logger_name: str = "base_agent", 
                       model: str = "deepseek-r1:1.5b", system_prompt: str = "",
                       has_system_prompt: bool = False, chat_history_path="Results/chat_history.json"):
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
        self.has_system_prompt = has_system_prompt
        self.chat_history = []
        self.log = logger(name=logger_name, log_file=f"Logs/{logger_name}.log")

        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        }

        # Append system message to the chat history
        self._append_chat_history("system" if self.has_system_prompt else "user", self.system_prompt)

        self.log.info(f"Initialized LLMAgent '{self.name}' with model {self.model}")

    def response_chat(self, input_prompt: str) -> tuple[str, str]:
        """
        Handles chat interaction with the DeepSeek API, returning the raw response.

        :param input_prompt: The user's input prompt.
        :param has_system_prompt: Whether the model supports system prompts.
        :return: A tuple containing the raw response and a status message.
        """
        # Prepares the chat history
        self._append_chat_history(role="user",content=input_prompt)
        chat_history = self.chat_history

        # Prepare messages in DeepSeek format
        messages = chat_history
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "use_web_search": False, # Prevents data leakage
        }

        self.log.info(f"Sending request to {self.name}...")

        # self._debug_messages(messages)

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
            self._append_chat_history(role="assistant",content=response_content)

            return response_content, "Success"

        except requests.exceptions.RequestException as e:
            self.log.error(f"Request error: {e}")
            return "", "Network error or API unreachable."

        except Exception as e:
            self.log.error(f"Unexpected error during LLM response processing: {e}")
            return "", "An unexpected error occurred."

    # Appends a message to the chat history.
    def _append_chat_history(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    # Appends this agent's chat history to its file without overwriting existing data.
    def save_chat_history(self, date: str):

        # Create a new directory based on the date
        date = date.strftime("%Y-%m-%d")
        base_filename = os.path.basename(self.chat_history_path)
        directory_path = os.path.dirname(self.chat_history_path)
        new_directory_path = os.path.join(directory_path, date) 
        self.chat_history_path = os.path.join(new_directory_path, base_filename)
        os.makedirs(os.path.dirname(self.chat_history_path), exist_ok=True)

        # Load existing chat history file or create an empty structure
        if os.path.exists(self.chat_history_path):
            with open(self.chat_history_path, "r") as f:
                existing_history = json.load(f)
        else:
            existing_history = {}

        # Check if the date exists, then check agent's history for that date
        if date in existing_history:
            existing_history[date][self.name] = self.chat_history
        else:
            existing_history[date] = {self.name: self.chat_history}

        with open(self.chat_history_path, "w") as f:
            json.dump(existing_history, f, indent=2)

    # Prints out the key and the first few characters of each message for debugging purposes
    def _debug_messages(self, messages):

        for message in messages:
            key = message.get('role', 'Unknown key')
            content = message.get('content', '')
            preview = content[:200]
            print(f"Role: {key}\nContent (Preview): {preview}\n")


    # Deprecated: start Ollama server in the background
    def start_ollama_server(self):
        """Start Ollama server in the background if not already running."""
        self.log.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give some time for the server to start
        self.log.info("Ollama server started successfully.")