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

# Initialize Logger
log = logger(name="TradingAgent", log_file="Logs/trading_agent.log")


class TradingAgent:
    def __init__(self, name, asset, ticker, model="deepseek-r1:1.5b", style="mood", risk_tolerance="medium"):
        self.name = name
        self.asset = asset
        self.ticker = ticker
        self.model = model
        self.style = style
        self.risk_tolerance = risk_tolerance
        self.chat_history = []

        # Define system message using procoder
        self.STYLE_PROMPT = NamedBlock(
            name="SystemMessage",
            content="""
                You are analyzing market data and need to make a decision based on the following criteria:
                - Trading Style: {style}
                - Risk Tolerance: {risk_tolerance}
            """
        )
        self.SYSTEM_PROMPT = Collection(BACKGROUND_PROMPT, self.STYLE_PROMPT)

        # Format the system message
        self.system_prompt = format_prompt(
            self.SYSTEM_PROMPT,
            {"style": self.style, "risk_tolerance": self.risk_tolerance, "asset_name": self.asset, "ticker_name": self.ticker}
        )

        log.info(f"Initialized TradingAgent for {self.asset} ({self.ticker}) with model {self.model}")

    def start_ollama_server(self):
        """Start Ollama server in the background if not already running."""
        log.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give some time for the server to start
        log.info("Ollama server started successfully.")

    def response(self, input_prompt):
        """Handles chat interaction with the LLM, returning a trading decision and explanation."""

        # Update chat history
        self.chat_history.append({"role": "system", "content": self.system_prompt})
        self.chat_history.append({"role": "user", "content": input_prompt})

        # Query the model
        log.info(f"Sending request to {self.name}...")

        try:
            response = ollama.chat(model=self.model, messages=self.chat_history)

            # Check if response is empty (potential token limit issue)
            if not response or "message" not in response or "content" not in response["message"]:
                log.error("Invalid response format from LLM. Possible input token limit exceeded.")
                return "Error", "Invalid response format. Input may be too long."

            response_content = response["message"]["content"]

            # Store assistant's response
            self.chat_history.append({"role": "assistant", "content": response_content})

            # Extract prediction and explanation
            extracted_response = self.extract_prediction(response_content)

            if extracted_response["prediction"] == "Unknown":
                log.warning("LLM returned 'Unknown' as the prediction. Check the response format.")

            return extracted_response["prediction"], extracted_response["explanation"]

        except ollama.OllamaError as e:
            log.critical(f"Ollama model error: {e}. Ensure the model '{self.model}' is pulled and available.")
            return "Error", "Ollama model not found or not running."

        except ValueError as e:
            log.error(f"Input exceeded token limit: {e}")
            return "Error", "Input too long. Reduce the size of your query."

        except Exception as e:
            log.error(f"Unexpected error during LLM response processing: {e}")
            return "Error", "An unexpected error occurred." 


    def extract_prediction(self, response_content):
        """
        Extracts the prediction (e.g., Bullish, Bearish) and explanation from the LLM response.
        """

        # Extract prediction using regex
        prediction_match = re.search(r"(?:\*\*)?\s*Prediction:\s*([^\n*]+)", response_content, re.IGNORECASE)
        prediction = prediction_match.group(1).strip() if prediction_match else "Unknown"

        # Extract explanation using regex
        explanation_match = re.search(r"(?:\*\*)?\s*Explanation:\s*\n*(.*)", response_content, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

        extracted_response = {"prediction": prediction, "explanation": explanation}

        return extracted_response




if __name__ == "__main__":

    # Sample DataFrame with macroeconomic indicators
    data = {
        "Date": ["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"],
        "GDP": [23000, 23150, 23200, 23350, 23500, 23650],
        "Inflation Rate": [2.5, 2.6, 2.4, 2.3, 2.2, 2.1],
        "Unemployment Rate": [4.1, 4.0, 3.9, 3.8, 3.7, 3.6],
        "Urban CPI": [300.1, 302.2, 304.3, 306.5, 308.8, 311.0],
        "S&P500": [4600, 4650, 4700, 4750, 4800, 4850],
        "USD/YEN": [145, 144, 143, 142, 141, 140],
        "USD/EUR": [1.1, 1.09, 1.08, 1.07, 1.06, 1.05],
    }

    df = pd.DataFrame(data)

    # Convert DataFrame to CSV-like string
    macroeconomic_dataframe_csv = df.to_csv(index=False)

    # Example DataFrame with recent macroeconomic news
    df_news = pd.DataFrame({
        "date": ["2024-02-05", "2024-02-04"],
        "title": ["Fed Hints at Interest Rate Cut", "US Jobs Report Beats Expectations"],
        "summary": [
            "The Federal Reserve signaled a potential rate cut in the next quarter as inflation slows.",
            "The US added 250,000 jobs in January, surpassing expectations and lowering unemployment."
        ]
    })

    asset = "US Treasury Bond",
    ticker = "US10Y",
    news_entries = format_news_entries(df_news)

    input = {"asset_name": asset, 
            "ticker_name": ticker,
            "news_entries": news_entries,
            "macroeconomic_dataframe_csv": macroeconomic_dataframe_csv
            }


    formatted_prompt = format_prompt(LLMSTRATEGY_PROMPT, input)

    agent = TradingAgent(name="MoodAgent", asset=asset, ticker=ticker, style="mood", risk_tolerance="high")
    agent.start_ollama_server()
    decision = agent.response(formatted_prompt)
    print(f"\n\n\n\n{agent.name} Trading Decision: {decision}")
