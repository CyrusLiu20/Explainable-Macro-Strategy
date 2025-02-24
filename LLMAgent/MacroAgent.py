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
from LLMAgent.BaseAgent import BaseAgent

import re
from typing import Dict
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

class TradingAgent(BaseAgent):
    def __init__(self, asset: str, ticker: str, name: str = "TradingAgent", 
                 logger_name: str ="TradingAgent", model: str = "deepseek-r1:1.5b", 
                 style: str = "risk_neutral", risk_tolerance: str = "medium"):
        """
        Subclass of LLMAgent for trading-specific functionality.

        :param name: Name of the agent.
        :param asset: The asset being traded.
        :param ticker: The ticker symbol of the asset.
        :param model: The LLM model to use (default: "deepseek-r1:1.5b").
        :param style: The trading style (e.g., "mood").
        :param risk_tolerance: The risk tolerance level (e.g., "medium").
        """
        self.asset = asset
        self.ticker = ticker
        self.style = style
        self.risk_tolerance = risk_tolerance

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
        system_prompt = format_prompt(
            self.SYSTEM_PROMPT,
            {"style": self.style, "risk_tolerance": self.risk_tolerance, "asset_name": self.asset, "ticker_name": self.ticker}
        )

        # Initialize the base class
        super().__init__(name=name, logger_name=logger_name, model=model, system_prompt=system_prompt)

        self.log.info(f"Initialized TradingAgent for {self.asset} ({self.ticker}) with model {self.model}")

    def extract_prediction(self, response_content: str) -> Dict[str, str]:
        """
        Extracts the prediction (e.g., Bullish, Bearish) and explanation from the LLM response.

        :param response_content: The raw response content from the LLM.
        :return: A dictionary containing the prediction and explanation.
        """
        # Extract prediction using regex
        prediction_match = re.search(r"(?:\*\*)?\s*Prediction:\s*([^\n*]+)", response_content, re.IGNORECASE)
        prediction = prediction_match.group(1).strip() if prediction_match else "Unknown"

        # Extract explanation using regex
        explanation_match = re.search(r"(?:\*\*)?\s*Explanation:\s*\n*(.*)", response_content, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

        return {"prediction": prediction, "explanation": explanation}

    def get_trading_decision(self, input_prompt: str) -> tuple[str, str]:
        """
        Gets a trading decision by interacting with the LLM and extracting the prediction and explanation.

        :param input_prompt: The user's input prompt.
        :return: A tuple containing the prediction and explanation.
        """
        # Get raw response from the base class
        raw_response, status = self.response(input_prompt)

        if status != "Success":
            return "Error", status

        # Extract prediction and explanation
        extracted_response = self.extract_prediction(raw_response)

        if extracted_response["prediction"] == "Unknown":
            self.loglog.warning("LLM returned 'Unknown' as the prediction. Check the response format.")

        return extracted_response["prediction"], extracted_response["explanation"]
    





class SummaryAgent(BaseAgent):
    def __init__(self, name: str, asset: str, logger_name: str = "SummaryAgent", model: str = "deepseek-r1:1.5b", has_system_prompt: bool = False):
        """
        Superclass of LLMAgent for summarizing and selecting impactful news.

        :param name: Name of the agent.
        :param asset: The asset being analyzed (e.g., "US 10-year Treasury bonds").
        :param logger_name: Name of the logger (default: "SummaryAgent").
        :param model: The LLM model to use (default: "deepseek-r1:1.5b").
        """
        self.asset = asset
        self.has_system_prompt = has_system_prompt

        # Define the expected output format using NamedBlock
        self.EXPECTED_OUTPUT_PROMPT = EXAMPLE_SUMMARY_PROMPT

        # Combine the system prompt with the expected output format
        self.SYSTEM_PROMPT = Collection(
            MACROECONOMIC_NEWS_SELECTION_PROMPT
        )

        # Format the system message
        system_prompt = format_prompt(
            self.SYSTEM_PROMPT,
            {"asset": self.asset}
        )

        # Initialize the base class
        super().__init__(name=name, logger_name=logger_name, model=model, system_prompt=system_prompt)

        self.log.info(f"Initialized SummaryAgent for {self.asset} with model {self.model}")

    def get_impactful_news(self, news_entries: str) -> Dict[str, list[Dict[str, str]]]:
        """
        Gets the most impactful news by interacting with the LLM and extracting the selected news.

        :param news_entries: A string containing news titles and summaries.
        :return: A dictionary containing two keys:
                 - "selected_news": A list of dictionaries containing the selected news titles and summaries.
                 - "summary": A string containing an organized summary of the selected news.
        """
        # Format the input prompt
        self.INPUT_PROMPT = Collection(MACROECONOMIC_NEWS_PROMPT,
                                       self.EXPECTED_OUTPUT_PROMPT)
        input = {"news_entries": news_entries, "asset": self.asset}
        input_prompt = format_prompt(self.INPUT_PROMPT, input)

        # Get raw response from the base class
        raw_response, status = self.response(input_prompt, has_system_prompt=self.has_system_prompt)

        if status != "Success":
            self.log.error(f"Failed to get response from LLM: {status}")
            return {"selected_news": [], "summary": ""}

        self.log.debug(raw_response)
        return raw_response

        # # Parse the raw response into the expected format
        # selected_news = self._parse_selected_news(raw_response)
        # summary = self._parse_summary(raw_response)

        # return {"selected_news": selected_news, "summary": summary}

    def _parse_selected_news(self, raw_response: str) -> list[Dict[str, str]]:
        """
        Parses the selected news from the raw response.

        :param raw_response: The raw response from the LLM.
        :return: A list of dictionaries containing the selected news titles and summaries.
        """
        selected_news = []
        # Use regex to extract news entries
        news_matches = re.findall(
            r"Date: \*\*(.*?)\*\*\nTitle: \*(.*?)\* \(Source: (.*?)\)\nSummary: (.*?)\n(?=\n|$)",
            raw_response,
            re.DOTALL
        )

        for match in news_matches:
            date, title, source, summary = match
            selected_news.append({
                "date": date.strip(),
                "title": title.strip(),
                "source": source.strip(),
                "summary": summary.strip()
            })

        return selected_news

    def _parse_summary(self, raw_response: str) -> str:
        """
        Parses the organized summary from the raw response.

        :param raw_response: The raw response from the LLM.
        :return: A string containing the organized summary.
        """
        # Use regex to extract the summary
        summary_match = re.search(
            r"Summary: (.*?)(?=\n\n|$)",
            raw_response,
            re.DOTALL
        )
        return summary_match.group(1).strip() if summary_match else ""













