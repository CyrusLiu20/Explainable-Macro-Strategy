import ollama
import subprocess
import time
import pandas as pd
import re
import sys
import os
import textwrap
from typing import Dict, List
from rapidfuzz import process, fuzz
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

# Ensure Utilities is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *
from LLMAgent.BaseAgent import BaseAgent

class TradingAgent(BaseAgent):
    def __init__(self, asset: str, ticker: str, name: str = "TradingAgent", 
                 logger_name: str ="TradingAgent", model: str = "deepseek-r1:1.5b", 
                 style: str = "risk_neutral", risk_tolerance: str = "medium",
                 has_system_prompt: bool = False,
                 chat_history_path: str = "Results/chat_history.json"):
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
        self.has_system_prompt = has_system_prompt
        self.chat_history_path = chat_history_path

        # Define Risk Tolerance and Style of the Trader
        self.STYLE_PROMPT = STYLE_PROMPT
        self.SYSTEM_PROMPT = Collection(BACKGROUND_PROMPT, self.STYLE_PROMPT)

        # Format the system message
        system_prompt = format_prompt(
            self.SYSTEM_PROMPT,
            {"style": self.style, "risk_tolerance": self.risk_tolerance, "asset_name": self.asset, "ticker_name": self.ticker}
        )

        # Define the expected and example output
        self.EXAMPLE_PROMPT = Collection(DECISION_PROMPT, EXAMPLE_DECISION_PROMPT)
        self.example_prompt = format_prompt(self.EXAMPLE_PROMPT,{"asset": self.asset})

        # Discussion based strategy for multi-agent network
        self.ARGUMENT_PROMPT = Collection(ARGUMENT_PROMPT, EXAMPLE_ARGUMENT_PROMPT)
        self.REFLECTION_PROMPT = Collection(FINAL_REFLECTION_PROMPT, EXAMPLE_FINAL_REFLECTION_PROMPT)

        # Initialize the base agent class
        super().__init__(name=name, 
                         logger_name=logger_name, 
                         model=model, 
                         system_prompt=system_prompt, 
                         has_system_prompt=self.has_system_prompt,
                         chat_history_path=self.chat_history_path)

        self.log.info(f"Initialized TradingAgent for {self.asset} ({self.ticker}) with model {self.model}")


    def get_trading_decision(self, input_prompt: str) -> tuple[str, str]:
        """
        Gets a trading decision by interacting with the LLM and extracting the prediction and explanation.

        :param input_prompt: The user's input prompt.
        :return: A tuple containing the prediction and explanation.
        """
        # Get raw response from the base class
        input_prompt = f"{input_prompt}\n\n{self.example_prompt}"
        raw_response, status = self.response_chat(input_prompt=input_prompt)

        if status != "Success":
            return "Error", status

        # Extract prediction and explanation
        extracted_response = self.extract_prediction(raw_response)

        if extracted_response["prediction"] == "Unknown":
            self.log.warning("LLM returned 'Unknown' as the prediction. Check the response format.")

        return extracted_response["prediction"], extracted_response["explanation"]

    def argue(self, other_opinions):

        other_opinions_block = "\n\n".join([
                f"Agent {o['name']} predicted: \"{o['prediction']}\"\nExplanation: \"{o['explanation']}\""
                for o in other_opinions
            ])

        arguments = {
            "other_opinions_block": other_opinions_block,
            "risk_tolerance": self.risk_tolerance,
            "style": self.style,
        }

        argument_prompt = format_prompt(self.ARGUMENT_PROMPT,arguments)

        # Get raw response from the base class
        input_prompt = argument_prompt
        raw_response, status = self.response_chat(input_prompt=input_prompt)

        if status != "Success":
            return "Error", status

        # Extract prediction and explanation
        extracted_response = self.extract_argument(raw_response)

        if extracted_response["agreement"] == "Unknown":
            self.log.warning("LLM returned 'Unknown' as the prediction. Check the response format.")

        return extracted_response["agreement"], extracted_response["response"], extracted_response["prediction"]
    
    def reflection(self):

        reflect_dict = {"asset": self.asset}
        reflect_prompt = format_prompt(self.REFLECTION_PROMPT,reflect_dict)

        # Get raw response from the base class
        input_prompt = reflect_prompt
        raw_response, status = self.response_chat(input_prompt=input_prompt)

        if status != "Success":
            return "Error", status

        # Extract prediction and explanation
        extracted_response = self.extract_prediction(raw_response)

        if extracted_response["prediction"] == "Unknown":
            self.log.warning("LLM returned 'Unknown' as the prediction. Check the response format.")

        return extracted_response["prediction"], extracted_response["explanation"]

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

    def extract_argument(self, response_content: str) -> Dict[str, str]:

        # Extract prediction using regex
        agreement_match = re.search(r"(?:\*\*)?\s*Agreement:\s*([^\n*]+)", response_content, re.IGNORECASE)
        agreement = agreement_match.group(1).strip() if agreement_match else "Unknown"

        # Extract explanation using regex
        response_match = re.search(r"(?:\*\*)?\s*Response:\s*\n*(.*)", response_content, re.DOTALL | re.IGNORECASE)
        response = response_match.group(1).strip() if response_match else "No explanation found."

        # Extract prediction using regex
        prediction_match = re.search(r"(?:\*\*)?\s*Prediction:\s*([^\n*]+)", response_content, re.IGNORECASE)
        prediction = prediction_match.group(1).strip() if prediction_match else "Unknown"

        return {"agreement": agreement, "response": response, "prediction": prediction} 



class FilterAgent(BaseAgent):
    def __init__(self, name: str, asset: str, prompt_num_relevance: str = "1-2", logger_name: str = "FilterAgent", model: str = "deepseek-r1:1.5b", has_system_prompt: bool = False):
        """
        Superclass of LLMAgent for summarizing and selecting impactful news.

        :param name: Name of the agent.
        :param asset: The asset being analyzed (e.g., "US 10-year Treasury bonds").
        :param logger_name: Name of the logger (default: "SummaryAgent").
        :param model: The LLM model to use (default: "deepseek-r1:1.5b").
        """
        self.asset = asset
        self.prompt_num_relevance = prompt_num_relevance
        self.has_system_prompt = has_system_prompt

        # Define the expected output and system prompt
        self.EXPECTED_OUTPUT_PROMPT = EXAMPLE_SUMMARY_PROMPT
        self.SYSTEM_PROMPT = Collection(MACROECONOMIC_NEWS_SELECTION_PROMPT)

        # Format the system message
        system_prompt = format_prompt(
            self.SYSTEM_PROMPT,
            {"asset": self.asset, "prompt_num_relevance": self.prompt_num_relevance}
        )

        # Initialize the base class
        super().__init__(name=name, logger_name=logger_name, model=model, system_prompt=system_prompt, has_system_prompt=self.has_system_prompt)

        self.log.info(f"Initialized SummaryAgent for {self.asset} with model {self.model}")

    def filter_news(self, news_entries: str) -> pd.DataFrame:
        """
        Gets the most impactful news by interacting with the LLM and extracting the selected news.

        :param news_entries: A string containing news titles and summaries.
        :return: A DataFrame containing Date, Source, Title, and Summary.
        """
        # Format the input prompt
        self.INPUT_PROMPT = Collection(MACROECONOMIC_NEWS_PROMPT,
                                       self.EXPECTED_OUTPUT_PROMPT)
        input = {"news_entries": news_entries, "asset": self.asset, "prompt_num_relevance": self.prompt_num_relevance}
        input_prompt = format_prompt(self.INPUT_PROMPT, input)

        # Get raw response from the base class
        raw_response, status = self.response_chat(input_prompt)

        if status != "Success":
            self.log.error(f"Failed to get response from LLM: {status}")
            return pd.DataFrame(columns=["Date", "Source", "Title", "Summary"]), False

        # Extract selected titles from the response
        title_relevance_map = self._extract_titles(raw_response)
        # self.log.info(raw_response)
        # print("Extracted Relevant Titles")
        # self.log.info(title_relevance_map)

        # Extract full details from the original news entries
        df, status_flag = self._extract_news_details(news_entries, title_relevance_map)

        return df, status_flag


    def _extract_titles(self, raw_response: str) -> dict:
        """
        Extracts Titles and their corresponding Relevance from the raw LLM response.

        :param raw_response: The full text response from the LLM.
        :return: A dictionary {title: relevance}
        """
        matches = re.findall(
            r"(?:\*\*\[?Title\]?\*\*|\[?Title\]?:?)\s*\**(.*?)\**\s*\n?"  # Matches **[Title]**, [Title]:, **Title**
            r"(?:\*\*\[?Relevance\]?\*\*|\[?Relevance\]?:?|Relevance:)\s*\**(.*?)\**\s*(?:\n|---)",  # Matches relevance
            raw_response, 
            re.DOTALL
        )

        # Convert list of tuples into a dictionary {title: relevance}
        title_relevance_map = {title.strip(): relevance.strip() for title, relevance in matches}
        return title_relevance_map


    def _extract_news_details(self, news_entries: str, title_relevance_map: dict) -> pd.DataFrame:
        """
        Extracts Date, Title, Source, Summary, and Relevance from news_entries.

        :param news_entries: The full text of all news entries.
        :param title_relevance_map: Dictionary mapping titles to their relevance from the LLM response.
        :return: A DataFrame containing Date, Source, Title, Summary, and Relevance.
        """

        # If title_relevance_map is empty, return an empty DataFrame immediately
        if not title_relevance_map:
            self.log.warning("title_relevance_map is empty! Returning an empty DataFrame.")
            return pd.DataFrame(columns=["Date", "Source", "Title", "Summary", "Relevance"]), False


        pattern = re.compile(
            r"Date: \*\*(.*?)\*\*\n"
            r"Title: \*(.*?)\* \(Source: (.*?)\)\n"
            r"Summary: (.*?)(?=\nDate: |\Z)",  # Matches everything until the next "Date" or end of text
            re.DOTALL
        )

        matches = pattern.findall(news_entries)
        extracted_data = []

        for date, title, source, summary in matches:
            title = title.strip()

            # Find the best matching title from the LLM response
            match_result = process.extractOne(title, title_relevance_map.keys(), scorer=fuzz.token_sort_ratio) if title_relevance_map else None
            
            if match_result:
                best_match, score, _ = match_result
                if score >= 70:  # Consider titles with at least 80% similarity
                    extracted_data.append({
                        "Date": date.strip(),
                        "Source": source.strip(),
                        "Title": best_match,  # Use the best-matching extracted title
                        "Summary": summary.strip(),
                        "Relevance": title_relevance_map[best_match]  # Retrieve the corresponding relevance
                    })
            else:
                self.log.warning(f"No match found for news title: '{title}'")

        # df = pd.DataFrame(extracted_data)
        df = pd.DataFrame(extracted_data, columns=["Date", "Source", "Title", "Summary", "Relevance"])

        # Convert the Date column to pandas datetime format
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        # Log a warning if the number of extracted news items is different from expected
        status_flag = True
        if len(df) != len(title_relevance_map):
            if len(title_relevance_map) >= 2 and abs(len(df) - len(title_relevance_map)) == 1:
                self.log.warning(
                    f"Small mismatch in extracted news count, but proceeding. Expected {len(title_relevance_map)}, got {len(df)}."
                )
            else:
                status_flag = False  # Only set to False if the mismatch is significant
                self.log.warning(
                    f"Mismatch in extracted news count! Expected {len(title_relevance_map)}, but got {len(df)}."
                )


        return df, status_flag      
