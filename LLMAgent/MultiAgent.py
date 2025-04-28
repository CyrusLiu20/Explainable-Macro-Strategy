import sys
import os

import re
from typing import Dict
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

# Ensure Utilities is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *
from LLMAgent.MacroAgent import TradingAgent



class MultiAgentNetwork():
    def __init__(self, asset: str, ticker: str, name: list, logger_name: list, model: list, 
                 style: list, risk_tolerance: list, has_system_prompt: list):
        """
        A network managing multiple TradingAgents.

        :param name: List of agent names.
        :param asset: The asset being traded.
        :param ticker: The ticker symbol of the asset.
        :param model: List of LLM models to use.
        :param style: List of trading styles (e.g., "mood", "momentum").
        :param risk_tolerance: List of risk tolerance levels (e.g., "low", "high").
        :param has_system_prompt: List indicating if each agent uses a system prompt.
        """
        self.asset = asset
        self.ticker = ticker
        self.name = name
        self.logger_name = logger_name
        self.model = model
        self.style = style
        self.risk_tolerance = risk_tolerance
        self.has_system_prompt = has_system_prompt

        # Initialize a list of trading agents
        self.trading_agents = []
        for i in range(len(name)):
            agent = TradingAgent(
                asset=self.asset,
                ticker=self.ticker,
                name=self.name[i],
                logger_name=self.logger_name[i],
                model=self.model[i],
                style=self.style[i],
                risk_tolerance=self.risk_tolerance[i],
                has_system_prompt=self.has_system_prompt[i]
            )
            self.trading_agents.append(agent)

        # Optionally initialize a logger here
        self.log = logger(name="MultiAgentNetwork", log_file=f"Logs/backtest.log")
        self.log.info(f"Initialized MultiAgentNetwork for {self.asset} ({self.ticker}) with {len(self.trading_agents)} agents.")



    def get_trading_decision(self, input_prompt: str, num_rounds: int = 1):
        """
        Manages the discussion between multiple agents in three phases.
        """
        # Step 1: Agents form initial opinions
        agent_opinions = self.constructive_speech(input_prompt)

        # Step 2: Agents communicate in rounds
        self.cross_examination(agent_opinions, num_rounds)

        # Step 3: Agents reflect and update final opinions
        final_opinions = self.reflection_phase()

        self.log.info(final_opinions)
        return final_opinions

    def constructive_speech(self, input_prompt: str):
        """
        Agents form initial opinions based on the input prompt.
        """
        agent_opinions = {}
        print("\n[Step 1] Agents form initial opinions:\n")

        for agent in self.trading_agents:
            prediction, explanation = agent.get_trading_decision(input_prompt)
            agent_opinions[agent.name] = {
                "prediction": prediction,
                "explanation": explanation
            }
            print(f"🔹 {agent.name} predicts: {prediction}")
            print(f"🔹 Explanation: {explanation}\n")

        return agent_opinions

    def cross_examination(self, agent_opinions, num_rounds: int):
        """
        Agents communicate in rounds where they argue and discuss each other's opinions.
        """
        print("\n[Step 2] Agents start discussion rounds:\n")

        for round_num in range(0, num_rounds):
            print(f"\n--- Round {round_num+1} ---\n")

            for agent in self.trading_agents:
                print(f"\n {agent.name} is responding to other agents...\n")

                for other_agent_name, other_opinion in agent_opinions.items():
                    if other_agent_name == agent.name:
                        continue  # Skip responding to own opinion

                    print(f"  ➡️ Responding to {other_agent_name}'s view:")
                    print(f"    - Prediction: {other_opinion['prediction']}")
                    print(f"    - Explanation: {other_opinion['explanation']}")

                    agreement, response = agent.argue(
                        other_agent_name=other_agent_name,
                        other_prediction=other_opinion["prediction"],
                        other_explanation=other_opinion["explanation"]
                    )

                    print(f"[{agent.name}]  Agreement: {agreement} | Response: {response}\n")

    def reflection_phase(self):
        """
        After the discussion rounds, each agent reflects and provides a final prediction and explanation.
        """
        print("\n[Step 3] Agents reflect and update their final views:\n")
        final_opinions = {}

        for agent in self.trading_agents:
            print(f"🔄 {agent.name} is reflecting on the discussion...")

            final_prediction, final_explanation = agent.reflection()
            final_opinions[agent.name] = {
                "prediction": final_prediction,
                "explanation": final_explanation
            }

            print(f"✅ {agent.name}'s Final Prediction: {final_prediction}")
            print(f"✅ {agent.name}'s Final Explanation: {final_explanation}\n")

        return final_opinions
