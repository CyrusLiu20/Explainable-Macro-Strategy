import sys
import os
import re
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, Collection

# Ensure Utilities is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *
from LLMAgent.MacroAgent import TradingAgent

lock = Lock()

class MultiAgentNetwork():
    def __init__(self, asset: str, ticker: str, name: list, logger_name: list, model: list, verbose_debate: bool,
                 style: list, risk_tolerance: list, has_system_prompt: list, chat_history_path: str):
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
        self.chat_history_path=chat_history_path

        self.verbose_debate = verbose_debate

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
                has_system_prompt=self.has_system_prompt[i],
                chat_history_path=self.chat_history_path,
            )
            self.trading_agents.append(agent)

        # Optionally initialize a logger here
        self.log = logger(name="MultiAgentNetwork", log_file=f"Logs/backtest.log")
        self.log.info(f"Initialized MultiAgentNetwork for {self.asset} ({self.ticker}) with {len(self.trading_agents)} agents.")



    def get_trading_decision(self, input_prompt: str, max_rounds: int = 5):
        """
        Manages the discussion between multiple agents in three phases.
        """
        # Step 1: Agents form initial opinions
        agent_opinions = self.constructive_speech(input_prompt)

        # Step 2: Agents communicate in rounds
        self.cross_examination(agent_opinions, max_rounds)

        # Step 3: Agents reflect and update final opinions
        final_opinions = self.reflection_phase()

        # self.log.info(final_opinions)
        return final_opinions

    def constructive_speech(self, input_prompt: str):
        """
        Agents form initial opinions based on the input prompt.
        """
        agent_opinions = {}
        if self.verbose_debate:
            print("\n[Step 1] Agents form initial opinions:\n")

        def get_opinion(agent):
            prediction, explanation = agent.get_trading_decision(input_prompt)
            return agent.name, prediction, explanation

        with ThreadPoolExecutor() as executor:
            future_to_agent = {executor.submit(get_opinion, agent): agent for agent in self.trading_agents}

            for future in as_completed(future_to_agent):
                agent_name, prediction, explanation = future.result()
                agent_opinions[agent_name] = {
                    "prediction": prediction,
                    "explanation": explanation
                }
                if self.verbose_debate:
                    print(f"{agent_name} predicts: {prediction}")
                    print(f"Explanation: {explanation}\n")

        return agent_opinions

    def cross_examination(self, agent_opinions, max_rounds: int):
        """
        Agents communicate in rounds where they argue and discuss all other agents' opinions
        collectively, until they all reach the same decision or the maximum number of rounds is reached.
        """
        if self.verbose_debate:
            print("\n[Step 2] Agents start discussion rounds:\n")

        for round_num in range(max_rounds):
            if self.verbose_debate:
                print(f"\n--- Round {round_num + 1} ---\n")

            new_opinions = {}
            decisions = {}

            def process_agent(agent):
                outputs = [f"\n{agent.name} is responding to all other agents...\n"]
                agent_prediction = None
                agent_explanation = None
                agent_decision = None

                # Aggregate all other agents' predictions and explanations
                other_views = []
                for other_name, opinion in agent_opinions.items():
                    if other_name != agent.name:
                        other_views.append({
                            "name": other_name,
                            "prediction": opinion["prediction"],
                            "explanation": opinion["explanation"]
                        })

                # # Call agent's reasoning method with all other opinions at once
                # with lock:
                agreement, response, prediction = agent.argue(
                    other_opinions=other_views
                )

                agent_prediction = prediction
                agent_explanation = response
                agent_decision = sentiment_to_decision(prediction)

                outputs.append(f"[{agent.name}]  Agreement: {agreement}")
                outputs.append(f"Response: {response}")
                outputs.append(f"Prediction: {prediction}")
                outputs.append(f"Decision: {agent_decision} \n")

                # Store updated opinions and decisions
                new_opinions[agent.name] = {
                    "prediction": agent_prediction,
                    "explanation": agent_explanation
                }
                decisions[agent.name] = agent_decision

                return '\n'.join(outputs)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_agent, agent) for agent in self.trading_agents]

                for future in as_completed(futures):
                    if self.verbose_debate:
                        print(future.result())

            # Check for consensus
            all_decisions = list(decisions.values())
            if len(set(all_decisions)) == 1:
                if self.verbose_debate:
                    print(f"\nAll agents have reached consensus decision: '{all_decisions[0]}'\n")
                break
            else:
                if self.verbose_debate:
                    print("\nNo consensus on decision yet. Continuing to next round...\n")

            agent_opinions = new_opinions  # Update for next round
        else:
            if self.verbose_debate:
                print("\nMaximum rounds reached. No consensus on decision.\n")


    def reflection_phase(self):
        """
        After the discussion rounds, each agent reflects and provides a final prediction and explanation.
        """
        if self.verbose_debate:
            print("\n[Step 3] Agents reflect and update their final views:\n")
        final_opinions = {}

        def agent_reflect(agent):
            if self.verbose_debate:
                print(f"{agent.name} is reflecting on the discussion...")
            final_prediction, final_explanation = agent.reflection()
            return agent.name, final_prediction, final_explanation

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(agent_reflect, agent) for agent in self.trading_agents]

            for future in as_completed(futures):
                agent_name, final_prediction, final_explanation = future.result()
                final_opinions[agent_name] = {
                    "prediction": final_prediction,
                    "explanation": final_explanation
                }
                if self.verbose_debate:
                    print(f"{agent_name}'s Final Prediction: {final_prediction}")
                    print(f"{agent_name}'s Final Explanation: {final_explanation}\n")

        return final_opinions
    

    # Save chat history for all agents
    def save_chat_history(self, date):
        for agent in self.trading_agents:
            agent.save_chat_history(date=date)