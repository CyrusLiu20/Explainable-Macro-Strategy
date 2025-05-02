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

        self.log.info(final_opinions)
        return final_opinions

    def constructive_speech(self, input_prompt: str):
        """
        Agents form initial opinions based on the input prompt.
        """
        agent_opinions = {}
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
                print(f"{agent_name} predicts: {prediction}")
                print(f"Explanation: {explanation}\n")

        return agent_opinions

    def cross_examination(self, agent_opinions, max_rounds: int):
        """
        Agents communicate in rounds where they argue and discuss each other's opinions
        until they all reach the same decision, or the maximum number of rounds is reached.
        """
        print("\n[Step 2] Agents start discussion rounds:\n")

        for round_num in range(max_rounds):
            print(f"\n--- Round {round_num + 1} ---\n")

            new_opinions = {}
            decisions = {}

            def process_agent(agent):
                outputs = [f"\n{agent.name} is responding to other agents...\n"]
                agent_prediction = None
                agent_explanation = None
                agent_decision = None

                for other_agent_name, other_opinion in agent_opinions.items():
                    if other_agent_name == agent.name:
                        continue  # Skip self

                    outputs.append(f"  Responding to {other_agent_name}'s view:")
                    outputs.append(f"    - Prediction: {other_opinion['prediction']}")
                    outputs.append(f"    - Explanation: {other_opinion['explanation']}")

                    with lock:
                        agreement, response, prediction = agent.argue(
                            other_agent_name=other_agent_name,
                            other_prediction=other_opinion["prediction"],
                            other_explanation=other_opinion["explanation"]
                        )

                    agent_prediction = prediction
                    agent_explanation = response
                    agent_decision = sentiment_to_decision(prediction)

                    outputs.append(
                        f"[{agent.name}]  Agreement: {agreement} | "
                        f"Response: {response} | Prediction: {prediction} | "
                        f"Decision: {agent_decision} \n"
                    )

                # Store final values for this agent
                new_opinions[agent.name] = {
                    "prediction": agent_prediction,
                    "explanation": agent_explanation
                }
                decisions[agent.name] = agent_decision

                return '\n'.join(outputs)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_agent, agent) for agent in self.trading_agents]

                for future in as_completed(futures):
                    print(future.result())

            # Check for consensus in decision
            all_decisions = list(decisions.values())
            if len(set(all_decisions)) == 1:
                print(f"\nAll agents have reached consensus decision: '{all_decisions[0]}'\n")
                break
            else:
                print(f"\nNo consensus on decision yet. Continuing to next round...\n")

            agent_opinions = new_opinions  # Update opinions for next round
        else:
            print("\nMaximum rounds reached. No consensus on decision.\n")


    def reflection_phase(self):
        """
        After the discussion rounds, each agent reflects and provides a final prediction and explanation.
        """
        print("\n[Step 3] Agents reflect and update their final views:\n")
        final_opinions = {}

        def agent_reflect(agent):
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

                print(f"{agent_name}'s Final Prediction: {final_prediction}")
                print(f"{agent_name}'s Final Explanation: {final_explanation}\n")

        return final_opinions