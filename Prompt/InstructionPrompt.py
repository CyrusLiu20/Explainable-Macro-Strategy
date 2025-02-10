from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, NamedVariable, Collection



# Define the background of a quant macro trader
BACKGROUND_PROMPT = NamedBlock(
    name="Background",
    content="""
        You are a quantitative macro trader developing an **Explainable LLM-based Trading Strategy**.  
        Your goal is to predict the future trend of **[{asset_name}]** (Ticker: **[{ticker_name}]**)  
        by analyzing both **structured macroeconomic numerical data** and **unstructured financial news text data**.  

        Your strategy must not only make accurate predictions but also provide clear explanations  
        on how macroeconomic indicators and market sentiment contribute to the decision-making process.  
        The model should highlight key factors driving its forecast, ensuring interpretability and transparency  
        in the trading signals generated.
    """
)


# Define macroeconomic indicators description with units
MACROECONOMIC_DESCRIPTION_PROMPT = NamedBlock(
    name="Macroeconomic Indicator Of Past Six Months",
    content="""
        These macroeconomic indicators include:
        - **US GDP** (Billion USD): Measures the total economic output of the United States.
        - **US Inflation Rate** (%): Represents the year-over-year percentage change in consumer prices.
        - **US Unemployment Rate** (%): The percentage of the labor force that is unemployed.
        - **US Urban CPI** (Index, 1982-84=100): Tracks changes in the cost of goods and services in urban areas.
        - **S&P500 Index** (Index Points): Measures the performance of 500 large publicly traded companies in the US.
        - **USD/YEN Exchange Rate** (JPY per USD): The value of one US dollar in Japanese yen.
        - **USD/EUR Exchange Rate** (EUR per USD): The value of one US dollar in euros.
    """
)

# Define dataset dynamically using the generated CSV string
MACROECONOMIC_DATASET_PROMPT = NamedBlock(
    name="Macroeconomic Data",
    content=NamedVariable(
        refname="macro_data",
        name="Macroeconomic Indicators",
        content="{macroeconomic_dataframe_csv}"
    )
)

# Recent Macroeconomic News Block
MACROECONOMIC_NEWS_PROMPT = NamedBlock(
    name="Recent Macroeconomic News",
    content="""
        Below is a summary of recent macroeconomic news articles and key events  
        that may impact financial markets.  

        {news_entries}
    """
)


from procoder.prompt import NamedBlock

DECISION_PROMPT = NamedBlock(
    name="Decision (Forecast Trend)",
    content="""
        Based on the provided macroeconomic indicators and recent economic news, 
        predict the overall trend of the next half-year price movement for [{asset_name}].

        Only output one of the following options:
        - Strongly Bullish
        - Bullish
        - Slightly Bullish
        - Flat
        - Fluctuating
        - Slightly Bearish
        - Bearish
        - Strongly Bearish

        Provide an explanation for your prediction.
    """
)


EXAMPLE_DECISION_PROMPT = NamedBlock(
    name="Expected Output Format",
    content="""
        ```
        Prediction: [Bullish]
        Explanation: The US GDP growth remains strong, inflation is declining, 
        and the Federal Reserve has signaled potential interest rate cuts. 
        This is likely to create a favorable market environment.
        ```
    """
)


# Combine all elements into a single prompt
LLMSTRATEGY_PROMPT = Collection(
    BACKGROUND_PROMPT,
    MACROECONOMIC_DESCRIPTION_PROMPT,
    MACROECONOMIC_DATASET_PROMPT,
    MACROECONOMIC_NEWS_PROMPT,
    DECISION_PROMPT,
    EXAMPLE_DECISION_PROMPT,
)


def format_news_entries(df):
    return "\n".join(
        f"- **Date**: {row.date}  \n  **Title**: {row.title}  \n  **Summary**: {row.summary}"
        for _, row in df.iterrows()
    )