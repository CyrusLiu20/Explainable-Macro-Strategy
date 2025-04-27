import textwrap
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock, NamedVariable, Collection
from Utilities.Logger import logger

import ast
import pandas as pd
import matplotlib.pyplot as plt

prompt_logger = logger(name="PromptLogger", log_file="Logs/backtest.log")

# Define the background of a quant macro trader
BACKGROUND_PROMPT = NamedBlock(
    name="Background",
    content="""
You are a quantitative macro trader developing an **Explainable LLM-based Trading Strategy**.  
Your goal is to predict the future price of **[{asset_name}]** (Ticker: **[{ticker_name}]**)  
by analyzing both **structured macroeconomic numerical data** and **unstructured financial news text data**.  

Your strategy must not only make accurate predictions but also provide clear explanations  
on how macroeconomic indicators and market sentiment contribute to the decision-making process.  
The model should highlight key factors driving its forecast, ensuring interpretability and transparency  
in the trading signals generated.
    """
)


# Define system message using procoder
STYLE_PROMPT = NamedBlock(
    name="Your Task as a Quant Trader",
    content=textwrap.dedent("""
        You are analyzing market data and need to make a decision based on the following criteria:
        - Trading Style: {style}
        - Risk Tolerance: {risk_tolerance}
    """)
)


# Define prompt for LLM to choose relevant news for trading US 10 Year Treasury bonds
MACROECONOMIC_NEWS_SELECTION_PROMPT = NamedBlock(
    name="Summarise Macroeconomic News for {asset}",
    content=textwrap.dedent("""
    Your task is to review a list of news articles each day and identify which headlines and summaries provide relevant information for a macro strategy 
    to trade {asset}. The selected news should be related to global economic conditions, government policies, or major market-moving 
    events that could influence {asset}. Your output will be fed into another Large Language Model to further infer trading decisions.

    The model should output:
    1. Select the top {prompt_num_relevance} most relevant news items in order of importance for {asset} (keeping the exact title and providing a relevance reason).
    2. Follow the exact format below (with keywords [Title]: **Title 1** and [Relevance]: **Relevance 1** so RegEx can extract).
    3. A brief explanation of why the news is relevant to trading {asset}.
    """)
)


# Recent Macroeconomic News Block
MACROECONOMIC_NEWS_PROMPT = NamedBlock(
    name="Recent Macroeconomic News",
    content=textwrap.dedent("""
    Below is a summary of recent macroeconomic news articles and key events  
    that may impact financial markets. Please select the top {prompt_num_relevance} most relevant news in order of importance {asset}

    {news_entries}
    """)
)




DECISION_PROMPT = NamedBlock(
    name="Decision (Forecast Price)",
    content=textwrap.dedent("""
    Based on the provided macroeconomic indicators and recent economic news, 
    predict the price of the future movement for [{asset}].

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
    """)
)


EXAMPLE_DECISION_PROMPT = NamedBlock(
    name="Expected Output Format",
    content=textwrap.dedent("""
    ```
    Prediction: [Trend]\n
    Explanation: [Explanation]
    ```
    """)
)


EXAMPLE_SUMMARY_PROMPT = NamedBlock(
    name="Example Expected Output Format (Select the top {prompt_num_relevance} most relevant news in order of importance for {asset}, keeping the exact title and providing a reason why it is relevant ([Relevance]). \
          Follow the exact format below (with keywords [Title]: **Title 1** and [Relevance]: **Relevance 1** so RegEx can extract) and include an overall summary.)",
    content="""
    ```
    [Title]: **Title 1**
    [Relevance]: **Relevance 1**

    [Title]: **Title 2**
    [Relevance]: **Relevance 2**

    **Overall Summary**\n
    YOUR OVERALL SUMMARY
    ```
    """
)


def format_macro_news(csv_file, filter_dates=None, chunk_size=10):
    # Read CSV file
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates()

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter by dates if filter_dates is provided
    if filter_dates:
        filter_dates = [pd.to_datetime(date).date() for date in filter_dates]
        df = df[df['Date'].dt.date.isin(filter_dates)]
    
    # Log the number of news selected
    num_news = len(df)
    if num_news > 0:
        prompt_logger.info(f"{num_news} news entries selected.")
    else:
        prompt_logger.warning("No news entries selected after applying filters.")
    
    # Process each row to extract relevant fields
    df.sort_values(by='Date', ascending=True, inplace=True)
    
    news_chunks = []
    chunk = []
    
    for _, row in df.iterrows():
        date = row['Date'].strftime('%Y-%m-%d %H:%M:%S')
        title = row['Title']
        summary = row['Summary']
        source = row['Source']
        
        # Format each news entry
        entry = f"Date: **{date}**\nTitle: *{title}* (Source: {source})\nSummary: {summary}\n"
        chunk.append(entry)
        
        # Append chunk if it reaches chunk_size
        if len(chunk) == chunk_size:
            news_chunks.append('\n\n'.join(chunk))
            chunk = []
    
    # Append any remaining news entries as the last chunk
    if chunk:
        news_chunks.append('\n\n'.join(chunk))
    
    return news_chunks, num_news


def format_macro_indicator(macro_csv, mapping_csv, current_date, last_periods=4):
    # Load macroeconomic data
    macro_df = pd.read_csv(macro_csv)
    mapping_df = pd.read_csv(mapping_csv)
    
    # Convert date columns to datetime
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    current_date = pd.to_datetime(current_date)

    # Map Series IDs to Renamed Series
    rename_dict = dict(zip(mapping_df['Series ID'], mapping_df['Renamed Series']))
    macro_df.rename(columns=rename_dict, inplace=True)
    
    # Ensure all columns except 'Date' are treated as strings before filling NaNs
    for col in macro_df.columns:
        if col != "Date":
            macro_df[col] = macro_df[col].astype(str)  # Convert to string to avoid dtype issues
    macro_df.fillna("N/A", inplace=True)

    # Infer frequency from the filename
    freq_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 90}
    inferred_freq = next((freq for freq in freq_days if freq in str(macro_csv)), "Unknown")
    
    # Apply delay to prevent data leakage
    delay_dict = dict(zip(mapping_df['Renamed Series'], mapping_df['Delay (Days)']))
    for col in macro_df.columns:
        if col in delay_dict:
            delay_days = delay_dict[col]
            published_cutoff = current_date - pd.Timedelta(days=delay_days)
            macro_df.loc[macro_df['Date'] > published_cutoff, col] = "Not Yet Published"

    # Present the latest data up to the current date
    macro_df = macro_df[macro_df['Date'] < current_date]

    # Compute period difference AFTER filtering for concise LLM prompt
    if inferred_freq in freq_days:
        period_days = freq_days[inferred_freq]
        macro_df['Period Diff'] = (current_date - macro_df['Date']).dt.days // period_days
        macro_df['Date'] = macro_df['Date'].dt.strftime('%Y-%m-%d') + macro_df['Period Diff'].apply(lambda x: f" (Past {x} {inferred_freq})")
    else:
        macro_df['Period Diff'] = "N/A"

    # Sort data in ascending order and keep only the last `last_periods` entries
    macro_df = macro_df.sort_values(by='Date', ascending=True).tail(last_periods)
    
    # Determine column widths for alignment
    col_widths = {col: max(len(str(col)), max(macro_df[col].astype(str).apply(len))) + 2 for col in macro_df.columns}
    
    # Construct formatted table
    header = " | ".join(col.ljust(col_widths[col]) for col in macro_df.columns)
    separator = "-" * len(header)
    rows = [" | ".join(str(row[col]).ljust(col_widths[col]) for col in macro_df.columns) for _, row in macro_df.iterrows()]
    
    # Final formatted output
    formatted_output = f"Here are the {inferred_freq} macroeconomic indicators and their values:\n{header}\n{separator}\n" + "\n".join(rows)

    return formatted_output



### Deprecated and will be deleted
def format_news_entries(df):
    return "\n".join(
        f"- **Date**: {row.date}  \n  **Title**: {row.title}  \n  **Summary**: {row.summary}"
        for _, row in df.iterrows()
    )





################### Depracated (Tutorial Use Only) ####################

### Deprecated and will be deleted
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


### Deprecated and will be deleted
# Define dataset dynamically using the generated CSV string
MACROECONOMIC_DATASET_PROMPT = NamedBlock(
    name="Macroeconomic Data",
    content=NamedVariable(
        refname="macro_data",
        name="Macroeconomic Indicators",
        content="{macroeconomic_dataframe_csv}"
    )
)


### Deprecated and will be deleted
# Combine all elements into a single prompt
LLMSTRATEGY_PROMPT = Collection(
    # BACKGROUND_PROMPT,
    # MACROECONOMIC_DESCRIPTION_PROMPT,
    MACROECONOMIC_DATASET_PROMPT,
    MACROECONOMIC_NEWS_PROMPT,
    DECISION_PROMPT,
    EXAMPLE_DECISION_PROMPT,
)