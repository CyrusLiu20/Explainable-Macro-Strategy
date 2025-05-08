# Explainable Macro Strategy (Debate-Driven Strategy)

In recent years, algorithmic trading has become a dominant force in financial markets, with automated and often opaque black-box algorithms driving a significant portion of trading activity. While these models offer speed and efficiency, they also introduce complexity and a lack of transparency, raising concerns among regulators and internal stakeholders who increasingly demand clear, interpretable decision-making. At the same time, the rapid advancement of large language models (LLMs) has transformed the landscape of natural language processing, enabling near-human understanding of vast amounts of unstructured data. These models can analyze news, earnings calls, regulatory filings, and social media without the need for extensive data cleaning or feature engineering, pushing them closer to achieving generalist intelligence.
<br>
<br>
This project is motivated by the intersection of these two trends—algorithmic trading and LLM technology. It explores how LLMs can be harnessed to create an explainable, context-aware trading strategy. By leveraging LLMs, the strategy can extract insights from unstructured text data, provide clear explanations for its decisions, and enhance transparency in an otherwise opaque field. 
<br>
<br>
Data is mainly obtained from FRED (Federal Reserve Economic Data) for numerical indicators and Alpha Vantage for the news feed. The user will need API keys to scrape data using `CombinedScraper` and the scrapers in the `DataPipeline` folder.
<br>
<br>
LLM inference is powered by the DeepSeek API, with DeepSeek-Chat handling lower-priority tasks, such as filtering news headlines, and DeepSeek-Reasoner managing higher-priority tasks, including debating and generating trading signals, thanks to its powerful chain-of-thought feature. Access to these models requires purchasing credits, which can be costly at full price. Given the long computational times, both multi-processing and multi-threading are employed to maximize efficiency. However, for improved logging, it is recommended to set num_processes to 1 in the configuration YAML files.
<br>
<br>
**Note:** PromptCoder, developed by dhh1995, is designed to streamline and modularize the creation of LLM prompts.
<br>
**Note:** This codebase is optimized for macOS, and Windows users may encounter errors with `datetime.strptime` and the date format `"%Y%m%dT%H%M"`.


---

## Strategy Schematic Diagrams

To provide a clear understanding of the strategies implemented, the following schematic diagrams illustrate the decision-making workflows for both the News-Driven and Debate-Driven Strategies:

### News-Driven Strategy (Single-Agent Approach)

![News-Driven Strategy](Backtest/DiagramPics/News-Driven%20Strategy.png)

### Debate-Driven Strategy (Multi-Agent Approach)

![Debate-Driven Strategy](Backtest/DiagramPics/Debate-Driven%20Strategy.png)

---


# Start using this repository

Follow the steps below to set up your environment and run the data scraper:

### 1. Create and activate a Conda environment
```bash
conda create -n LLMStrategy python==3.9.21
conda activate LLMStrategy
```

### 2. Install dependencies
```bash
pip3 install -r requirement.txt
```
## DataPipeline
### 3. Set up API Keys and Run the Scraper

You need to register for free API keys from the following sources:

- [US Census Bureau](https://api.census.gov/data/key_signup.html)  
- [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/docs/api/api_key.html)  
- [Alpha Vantage](https://www.alphavantage.co/support/#api-key)  

Once you have your API keys, set them as environment variables:

#### macOS/Linux:
```bash
export CENSUS_API_KEY='YOUR_CENSUS_API_KEY'
export FRED_API_KEY='YOUR_FRED_API_KEY'
export ALPHAVANTAGE_API_KEY='YOUR_ALPHAVANTAGE_API_KEY'
```

#### Windows (Command Prompt):
```cmd
set CENSUS_API_KEY='YOUR_CENSUS_API_KEY'
set FRED_API_KEY='YOUR_FRED_API_KEY'
set ALPHAVANTAGE_API_KEY='YOUR_ALPHAVANTAGE_API_KEY'
```

Now, run the scraper:
```bash
python CombinedScraper.py --scrape --process
```


### 4. Install the PromptCoder Package
```bash
cd PromptCoder
pip install -e .
cd ..
```

## Backtesting

### 5.1 Configuring the Configuration Files

For the News-Driven Strategy, the configuration file is defined in `single_agent_config.yaml`, while the Debate-Driven Strategy uses `multi_agent_config.yaml`. It is recommended to keep the default settings for these files.

### 5.2 Running the Backtesting Process

If the data has already been filtered, it will be stored in `Backtest/AggregatedData/AggregatedNews.csv`. This significantly reduces processing time by bypassing the need for the LLM-powered `FilterAgent` to filter news headlines, allowing the system to directly import the pre-aggregated news data.

To run the News-Driven Strategy (Single-Agent Approach):

```bash
python BacktestEngine.py --config single_agent_config.yaml
```

To run the Debate-Driven Strategy (Multi-Agent Approach):

```bash
python BacktestEngine.py --config multi_agent_config.yaml
```

## Visualization

To visualize the backtesting results on an ETF (e.g., iShares 7-10 US Treasury bonds), save the price data CSV file at `DataPipeline/Data/Benchmark/IEF_price_data.csv`, then run:

```bash
python Visualization.py
```

---
