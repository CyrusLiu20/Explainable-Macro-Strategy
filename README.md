# LLMStrategy Data Pipeline

Presently, the data pipeline is almost fully developed, whereas the backtesting framework and LLM agent communications have just started, but stay tuned for future updates. Raw data can now be scraped, processed and fed into LLM agents in the backtest framework, providing trend predictions for the underlying asset.
<br>
<br>
Data is mainly obtained from FRED (Federal Reserve Economic Data) for numerical indicators and Alpha Vantage for the news feed. The user will need API keys to scrape data using `CombinedScraper` and the scrapers in the `DataPipeline` folder. Due to the free tier API request limit (25 requests per day for Alpha Vantage), already scraped data will also be provided and stored in cloud storage to save time.
<br>
<br>
LLM inferencing is powered by Ollama, so feel free to explore Ollama for a variety of available models. Due to the high computational demands, users may need to utilize a free GPU on Google Colab and set up the environment using the `ColabTerminal.ipynb` file.
<br>
<br>
Note: PromtCoder is developed by dhh1995 to streamline and modularise the creation of LLM Prompts.
<br>
<br>
Stay tuned for future updates!

---

# Developer Tutorials (Getting Started)

- **Using the API for Data**: In `DataPipeLine/API Tutorial.ipynb`, you'll find the basic coding structure along with example usage on how to fetch data from the US Census Bureau, FRED Economic Data, and the Alpha Vantage Portal.

- **Using Procoder for Formatting LLM Prompts**: In `Prompt/Prompt Tutorial.ipynb`, you'll find the essential coding structure and example usage for dynamically formatting LLM prompts.

- **Current Trends in LLMs for Quantitative Trading**: For an updated understanding of the latest advancements in using LLMs to generate explainable alpha signals, please refer to the ReferencePapers folder. One interesting paper that the reader is advised to read is 'Large Language Model Agent in Financial Trading: A Survey' (for the web version, URL: https://doi.org/10.48550/arXiv.2408.06361).

---


# Start using this repository

Follow the steps below to set up your environment and run the scraper:

### 1. Create and activate a Conda environment
```bash
conda create -n LLMStrategy python==3.9.21
conda activate LLMStrategy
```

### 2. Install dependencies
```bash
pip3 install -r requirement.txt
```

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

### 5.1 (Google Colab Only) Download LLM Models and Run This Repository in Colab

Running this repository in Google Colab can be a bit tricky. To begin, upload the `ColabTerminal.ipynb` file to your Colab account. The first four cells of the notebook will:

1. Download Ollama in Colab.
2. Start the Ollama server in the background.
3. Pull the relevant LLM models to be used.
4. Clone this repository in Colab.

The fifth cell acts as the user's terminal in Google Colab. The following line runs the `BacktestEngine.py` script:

```python
process = subprocess.Popen(["python", "BacktestEngine.py", "--aggregate"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
```

which is effectively this in a normal terminal
```bash
python BacktestEngine.py --aggregate
```



### 5.2 (Local Machine Only) Download LLM Transformer Models from DownloadLLM.sh (MacOS/Linux)
This script allows the user to download LLMs concurrently on a local machine. Feel free to modify the models to download. Windows users can use the Git Bash terminal or any Linux subsystem.
```bash
chmod +x DownloadLLM.sh
./DownloadLLM.sh -p 6
```

---

