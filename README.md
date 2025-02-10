# LLMStrategy Data Pipeline

(Currently only data pipeline and prompt specification available (Just the Framework), stay tuned for future updates)
Note: PromtCoder is developed by dhh1995 to streamline and modularise the creation of LLM Prompts.
<br>
<br>
Stay tuned for future updates!

---

# Developer Tutorials (Getting Started)

- **Using the API for Data**: In `DataPipeLine/API Tutorial.ipynb`, you'll find the basic coding structure along with example usage on how to fetch data from the US Census Bureau, FRED Economic Data, and the Alpha Vantage Portal.

- **Using Procoder for Formatting LLM Prompts**: In `Prompt/Prompt Tutorial.ipynb`, you'll find the essential coding structure and example usage for dynamically formatting LLM prompts.

- **Current Trends in LLMs for Quantitative Trading**: For an updated understanding of the latest advancements in using LLMs to generate explainable alpha signals, please refer to the `ReferencePapers` folder.

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
python CombinedScraper.py
```


### 4. Install the PromptCoder Package
```bash
cd PromptCoder
pip install -e .
cd ..
```
---

