# LLMStrategy Data Pipeline

(Currently only data pipeline and prompt specification available (Just the Framework), stay tuned for future updates)
Note: PromtCoder is developed by dhh1995 to streamline and modularise the creation of LLM Prompts.
<br>
<br>
Stay tuned for future updates!

---

# Tutorials (Getting Started)

- **Using the API for Data**: In `DataPipeLine/API Tutorial.ipynb`, you'll find the basic coding structure along with example usage on how to fetch data from the US Census Bureau, FRED Economic Data, and the Alpha Vantage Portal.

- **Using Procoder for Formatting LLM Prompts**: In `Prompt/Prompt Tutorial.ipynb`, you'll find the essential coding structure and example usage for dynamically formatting LLM prompts.

---


## Start using this repository

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

### 3. Run the scraper
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

