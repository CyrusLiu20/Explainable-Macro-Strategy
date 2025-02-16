#!/bin/bash

# ===== Configuration =====
SAVE_DIR="TransformerModels"  # Directory to store models
LOG_FILE="DownloadTransformer.log"  # Log file to track progress
MAX_PARALLEL=4                # Default number of parallel downloads

# Colors for Output Formatting
GREEN="\033[32m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

# ===== List of Model Names =====
models=(
  # "mistralai/Mistral-7B-Instruct-v0.1"
  # "facebook/opt-6.7b"
  # "EleutherAI/gpt-neo-2.7B"
  # "microsoft/Phi-3-mini"
  # "databricks/dolly-v2-3b"
  # "google/gemma-2b"
  # "stabilityai/stablelm-3b-4e1t"
  # "mosaicml/mpt-7b"
  # "allenai/tulu-2-7b"
  # "tiiuae/falcon-7b"
  "distilbert-base-uncased"         # A smaller version of BERT
  "bert-base-uncased"               # Standard BERT model
  "albert-base-v2"                  # A lightweight version of BERT
  "distilroberta-base"              # A distilled version of RoBERTa
)

# ===== Download Function =====
download_model() {
  local model_name="$1"
  local folder_name="$2"
  local log_file="$3"
  local model_dir="$folder_name/$(basename "$model_name")"

  GREEN="\033[32m"
  RED="\033[31m"
  CYAN="\033[36m"
  RESET="\033[0m"

  echo -e "${CYAN}Downloading $model_name...${RESET}"
  huggingface-cli download "$model_name" --local-dir "$model_dir" --resume >> "$log_file" 2>&1
  
  if [[ $? -eq 0 ]]; then
    printf "${GREEN}Successfully downloaded: $model_name${RESET}\n"
  else
    printf "${RED}Failed to download: $model_name. Check $LOG_FILE for details.${RESET}\n"
  fi
}

# ===== Parse Command-line Arguments =====
while getopts ":p:d:h" opt; do
  case ${opt} in
    p ) MAX_PARALLEL=$OPTARG ;;
    d ) SAVE_DIR=$OPTARG ;;
    h ) show_help ;;
    \? ) echo -e "${RED}Invalid option: -$OPTARG${RESET}" >&2; exit 1 ;;
    : ) echo -e "${RED}Option -$OPTARG requires an argument.${RESET}" >&2; exit 1 ;;
  esac
done

# ===== Login =====
huggingface-cli login --token $HF_API_KEY

# ===== Setup =====
mkdir -p "$SAVE_DIR"
printf "${CYAN}Saving models to: $SAVE_DIR${RESET}\n"
printf "${CYAN}Using up to $MAX_PARALLEL parallel downloads${RESET}\n"

# ===== Start Parallel Downloads =====
echo "${CYAN}Downloading models... (Logging to $LOG_FILE)${RESET}"
echo "" > "$LOG_FILE"  # Clear previous log file

# Run downloads in parallel
export -f download_model
echo "${models[@]}" | xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'download_model "$1" "$2" "$3"' _ {} "$SAVE_DIR" "$LOG_FILE"
printf "${GREEN}All downloads complete!${RESET}"