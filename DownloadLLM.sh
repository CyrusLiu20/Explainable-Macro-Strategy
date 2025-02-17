#!/bin/bash

# ===== Configuration =====
SAVE_DIR="TransformerModels"  # Directory to store models
LOG_FILE="Logs/DownloadTransformer.log"  # Log file to track progress
MAX_PARALLEL=12                # Default number of parallel downloads

# Colors for Output Formatting
GREEN="\033[32m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

# ===== List of Hugging Face Model Names =====
models=(
  "deepseek-r1:1.5b"
  "llama3.2"
  "qwen:0.5b"
  "qwen:1.8b"
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
  # huggingface-cli download "$model_name" --local-dir "$model_dir" --resume >> "$log_file" 2>&1
  ollama pull "$model_name" >> "Logs/$log_file" 2>&1

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
# huggingface-cli login --token $HF_TOKEN
ollama serve > OllamaServe.log 2>&1 &
printf "${CYAN}Running ollama serve in the background${RESET} (Using up to $MAX_PARALLEL parallel downloads)\n"

# ===== Start Parallel Downloads =====
printf "${CYAN}Downloading models... (Logging to $LOG_FILE)${RESET}"
echo "" > "$LOG_FILE"  # Clear previous log file

# Run downloads in parallel
export -f download_model
echo "${models[@]}" | xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'download_model "$1" "$2" "$3"' _ {} "$SAVE_DIR" "$LOG_FILE"
pkill ollama
printf "${GREEN}All downloads complete!${RESET}\n"
printf "${CYAN}Terminate ollama serve in the background${RESET}"