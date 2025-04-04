# Custom LLM Dataset Generator

A Python tool for automatically generating question-answer datasets from documents using Ollama for local LLM inference.

## Overview

This tool allows you to create custom training datasets for fine-tuning language models by:
1. Extracting text content from documents (PDF, TXT)
2. Generating relevant questions based on the document content
3. Creating answers for each question using only information from the source document
4. Formatting the results into various dataset templates

## Features

- 📄 Support for PDF and text files
- 🔍 Automatic question generation based on document content
- ✅ Answer generation strictly from document context
- 📊 Multiple export formats (default, gemma, llama)
- 🚀 Uses Ollama for local LLM inference
- 📝 Comprehensive logging

## Requirements

- Python 3.6+
- Ollama installed and running locally
- Required Python packages:
  - pymupdf (fitz)
  - pandas
  - ollama
- Optional Python packages:
  - argparse
  - pathlib
  - logging

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gokhaneraslan/llm-dataset-generator.git
   cd llm-dataset-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running:
   ```bash
   # Install Ollama (if not already installed)
   # See https://ollama.com/download for installation instructions
   
   # Start Ollama server
   ollama serve
   ```

4. Pull the model you want to use (e.g. Qwen 2.5 7B):
   ```bash
   ollama pull qwen2.5:7b
   ```

## Usage

Basic usage:

```bash
python main.py --file path/to/document.pdf
```

Advanced options:

```bash
python main.py --file path/to/document.pdf \
               --questions 15 \
               --template llama \
               --model-gen qwen2.5:7b \
               --model-ret qwen2.5:7b \
               --gen-temp 0.2 \
               --ret-temp 0.0 \
               --output-dir ./training_data \
               --log-level INFO
```

## Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--file` | `-f` | Path to document file (.txt or .pdf) | Required |
| `--questions` | `-q` | Number of questions to generate | 10 |
| `--template` | `-t` | Dataset template format (default, gemma, llama) | default |
| `--model-gen` | `-mg` | Ollama model for question generation | qwen2.5:7b |
| `--model-ret` | `-mr` | Ollama model for answering questions | qwen2.5:7b |
| `--gen-temp` | | Temperature for question generation | 0.1 |
| `--ret-temp` | | Temperature for answer generation | 0.0 |
| `--output-dir` | `-o` | Directory to save dataset files | ./datasets |
| `--log-level` | | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |


### Examples

Generate 15 questions using the Gemma template:
```bash
python main.py --file document.pdf --questions 15 --template gemma
```

Use a different model with custom temperature settings:
```bash
python main.py --file document.txt --model-gen llama3:latest --gen-temp 0.2 --ret-temp 0.1
```

Save output to a specific directory:
```bash
python main.py --file research_paper.pdf --output-dir ./custom_dataset
```

## Output Formats

The script supports multiple output formats:

### default Format (default)
```json
[
  {
    "input": "Question 1?",
    "output": "Answer 1"
  },
  {
    "input": "Question 2?",
    "output": "Answer 2"
  }
]
```

### gemma Format
```json
[
  {
    "content": "Question 1?",
    "role": "user"
  },
  {
    "content": "Answer 1",
    "role": "assistant"
  },
  {
    "content": "Question 2?",
    "role": "user"
  },
  {
    "content": "Answer 2",
    "role": "assistant"
  }
]
```

### llama Format
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Question 1?"
    },
    {
      "from": "gpt",
      "value": "Answer 1"
    },
    {
      "from": "human",
      "value": "Question 2?"
    },
    {
      "from": "gpt",
      "value": "Answer 2"
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **"Failed to connect to Ollama"**
   - Make sure Ollama is installed and running with `ollama serve`

2. **"Model not found in Ollama"**
   - Pull the model first: `ollama pull model_name`

3. **PDF extraction issues**
   - Ensure the PDF is not password-protected
   - Try converting to text first if the PDF has complex formatting

## Logging

The script logs information to both the console and a file named `llm_dataset_generator.log` in the logs directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
