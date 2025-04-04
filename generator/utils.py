import os
import ollama
import logging
from typing import List, Dict, Union, Optional
import pandas as pd
import json
import fitz



# Ollama Interaction Functions
def check_ollama_connection() -> bool:
    """Checks if Ollama is running and accessible."""
    try:
        # Simple ping to the Ollama API
        ollama.list()
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        logging.error("Make sure Ollama is installed and running (run 'ollama serve' in terminal)")
        return False


def check_ollama(model_gen, model_ret, default_gen_model, default_ret_model):
        # Check if Ollama is accessible
    if not check_ollama_connection():
        return None

    # Check if the model exists
    try:
      
        models = ollama.list()
        model_names = [m.get('model') for m in models.get('models', [])]
        
        if model_gen not in model_names:
            logging.error(f"Model '{model_gen}' not found in Ollama. Available models: {', '.join(model_names)}")
            logging.error(f"Run 'ollama pull {model_gen}' to download the model")
            return None
        elif model_ret not in model_names:
            logging.error(f"Model '{model_ret}' not found in Ollama. Available models: {', '.join(model_names)}")
            logging.error(f"Run 'ollama pull {model_ret}' to download the model")
            return None
        elif default_gen_model not in model_names:
            logging.error(f"Model '{default_gen_model}' not found in Ollama. Available models: {', '.join(model_names)}")
            logging.error(f"Run 'ollama pull {default_gen_model}' to download the model")
            return None
        elif default_ret_model not in model_names:
            logging.error(f"Model '{default_ret_model}' not found in Ollama. Available models: {', '.join(model_names)}")
            logging.error(f"Run 'ollama pull {default_ret_model}' to download the model")
            return None
        else:
            return True
          
    except Exception as e:
        logging.error(f"Error checking model availability: {e}")
        return None

    
# File Reading Functions
def read_text_file(filepath: str) -> Optional[str]:
    """Reads content from a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content:
            logging.warning(f"File '{filepath}' is empty")
        return content
    except FileNotFoundError:
        logging.error(f"File not found at '{filepath}'")
        return None
    except UnicodeDecodeError:
        logging.error(f"Unable to decode '{filepath}' with UTF-8 encoding, try converting the file")
        return None
    except Exception as e:
        logging.error(f"Error reading text file '{filepath}': {e}")
        return None

def read_pdf_file(filepath: str) -> Optional[str]:
    """Reads text content from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            if len(doc) == 0:
                logging.warning(f"PDF file '{filepath}' contains no pages")
                return ""
                
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")  # Extract plain text
                if page_text:
                    text += page_text + "\n"  # Add newline between pages
            
            # Basic cleanup
            text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
            
            if not text:
                logging.warning(f"No readable text extracted from PDF file '{filepath}'")
            
        return text
    except FileNotFoundError:
        logging.error(f"File not found at '{filepath}'")
        return None
    except fitz.FileDataError:
        logging.error(f"'{filepath}' is not a valid PDF file or is corrupted")
        return None
    except fitz.EmptyFileError:
        logging.error(f"PDF file '{filepath}' is empty")
        return None
    except fitz.FitzError as fe:
        logging.error(f"Error processing PDF file '{filepath}' with PyMuPDF: {fe}")
        return None
    except Exception as e:
        logging.error(f"Error reading PDF file '{filepath}': {e}")
        return None

def get_document_content(filepath: str, supported_file_types) -> Optional[str]:
    """Reads content from supported file types (.txt, .pdf)."""
    if not os.path.exists(filepath):
        logging.error(f"Path '{filepath}' does not exist")
        return None
        
    _, file_ext = os.path.splitext(filepath)
    file_ext = file_ext.lower()

    if file_ext == ".txt":
        return read_text_file(filepath)
    elif file_ext == ".pdf":
        return read_pdf_file(filepath)
    else:
        logging.error(f"Unsupported file type '{file_ext}'. Supported types: {', '.join(supported_file_types)}")
        return None

# Template Creation Functions
def create_default_template(generated_questions: List[str], answers_questions: List[str]) -> List[Dict[str, str]]:
    """Create a basic template with input/output pairs."""
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match")
        return []
    
    basic_temp = []
    for i in range(len(generated_questions)):
        basic_temp.append({"input": generated_questions[i], "output": answers_questions[i]})
    
    logging.debug(f"Created basic template with {len(basic_temp)} question-answer pairs")
    return basic_temp

def create_gemma_template(generated_questions: List[str], answers_questions: List[str]) -> List[Dict[str, str]]:
    """Create a template formatted for Gemma models."""
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match")
        return []
    
    gemma_temp = []
    for i in range(len(generated_questions)):
        gemma_temp.append({"content": generated_questions[i], "role": "user"})
        gemma_temp.append({"content": answers_questions[i], "role": "assistant"})
    
    logging.debug(f"Created Gemma template with {len(gemma_temp)//2} conversation pairs")
    return gemma_temp

def create_llama_template(generated_questions: List[str], answers_questions: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Create a template formatted for LLaMA models."""
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match")
        return {"conversations": []}
    
    conversations = []
    for i in range(len(generated_questions)):
        conversations.append({"from": "human", "value": generated_questions[i]})
        conversations.append({"from": "gpt", "value": answers_questions[i]})
    
    llama_temp = {
        "conversations": conversations
    }
    
    logging.debug(f"Created Llama template with {len(conversations)//2} conversation pairs")
    return llama_temp

def create_template(
    template: str, 
    generated_questions: List[str], 
    answers_questions: List[str],
    supported_templates ) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]], str]:
    
    """Create the appropriate template based on the specified format."""
    
    if not template.lower() in supported_templates:
        logging.error(f"Unsupported template: '{template}'. Supported templates: {', '.join(supported_templates)}")
        return f"Error: Unsupported template '{template}'. Choose from: {', '.join(supported_templates)}"
    
    if template.lower() == "default":
        return create_default_template(generated_questions, answers_questions)
    elif template.lower() == "gemma":
        return create_gemma_template(generated_questions, answers_questions)
    elif template.lower() == "llama":
        return create_llama_template(generated_questions, answers_questions)


def save_dataset(
    file: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]], 
    template: str,
    output_dir: str ) -> None:
    
    """Save the dataset in the appropriate format based on the template."""
    if isinstance(file, str) and file.startswith("Error:"):
        logging.error(file)
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(output_dir, f"custom_llm_dataset_{timestamp}.json")
    csv_filename = os.path.join(output_dir, f"custom_llm_dataset_{timestamp}.csv")

    # Save as JSON
    try:
        with open(json_filename, 'w', encoding="utf-8") as nf:
            json.dump(file, nf, ensure_ascii=False, indent=2)
        logging.info(f"Dataset saved to {json_filename}")
    except Exception as e:
        logging.error(f"Error saving JSON dataset: {e}")
    
    # For default template, also save as CSV
    if template.lower() == "default" and isinstance(file, list):
        try:
            df = pd.DataFrame(file)
            df.to_csv(csv_filename, index=False)
            logging.info(f"Dataset saved to {csv_filename}")
        except Exception as e:
            logging.error(f"Error saving CSV dataset: {e}")
