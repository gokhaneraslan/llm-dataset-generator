import sys
import os
import logging
import argparse
from generator.answers import answers_questions_ollama
from generator.questions import generate_questions_ollama
from generator.utils import get_document_content, check_ollama, create_template, save_dataset
import yaml


with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


# Set up logging
def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with appropriate format and level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=numeric_level,
        handlers=[
            logging.FileHandler("logs/llm_dataset_generator.log"),
            logging.StreamHandler(sys.stderr)
        ]
    )


# Main Execution
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a custom dataset from documents using Ollama LLM.')
    parser.add_argument('--file', '-f', type=str, required=True, 
                        help='Path to the document file (.txt or .pdf)')
    parser.add_argument('--questions', '-q', type=int, default=cfg["DEFAULT_NUM_QUESTIONS"],
                        help=f'Number of questions to generate (default: {cfg["DEFAULT_NUM_QUESTIONS"]})')
    parser.add_argument('--template', '-t', type=str, default=cfg["DEFAULT_TEMPLATE"], choices=cfg["SUPPORTED_TEMPLATES"],
                        help=f'Dataset template format (default: {cfg["DEFAULT_TEMPLATE"]})')
    parser.add_argument('--model-gen', '-mg', type=str, default=cfg["DEFAULT_GEN_OLLAMA_MODEL"],
                        help=f'Ollama model to use (default: {cfg["DEFAULT_GEN_OLLAMA_MODEL"]})')
    parser.add_argument('--model-ret', '-mr', type=str, default=cfg["DEFAULT_RET_OLLAMA_MODEL"],
                        help=f'Ollama model to use (default: {cfg["DEFAULT_RET_OLLAMA_MODEL"]})')
    parser.add_argument('--gen-temp', type=float, default=cfg["DEFAULT_GEN_TEMPERATURE"],
                        help=f'Temperature for question generation (default: {cfg["DEFAULT_GEN_TEMPERATURE"]})')
    parser.add_argument('--ret-temp', type=float, default=cfg["DEFAULT_RET_TEMPERATURE"],
                        help=f'Temperature for answer generation (default: {cfg["DEFAULT_RET_TEMPERATURE"]})')
    parser.add_argument('--output-dir', '-o', type=str, default=cfg["DEFAULT_OUTPUT_DIR"],
                        help=f'Directory to save the dataset (default: {cfg["DEFAULT_OUTPUT_DIR"]})')
    parser.add_argument('--log-level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    
    return parser.parse_args()

def main():
    """Main execution function with improved error handling and logging."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Log start of processing
    logging.info(f"Starting dataset generation with arguments: {vars(args)}")
    logging.info(f"Processing file: {args.file}")
    
    # Verify the file exists and can be read
    document_content = get_document_content(args.file, supported_file_types= cfg["SUPPORTED_FILE_TYPES"])
    if document_content is None:
        logging.error("Failed to process document content")
        return 1

    if not document_content.strip():
        logging.error(f"The file '{args.file}' appears to be empty or contains no readable text")
        return 1

    logging.info(f"Successfully read content ({len(document_content)} characters)")

    if not check_ollama(model_gen= args.model_gen, model_ret= args.model_ret, default_gen_model= cfg["DEFAULT_GEN_OLLAMA_MODEL"], default_ret_model= cfg["DEFAULT_RET_OLLAMA_MODEL"] ):
        return None
    
    # Generate Questions
    generated_questions = generate_questions_ollama(
        content=document_content,
        num_questions=args.questions or cfg["DEFAULT_NUM_QUESTIONS"],
        model=args.model_gen or cfg["DEFAULT_GEN_OLLAMA_MODEL"],
        temperature=args.gen_temp or cfg["DEFAULT_GEN_TEMPERATURE"],
        max_gen_prompt_content_len = cfg["MAX_GEN_PROMPT_CONTENT_LEN"]
    )

    if generated_questions is None:
        logging.error("Question generation failed")
        return 1

    if not generated_questions:
        logging.error("No questions were generated or parsed successfully")
        return 1

    logging.info(f"Generated {len(generated_questions)} questions successfully")
    
    # Answers Questions
    answers_questions = answers_questions_ollama(
      generated_questions=generated_questions,
      document_content=document_content,
      model=args.model_ret or cfg["DEFAULT_RET_OLLAMA_MODEL"],
      temperature=args.ret_temp or cfg["DEFAULT_RET_TEMPERATURE"]
    )

    if answers_questions is None:
        logging.error("Answers generation failed")
        return 1

    if not answers_questions:
        logging.error("No response were answered successfully")
        return 1

    logging.info(f"{len(answers_questions)} questions answered successfully.")

    # Create dataset from templates
    data = create_template(
        template=args.template,
        generated_questions=generated_questions,
        answers_questions=answers_questions,
        supported_templates= cfg["SUPPORTED_TEMPLATES"]
    )
    
    if data and not isinstance(data, str):
        # Save the dataset
        save_dataset(file=data, template=args.template or cfg["DEFAULT_TEMPLATE"], output_dir=args.output_dir or cfg["DEFAULT_OUTPUT_DIR"])
        logging.info("Dataset generation completed successfully")
        return 0
    else:
        logging.error("Failed to create dataset template")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)