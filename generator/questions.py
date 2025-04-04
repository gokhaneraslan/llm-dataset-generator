import ollama
import re
import logging
from typing import Optional, List

    
# Question Generation and Parsing
def parse_questions_from_response(response_text: str) -> List[str]:
    """
    Parses a numbered list of questions from the LLM response string.
    Tries to handle potential variations in formatting.
    """
    if not response_text or not response_text.strip():
        logging.warning("Empty response received from LLM when generating questions")
        return []
        
    questions = []
    # Regex to find lines starting with a number, period, optional space, then the question
    question_pattern = re.compile(r"^\s*\d+\.\s*(.*)", re.MULTILINE)
    matches = question_pattern.findall(response_text)

    for match in matches:
        question = match.strip()
        if question:
            question = question.strip(' *')
            questions.append(question)

    # Fallback if regex finds nothing
    if not questions and response_text:
        logging.warning("Primary question pattern matching failed, trying fallback method")
        lines = response_text.splitlines()
        potential_questions = []
        for line in lines:
            line = line.strip()
            # Basic check if it looks like a list item (number or bullet)
            if line and len(line) > 2 and (line[0].isdigit() or line[0] in ['-', '*']):
                # Try to remove the list marker
                cleaned_line = re.sub(r"^\s*[\d\.\-\*]+\s*", '', line)
                if cleaned_line and '?' in cleaned_line:  # Simple heuristic: does it contain a question mark?
                    potential_questions.append(cleaned_line)
            # Sometimes the model might just list them without clear markers if the prompt failed
            elif line and '?' in line and len(line.split()) > 3:  # Longer line with question mark
                potential_questions.append(line)

        # If fallback found something that looks like questions
        if potential_questions:
            logging.warning(f"Used fallback method to parse questions. Found {len(potential_questions)} potential questions.")
            questions = potential_questions

    return questions


def generate_questions_ollama(
    content: str,
    num_questions: int,
    model: str ,
    temperature: float,
    max_gen_prompt_content_len: int) -> Optional[List[str]]:
    """
    Generates questions based on content using Ollama and parses the response.
    """
    if not content:
        logging.error("Cannot generate questions from empty content")
        return None

    # Truncate content for the generation prompt if it's very long
    truncated_content = content[:max_gen_prompt_content_len]
    if len(content) > max_gen_prompt_content_len:
        logging.warning(f"Document content truncated to {max_gen_prompt_content_len} chars for question generation prompt")

    prompt = f"""
        Based *only* on the following document content, please generate exactly {num_questions} relevant and
        insightful questions that test understanding of the main topics, key details, or potential implications discussed.
        Do not ask questions that cannot be answered from the provided text.
        Format the output *strictly* as a numbered list, with each question on a new line (e.g., "1. Question one?").
        Do not include any introductory text like "Here are the questions:" or any concluding remarks.

        Document Content:
        --- START OF DOCUMENT ---
        {truncated_content}
        --- END OF DOCUMENT ---

        Generate {num_questions} questions now:
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        logging.info(f"Generating {num_questions} questions using model '{model}' (temp={temperature})...")
        response = ollama.chat(
            model=model,
            messages=messages,
            options={'temperature': temperature},
        )

        raw_response_text = response['message']['content']
        questions = parse_questions_from_response(raw_response_text)

        if not questions:
            logging.warning("Could not parse any questions from the model's response")
            return []

        if len(questions) != num_questions:
            logging.warning(f"Received {len(questions)} questions, but requested {num_questions}")

        return questions

    except ollama.ResponseError as re:
        logging.error(f"Ollama API Error during question generation: {re.status_code} - {re.error}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during question generation: {e}")
        return None
    
