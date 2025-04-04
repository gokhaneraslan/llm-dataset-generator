import ollama
import logging
from typing import Optional, List


def answer_question_ollama(
    question: str,
    context: str,
    model: str,
    temperature: float) -> Optional[str]:
    """
    Answers a given question based *only* on the provided context using Ollama.

    Args:
        question: The question to answer.
        context: The document text content to use as the knowledge base.
        model: The Ollama model to use.
        temperature: The generation temperature (lower is better for factual answers).

    Returns:
        The answer string, a specific "not found" message, or None if an error occurs.
    """
    if not context:
        logging.error("Cannot answer question without context")
        return None
    if not question:
        logging.error("Cannot answer an empty question")
        return None

    prompt = f"""
        You are an expert assistant specialized in answering questions based *strictly* on the provided text document.
        Use *only* the information available in the document below to answer the question.
        Do not use any prior knowledge or information from outside the document.
        If the answer cannot be found within the document text, you MUST respond with the exact phrase:
        "The document does not provide an answer to this question."
        Do not add any explanation or apology if the answer is not found.
        Provide a concise and direct answer based on the text. Avoid introductory phrases like "According to the document..." unless essential for clarity.

        --- START OF DOCUMENT ---
        {context}
        --- END OF DOCUMENT ---

        Based *only* on the document above, answer the following question:
        Question: {question}

        Answer:
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        logging.debug(f"Answering question using model '{model}' (temp={temperature})")
        response = ollama.chat(
            model=model,
            messages=messages,
            options={'temperature': temperature},
        )

        answer = response['message']['content'].strip()
        return answer

    except ollama.ResponseError as re:
        logging.error(f"Ollama API Error during question answering: {re.status_code} - {re.error}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during question answering: {e}")
        return None



def answers_questions_ollama(
    document_content: str, 
    generated_questions: Optional[List[str]],
    model: str,
    temperature: float ) -> Optional[List[str]]:
    
    answers_questions = []
    for i, question in enumerate(generated_questions, 1):
        answer = answer_question_ollama(
            question=question,
            context=document_content,
            model= model,
            temperature= temperature
        )
        
        if answer is None:
            logging.error(f"Error occurred while trying to answer question {i}")
            answer = "Error occurred while generating this answer."
        
        answers_questions.append(answer)
        
    return answers_questions