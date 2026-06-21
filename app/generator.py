import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#model for generation
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0, 
    api_key=os.getenv("GROQ_API_KEY")
)

def load_prompt(filename=None):
    """Loads the system prompt from the version-controlled text file."""
    filename = filename or os.getenv("RAG_SYSTEM_PROMPT_FILE", "system_prompt_v2.txt")
    prompt_path = os.path.join("prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()

def generate_answer(question: str, formatted_context: str):
    """Passes the formatted context and user question to the LLM."""
    system_prompt = load_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    parser = StrOutputParser()

    rag_chain = prompt | llm | parser
    
    return rag_chain.invoke({
        "context": formatted_context,
        "question": question
    })

def check_faithfulness(question: str, context: str, generated_answer: str) -> bool:
    """
    Acts as an auditor. Returns True if the generated answer is strictly 
    supported by the context, and False if it detects hallucinations.
    """
    eval_prompt_text = """You are a strict grading auditor. 
    Compare the GENERATED ANSWER to the PROVIDED CONTEXT.
    If the GENERATED ANSWER contains ANY facts, numbers, or claims that are NOT explicitly stated in the PROVIDED CONTEXT, you must output the word "FAIL".
    If the GENERATED ANSWER is entirely supported by the PROVIDED CONTEXT, output the word "PASS".
    
    PROVIDED CONTEXT:
    {context}
    
    GENERATED ANSWER:
    {answer}
    
    Output ONLY "PASS" or "FAIL". Do not explain."""
    
    eval_prompt = ChatPromptTemplate.from_template(eval_prompt_text)
    parser = StrOutputParser()
    eval_chain = eval_prompt | llm | parser
    
    result = eval_chain.invoke({
        "context": context,
        "answer": generated_answer
    })
    
    return "PASS" in result.strip().upper()
