import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0, # Set to 0 to prevent hallucinations and strictly follow facts
    api_key=os.getenv("GROQ_API_KEY")
)

# The strict citation prompt template
system_prompt = """You are a highly accurate expert assistant answering questions based STRICTLY on the provided documents.

INSTRUCTIONS:
1. Answer ONLY using the information found in the sources below. 
2. If the answer is not contained in the sources, you must reply: "I cannot answer this based on the provided documents." Do not guess.
3. For EVERY claim or fact you state, you MUST cite the source.
4. Format your citations at the end of the relevant sentence like this: [Source: filename.pdf, Page: X].

CONTEXT:
{context}"""

# Assemble the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

parser = StrOutputParser()

rag_chain = prompt | llm | parser

def generate_answer(question: str, formatted_context: str):
    """Passes the formatted context and user question to the LLM."""
    return rag_chain.invoke({
        "context": formatted_context,
        "question": question
    })