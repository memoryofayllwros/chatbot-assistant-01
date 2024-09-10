import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_MODEL"] = "gpt-3.5-turbo-0125"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"

#pip install pytesseract pymupdf pdfplumber pdf2image langchain langchain-openai langchain-community python-dotenv opencv-python-headless faiss-cpu
#pip install spacy
#python -m spacy download en_core_web_sm
import tempfile
import pytesseract
import fitz
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import cv2
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Tuple
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
import spacy
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

llm = ChatOpenAI()
# Load spaCy model once to avoid repeated loading

nlp = spacy.load('en_core_web_sm')

base_dir = "Richford_files"

def preprocess_image(image):
    """Preprocess the image for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return denoised

def ocr_pdf_page(pdf_path, page_number):
    """Perform OCR on a single page of a PDF"""
    text = ""
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(pdf_path, 
                                   output_folder=path, 
                                   first_page=page_number + 1, 
                                   last_page=page_number + 1)
        
        for image in images:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            preprocessed_image = preprocess_image(image_cv)
            response = pytesseract.image_to_string(preprocessed_image)
            text += response + " "
            
    return text

def extract_table_text_from_pdf(file_path):
    """Extract text from tables within a PDF"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean_row = [str(cell) if cell is not None else '' for cell in row]
                    text += " | ".join(clean_row) + "\n"
    return text

def process_pdf(file_path):
    """Process PDF to extract text with fallback to OCR for non-extractable text"""
    text = ""
    doc = fitz.open(file_path)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_page = {executor.submit(process_page, doc, page_number): page_number for page_number in range(len(doc))}
        for future in concurrent.futures.as_completed(future_to_page):
            page_number = future_to_page[future]
            try:
                page_text = future.result()
                text += page_text
            except Exception as exc:
                print(f'Page {page_number} generated an exception: {exc}')
                
    doc.close()
    text += extract_table_text_from_pdf(file_path)
    return text

def process_page(doc, page_number):
    """Extract text from a single page, with OCR fallback"""
    page = doc.load_page(page_number)
    page_text = page.get_text("text")
    if not page_text.strip():
        return ocr_pdf_page(doc.name, page_number)  # OCR only if text extraction fails
    return page_text

#info extraction, using spaCy to detect named entities (like dates, names, etc.) in the extracted text
def information_extraction(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def process_image(file_path):
    """Process an image file for OCR"""
    text = ""
    with open(file_path, 'rb') as image_file:
        image_pil = Image.open(image_file)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        preprocessed_image = preprocess_image(image_cv)
        response = pytesseract.image_to_string(preprocessed_image)
        text += response + " "
    return text


st.set_page_config(page_title="Product Help Desk", page_icon="🤖")
@st.cache_resource

def process_files(base_dir):
    """Process all files in the directory to extract text"""
    final_text = ""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                futures.append(executor.submit(process_pdf, file_path))
            elif file.endswith(('.png', '.jpeg', '.jpg')):
                futures.append(executor.submit(process_image, file_path))
                
        for future in concurrent.futures.as_completed(futures):
            final_text += future.result()
    return final_text


def get_text_chunks(final_text):
    """Split extracted text into manageable chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(final_text)
    return chunks

@st.cache_resource
def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using OpenAI embeddings"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

extract_text = process_files(base_dir)
text_chunks = get_text_chunks(extract_text)
vectorstore = get_vectorstore(text_chunks)
retriever = vectorstore.as_retriever()


#------create tools------
retriever_tool = create_retriever_tool(retriever, 
                                       "texts_retriever",
                                       "Searches queries and returns relevant excerpts based on user questions")


from langchain.tools import BaseTool
import json

# Create a tool class by extending BaseTool and implementing the _run method
class EntityExtractionTool(BaseTool):
    name = "entity_extractor"
    description = "Extracts named entities (like names, dates, organizations, etc.) from text using spaCy."
    
    # Implement the _run method that LangChain expects
    def _run(self, text: str):
        """Extract named entities from text using spaCy."""
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return json.dumps(entities)  # Ensure the output is JSON serializable

    # For async support, you can also implement _arun (optional)
    async def _arun(self, text: str):
        raise NotImplementedError("Async execution is not supported for this tool.")

# Instantiate the tool
entity_tool = EntityExtractionTool()

# Add the tool to the list of tools
tools = [retriever_tool, entity_tool]


#------create prompt------
import reasoning_prompts
system_prompt = reasoning_prompts.system_prompt
qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                              MessagesPlaceholder(variable_name="chat_history"),
                                              ("user", "{input}"),
                                              MessagesPlaceholder(variable_name="agent_scratchpad"),
                                           ])

llm_with_tools = llm.bind_tools(tools)


#------format_chat_history------
def _format_chat_history(chat_history: List[Tuple[str, str]]):
    """Format chat history for display"""
    buffer = []
    for human, AI in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=AI))
    return buffer


#------generate response------
@st.cache_resource
def Loading(agent_input):
    """Get response from the agent"""
    agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | qa_prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
    )

    class AgentInput(BaseModel):
        input: str
        chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}})
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(input_type=AgentInput)

    return agent_executor.invoke(agent_input)["output"]


#------streamlit UI------
def main():
    st.title("💬 Product Help Desk")
    st.caption("A chatbot powered by OpenAI.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input := st.chat_input():
        agent_input = {
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            }
        
        results = Loading(agent_input)
        st.session_state.chat_history.append((user_input, results))

        for user_msg, ai_msg in st.session_state.chat_history:
            st.markdown(f"""
            <div style="padding: 1.5rem; 
                        border-radius: 0.5rem; 
                        margin-bottom: 1rem; 
                        background-color: #dcdcdc;
                        color: #000000;">
                <strong>You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="padding: 1.5rem; 
                        border-radius: 0.5rem; 
                        margin-bottom: 1rem; 
                        background-color: #ffffff;
                        color: #000000;">
                <strong>Assistant:</strong> {ai_msg}
            </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()



