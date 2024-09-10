import tempfile
import pytesseract
import fitz
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2
import spacy
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

# Load spaCy model once to avoid repeated loading
nlp = spacy.load('en_core_web_sm')

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
