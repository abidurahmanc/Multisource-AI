import streamlit as st  # Streamlit for building the web app
from PyPDF2 import PdfReader  # Library to read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
import os  # For file and directory operations
import requests  # For making HTTP requests to fetch website content
import re  # Regular expressions for URL parsing
from bs4 import BeautifulSoup  # For parsing HTML content from websites
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embeddings for text vectorization
import google.generativeai as genai  # Google Generative AI SDK
from langchain.vectorstores import FAISS  # FAISS for vector storage and similarity search
from langchain_google_genai import ChatGoogleGenerativeAI  # Chat model from Google
from langchain.chains.question_answering import load_qa_chain  # For question-answering pipeline
from langchain.prompts import PromptTemplate  # For creating custom prompts
from dotenv import load_dotenv  # For loading environment variables from .env file
import shutil  # For directory manipulation (e.g., removing FAISS index)
import logging  # For controlling log output
import warnings  # For suppressing warnings
from youtube_transcript_api import YouTubeTranscriptApi  # For fetching YouTube transcripts
from PIL import Image  # For opening and processing image files
import pytesseract  # For OCR (extracting text from images)

# Suppress deprecation warnings and set logging to ERROR level to reduce verbosity
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.ERROR)

# Load environment variables from .env file (e.g., GOOGLE_API_KEY)
load_dotenv()
# Configure the Google Generative AI SDK with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""  # Initialize empty string to store extracted text
    for pdf in pdf_docs:  # Iterate over each uploaded PDF
        pdf_reader = PdfReader(pdf)  # Create a PdfReader object
        for page in pdf_reader.pages:  # Loop through all pages in the PDF
            text += page.extract_text()  # Append extracted text from each page
    return text  # Return the combined text

# Function to extract text from a website URL
def extract_clean_text(url):
    """Fetches and cleans text from a website URL."""
    # Suppress SSL warnings for unverified requests
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False)  # Fetch website content (ignoring SSL verification)
    soup = BeautifulSoup(response.text, "html.parser")  # Parse HTML content
    text = soup.get_text(separator=" ")  # Extract text with spaces between elements
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single space and trim
    return text

# Function to fetch transcript from a YouTube video URL
def get_youtube_transcript(youtube_url):
    """Extracts transcript text from a YouTube video URL."""
    try:
        # Extract video ID from URL using regex
        video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        video_id = video_id.group(1)  # Get the 11-character video ID
        # Fetch transcript using YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine all transcript segments into a single string
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {str(e)}")  # Display error in Streamlit
        return None  # Return None if transcript fetch fails

# Function to extract text from uploaded image files using OCR
def get_image_text(image_files):
    """Extracts text from a list of image files using Tesseract OCR."""
    text = ""  # Initialize empty string to store extracted text
    for image_file in image_files:  # Iterate over each uploaded image
        try:
            image = Image.open(image_file)  # Open the image file
            text += pytesseract.image_to_string(image) + " "  # Extract text and append with a space
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {str(e)}")  # Display error in Streamlit
    return text.strip()  # Return the combined text, trimmed of extra spaces

# Function to split text into manageable chunks for vectorization
def get_text_chunks(text):
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)  # Initialize splitter
    chunks = text_splitter.split_text(text)  # Split text into chunks
    return chunks  # Return list of text chunks

# Function to create and save a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks and saves it locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize embeddings model
    faiss_path = "faiss_index"  # Define directory for FAISS index

    # Remove existing FAISS index and recreate the directory
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    os.makedirs(faiss_path, exist_ok=True)

    try:
        # Create FAISS vector store from text chunks and embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(faiss_path)  # Save the vector store to disk
        st.session_state.processed = True  # Mark processing as complete
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")  # Display error in Streamlit
        st.session_state.processed = False  # Mark processing as failed
    return vector_store  # Return the vector store object

# Function to create a question-answering chain using a chat model
def get_conversational_chain():
    """Sets up a question-answering chain with a custom prompt."""
    # Define the prompt template for the QA chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "answer is not available in the context".
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)  # Initialize chat model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load QA chain
    return chain  # Return the chain object

# Function to process user questions and provide answers
def user_input(user_question):
    """Processes a user question and returns an answer from the FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize embeddings model
    faiss_path = "faiss_index"  # Define FAISS index directory
    index_file = os.path.join(faiss_path, "index.faiss")  # Path to FAISS index file

    # Check if FAISS index exists
    if not os.path.exists(index_file):
        st.error("No FAISS index found! Please process a PDF, Website, YouTube video, or Images first.")
        return

    try:
        # Load the FAISS index from disk
        new_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)  # Search for relevant documents
        chain = get_conversational_chain()  # Get the QA chain
        # Run the chain with the question and retrieved documents
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])  # Display the answer
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")  # Display error in Streamlit

# Main function to run the Streamlit app
def main():
    """Main function to set up and run the Streamlit app."""
    st.set_page_config("MultiSource AI")  # Set page title
    st.header("MultiSource AI")  # Display header

    # Initialize session state variables if not already set
    if 'processed' not in st.session_state:
        st.session_state.processed = False  # Tracks if data has been processed
    if 'input_source' not in st.session_state:
        st.session_state.input_source = "PDF"  # Default input source
    if 'question_key' not in st.session_state:
        st.session_state.question_key = 0  # Key for resetting question input

    # Sidebar for input source selection and processing
    with st.sidebar:
        st.title("Menu:")  # Sidebar title
        
        # Radio buttons to select input source
        selected_source = st.radio(
            "Select Input Source",
            ("PDF", "Website", "YouTube", "Images"),
            index={"PDF": 0, "Website": 1, "YouTube": 2, "Images": 3}.get(st.session_state.input_source, 0),
            key="input_source_radio"
        )

        # Reset state when input source changes
        if selected_source != st.session_state.input_source:
            st.session_state.input_source = selected_source
            st.session_state.processed = False  # Reset processed flag
            st.session_state.question_key += 1  # Increment to reset question input
            faiss_path = "faiss_index"
            if os.path.exists(faiss_path):
                shutil.rmtree(faiss_path)  # Clear old FAISS index

        # PDF processing section
        if st.session_state.input_source == "PDF":
            pdf_docs = st.file_uploader("Upload PDF Files and Click Submit", accept_multiple_files=True)
            if st.button("Submit & Process PDF") and pdf_docs:  # Process when button clicked and files uploaded
                with st.spinner("Processing PDF..."):  # Show spinner during processing
                    raw_text = get_pdf_text(pdf_docs)  # Extract text
                    text_chunks = get_text_chunks(raw_text)  # Split into chunks
                    get_vector_store(text_chunks)  # Create FAISS index
                    st.session_state.question_key += 1  # Reset question input
                    st.success("PDF Processing Complete!")  # Show success message

        # Website processing section
        elif st.session_state.input_source == "Website":
            with st.form(key="website_form"):  # Use form to avoid Enter key issues
                url = st.text_input("Enter Website URL")  # Input field for URL
                submit_button = st.form_submit_button(label="Submit & Process Website")  # Submit button
                
                if submit_button and url:  # Process when button clicked and URL provided
                    with st.spinner("Processing Website..."):
                        raw_text = extract_clean_text(url)  # Extract text
                        text_chunks = get_text_chunks(raw_text)  # Split into chunks
                        get_vector_store(text_chunks)  # Create FAISS index
                        st.session_state.question_key += 1  # Reset question input
                        st.success("Website Processing Complete!")

        # YouTube processing section
        elif st.session_state.input_source == "YouTube":
            with st.form(key="youtube_form"):  # Use form to avoid Enter key issues
                youtube_url = st.text_input("Enter YouTube Video URL")  # Input field for URL
                submit_button = st.form_submit_button(label="Submit & Process YouTube")  # Submit button
                
                if submit_button and youtube_url:  # Process when button clicked and URL provided
                    with st.spinner("Processing YouTube Video..."):
                        raw_text = get_youtube_transcript(youtube_url)  # Fetch transcript
                        if raw_text:  # Proceed only if transcript is fetched
                            text_chunks = get_text_chunks(raw_text)  # Split into chunks
                            get_vector_store(text_chunks)  # Create FAISS index
                            st.session_state.question_key += 1  # Reset question input
                            st.success("YouTube Processing Complete!")

        # Images processing section
        elif st.session_state.input_source == "Images":
            # File uploader for images with specific types
            image_files = st.file_uploader("Upload Image Files and Click Submit", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
            if st.button("Submit & Process Images") and image_files:  # Process when button clicked and files uploaded
                with st.spinner("Processing Images..."):
                    raw_text = get_image_text(image_files)  # Extract text via OCR
                    if raw_text:  # Proceed only if text is extracted
                        text_chunks = get_text_chunks(raw_text)  # Split into chunks
                        get_vector_store(text_chunks)  # Create FAISS index
                        st.session_state.question_key += 1  # Reset question input
                        st.success("Images Processing Complete!")
                    else:
                        st.error("No text extracted from images.")  # Show error if no text found

    # Main area for user question input
    user_question = st.text_input(
        "Ask a Question from the Processed Data",
        disabled=not st.session_state.processed,  # Disable until data is processed
        help="Please process a PDF, Website, YouTube video, or Images first before asking a question" if not st.session_state.processed else "",
        key=f"question_{st.session_state.question_key}"  # Unique key to reset input
    )
    
    # Submit button for question
    if st.button("Submit Question") and user_question and st.session_state.processed:
        user_input(user_question)  # Process the question and display answer

# Entry point of the script
if __name__ == "__main__":
    main()  # Run the main function