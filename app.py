import torch
from torch import autocast
import pandas as pd
import pdfplumber
import camelot
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy 
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import logging
import fitz

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
CHROMA_DB_PATH = "./chroma_db"
EXAMPLES = [["Sample Financial Statement.pdf"]]
keywords = ["Statement of Profit and Loss", "Revenue", "Expenses"]

# Global variables for models and collections
embedding_model = None
llm_model = None
llm_tokenizer = None
collection = None

# ChromaDB client initialization
def initialize_chroma_client():
    """Initialize ChromaDB client once."""
    global collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name="financial_data")
        logger.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")
        raise e

# Load models once and reuse for inference
def load_models():
    """Load embedding and LLM models once and keep them in memory."""
    global embedding_model, llm_tokenizer, llm_model
    try:
        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME).to(DEVICE)
        if llm_tokenizer is None or llm_model is None:
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

# Page extraction logic
def extract_page_no(pdf_path, keywords):
    """Extract the page number of the Profit & Loss statement."""
    try:
        doc = fitz.open(pdf_path)
        keyword_pages = {keyword: set() for keyword in keywords}

        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").lower()
            for keyword in keywords:
                if keyword.lower() in text:
                    keyword_pages[keyword].add(page_num + 1)  # Convert to 1-based index
        doc.close()
        common_pages = set.intersection(*keyword_pages.values()) if keyword_pages else set()
    
        return str(sorted(common_pages)[1])
    except Exception as e:
        logger.error(f"Error extracting page number from PDF: {e}")
        return None

# Table extraction and preprocessing
def extract_profit_loss_tables(pdf_path, page):
    """Extract Profit & Loss tables from the PDF."""
    try:
        tables = camelot.read_pdf(pdf_path, pages=page, flavor='stream')
        if tables:
            df = tables[0].df
            df = df.iloc[2:].reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df.drop(index=[0, 1])
            df[['Year ended March 31,2024', 'Year ended March 31,2023']] = df['Year ended March 31,'].str.split('\n', expand=True)
            df = df.drop(columns=['Note No.', 'Year ended March 31,'])
            df.columns.values[1] = 'Three months ended March 31,2024'
            df.columns.values[2] = 'Three months ended March 31,2023'
            df = df.reset_index(drop=True)
            return df
        else:
            logger.warning("No tables found on the specified page.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return pd.DataFrame()

# Embedding and storage
def embed_and_store(df):
    """Generate embeddings for the DataFrame and store in ChromaDB."""
    try:
        if df.empty:
            logger.warning("Empty DataFrame passed for embedding.")
            return
        rows_text = df.astype(str).agg(' '.join, axis=1).tolist()
        embeddings = embedding_model.encode(rows_text, convert_to_tensor=True, device=DEVICE).cpu().numpy()

        # Retrieve existing IDs and metadata
        existing_ids = set(collection.get()['ids'])  # Assuming collection.get() returns stored IDs
        new_data = []

        for i, (embedding, row) in enumerate(zip(embeddings, rows_text)):
            row_id = str(hash(row))  # Hash-based unique ID to prevent duplicates
            if row_id not in existing_ids:
                new_data.append((row_id, embedding.tolist(), {"row": row}))

        if new_data:
            ids, embed_list, metadata_list = zip(*new_data)
            collection.add(ids=list(ids),embeddings=list(embed_list),metadatas=list(metadata_list))
        logger.info("Embeddings stored successfully.")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")

# Batch Retrieval function
def retrieve_relevant_rows_batch(queries, df, top_k=7):
    """Retrieve the most relevant rows based on a batch of queries."""
    try:
        if df.empty:
            return pd.DataFrame()
        
        query_embeddings = embedding_model.encode(queries, convert_to_tensor=True, device=DEVICE).cpu().numpy().tolist()
        results = collection.query(query_embeddings=query_embeddings, n_results=top_k)
        
        all_retrieved_texts = []
        for res in results["metadatas"]:
            all_retrieved_texts.extend([r["row"] for r in res])
        
        relevant_rows = df[df.astype(str).agg(' '.join, axis=1).isin(all_retrieved_texts)]
        return relevant_rows
    except Exception as e:
        logger.error(f"Error retrieving relevant rows for batch queries: {e}")
        return pd.DataFrame()

# Context preparation
def prepare_context(relevant_rows):
    """Format the retrieved rows into a readable context."""
    try:
        if relevant_rows.empty:
            return ""
        formatted_data = []
        for index, row in relevant_rows.iterrows():
            row_str = ", ".join(f"{col}: {value}" for col, value in row.items())
            formatted_data.append(f"Row {index}: {row_str}")
        return "\n".join(formatted_data)
    except Exception as e:
        logger.error(f"Error preparing context: {e}")
        return ""

# Answer generation using LLM
def answer_question_batch(queries, contexts):
    """Generate answers for a batch of queries."""
    try:
        answers = []
        for query, context in zip(queries, contexts):
            if not context:
                answers.append("No relevant context found.")
                continue
            prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            inputs['input_ids'] = inputs['input_ids'].to(dtype=torch.long)
            with torch.no_grad():
                outputs = llm_model.generate(
                        **inputs, max_new_tokens=200, temperature=0.01, top_p=0.9, do_sample=True,
                        pad_token_id=llm_tokenizer.eos_token_id
                    )
            answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
            answers.append(answer)
        return answers
    except Exception as e:
        logger.error(f"Error generating answers for batch: {e}")
        return ["Error generating answer."] * len(queries)

# Function to handle PDF processing and queries
def process_pdf_and_queries(pdf_file, dropdown_queries, custom_queries):
    """Main function to handle PDF processing and query answering."""
    try:
        queries =  custom_queries.split(",") if custom_queries else dropdown_queries
        
        pl_page = extract_page_no(pdf_file,keywords)


        # Initialize ChromaDB and models if not done yet
        if collection is None:
            initialize_chroma_client()
        if embedding_model is None or llm_model is None or llm_tokenizer is None:
            load_models()

        with ThreadPoolExecutor() as executor:
            pnl_table_future = executor.submit(extract_profit_loss_tables, pdf_file, pl_page)
            pnl_table = pnl_table_future.result()

            embedding_future = executor.submit(embed_and_store, pnl_table)
            embedding_future.result()  # Ensure embeddings are stored

        # Retrieve relevant rows for all queries in parallel
        all_relevant_rows = pd.DataFrame()
        answers = ""

        with ThreadPoolExecutor() as executor:
            relevant_rows_future = executor.submit(retrieve_relevant_rows_batch, queries, pnl_table)
            relevant_rows = relevant_rows_future.result()

            
            contexts = [prepare_context(relevant_rows)] * len(queries)

            answer_futures = executor.submit(answer_question_batch, queries, contexts)
         
            answers = "\n".join(answer_futures.result())
       

            # Collect the relevant rows with separators for display
            separator = pd.DataFrame({col: ['---'] for col in relevant_rows.columns})
            all_relevant_rows = pd.concat([all_relevant_rows, separator, relevant_rows], ignore_index=True)
     
        return all_relevant_rows, answers
    except Exception as e:
        logger.error(f"Error processing PDF and queries: {e}")
        return pd.DataFrame(), "Error processing the queries."

# Gradio Interface setup
def build_gradio_interface():
    """Build the Gradio interface for the application."""
    example_pdf_path = "Sample Financial Statement.pdf"
    return gr.Interface(
        fn=process_pdf_and_queries,
        inputs=[
            gr.File(label="Upload PDF (P&L Statement)"),
            gr.Dropdown(
                label="Select multiple sample queries",
                choices=[
                    "What is the gross profit for Q3 2024?",
                    "What is the net income for 2024?",
                    "How much was the operating income for Q2 2024?",
                    "Show the operating margin for the past 6 months.",
                    "What are the total expenses for Q2 2023?"
                ],
                type="value",
                multiselect=True
            ),
            gr.Textbox(
                label="Or type custom queries (separate by comma) within quotes",
                placeholder="e.g., 'What is the net income for Q4 2024?, What is the operating margin for Q3 2024?'",
                lines=1,
                interactive=True
            )
        ],
        outputs=[
            gr.Dataframe(label="Retrieved financial data segments separated by ---", type="pandas", interactive=False),
            gr.Textbox(label="Answers", lines=10, interactive=False)
        ],
        title="Interactive Financial Data QA Bot",
        description="Upload a PDF with a P&L table and ask financial queries.",
        allow_flagging="never",
        examples=EXAMPLES,
    )

if __name__ == "__main__":
    # Initialize necessary components only once at the start
    initialize_chroma_client()
    load_models()
    
    iface = build_gradio_interface()
    iface.launch()
