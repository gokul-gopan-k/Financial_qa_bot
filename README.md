# Interactive Financial Data QA Bot

This project provides an interactive QA bot for extracting insights from Profit & Loss (P&L) financial statements in PDF format. It utilizes advanced machine learning models for document parsing, table extraction, and question answering.

## Features

- Upload a PDF containing financial data (Profit & Loss statement).
- Ask specific financial queries (e.g., gross profit, net income, operating margin).
- Retrieve relevant rows of financial data and generate answers based on the query context.
- Uses ChromaDB for storing and retrieving embeddings of financial data rows.
- Powered by models like all-MiniLM-L6-v2 for embeddings and deepseek-coder-1.3b-instruct for generating answers.

## Setup Instructions

### Prerequisites
Ensure that you have the following installed on your system:

- Python 3.8 or higher
- Pip (for installing dependencies)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/gokul-gopan-k/financial_qa_bot.git
cd financial_qa_bot
```
2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
.\venv\Scripts\activate  # For Windows
```

3. **Install dependencies:**
Install all the required dependencies using pip:
```bash
pip install -r requirements.txt
```

 **Set up the models:**
The SentenceTransformer and HuggingFace models are automatically downloaded during execution. Ensure you have internet access when running the application for the first time.

 **ChromaDB Setup**
ChromaDB is used for storing the embeddings of the financial data. The database will be created in the project directory (./chroma_db), which will be used for subsequent queries. Ensure that the folder is writeable by the application.

 **Example PDF**
You can use a sample PDF file (Sample Financial Statement.pdf) which is expected to be uploaded for querying. Ensure the PDF contains relevant tables (e.g., Profit & Loss statements).

### Usage Instructions

1. **Running the Bot:**
Run the following command to start the application:
```bash
python app.py
```

2. **Interacting with the Interface:**
After running the above command, a Gradio interface will open in your browser.

- Upload PDF: Upload a PDF file containing the financial statement.
- Select Queries: Choose multiple pre-defined queries from the dropdown or type your own custom queries.
- View Results: The bot will process the PDF, extract relevant financial data, and display the results (including a table and the answers to your queries).
- Example Queries
"What is the gross profit for Q3 2024?"
"What is the net income for 2024?"
"How much was the operating income for Q2 2024?"
"Show the operating margin for the past 6 months."
"What are the total expenses for Q2 2023?"
The bot will extract the relevant rows from the PDF and generate answers for the selected queries.


### Troubleshooting
- Model Errors: Ensure the device supports MPS if you're using Apple's hardware (Mac). For CUDA, you will need a compatible GPU.
- Table Extraction Issues: If the tables are not properly extracted, consider adjusting the camelot parameters or checking if the table structure is consistent.
