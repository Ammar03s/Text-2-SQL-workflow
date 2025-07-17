# Text-to-SQL Workflow with Smart MongoDB Integration

A workflow that converts natural language questions into SQL queries, excute them, and show viusalizations of the results. Also, intelligently manages them in MongoDB based on both relevancy and safety checks.

## The Workflow

```
User Question → Relevance Check → SQL Generation → Safety Check → Execution → Visualization → Smart MongoDB Storage
```

## Database Logic

The system decides what to store in MongoDB:

- **Relevant + Safe Query**: Stores query, SQL, execution results, and visualizations
- **Relevant + Unsafe Query**: Stores query and SQL but marks as dangerous (no execution)  
- **Irrelevant Query**: Stores query only with "not relevant" flag (no SQL generation)

All entries include metadata: timestamp, model used, relevance status, safety status.

## Demo

https://github.com/user-attachments/assets/textsql-demo.mp4

## Features

- Natural language to SQL conversion using LLM models
- Smart relevance detection (checks if question matches your data)
- SQL safety validation (prevents DROP, DELETE, etc.)
- CSV data upload and analysis
- Automatic chart generation from query results (When possible)
- Complete query history in MongoDB
- Support for multiple LLM models (Testing purposes)

## Test Data

You can test with this sample CSV: [Insert your CSV link here]

## Requirements

- **Python 3.8+**
- **MongoDB**
- **Ollama** with models installed

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama models:**
   ```bash
   ollama pull mistral:7b
   ollama pull llama3.2:3b
   ```

4. **Start MongoDB** (if local):
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   # or just run the MongoDB app
   ```

5. **Run the application:**
   ```bash
   streamlit run text-2-sql.py
   ```

## Configuration

- **Change LLM models**: Edit the selectbox options in `main()` function
- **MongoDB settings**: Modify `mongodb_uri`, `db_name`, `db_collection_name` at the top of the file

## How It Works

1. Upload your CSV file
2. Ask questions in natural language
3. System checks if question is relevant to your data
4. If relevant, generates SQL and checks safety
5. If safe, executes SQL and creates visualizations  
6. Everything gets stored in MongoDB with smart categorization
7. View history to see all past queries and results 
