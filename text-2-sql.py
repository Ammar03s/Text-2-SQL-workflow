import json
import re
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pymongo import MongoClient
from datetime import datetime
import sqlite3
import plotly.express as px


mongodb_uri = "mongodb://localhost:27017/"
db_name = "Text2SQL_DB"
db_collection_name = "Querizz4"


TEMPLATES = {
    "sql_generator": """
        You are a SQL generator. When given a schema and a user question, you MUST output only the SQL statement—nothing else. 
        No explanation is needed. 
        Use correct table/column names from the schema.

        Schema: {schema}
        User question: {query}
        Output (SQL only):
    """,
    "judge": """
        You are evaluating if a question can be answered using the provided database schema.
        
        Database schema fields:
        {schema}
        
        User question:
        "{query}"
        
        A question is RELEVANT ONLY if it directly asks about data in the schema fields.
        A question is NOT_RELEVANT if it asks about topics not present in the schema.
        
        Respond with ONLY one word: "RELEVANT" or "NOT_RELEVANT".
    """
}

def connect_mongodb():
    try:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        if db_collection_name not in db.list_collection_names(): #Was querizz4
            db.create_collection(db_collection_name) #was querizz4
        return client, db, db[db_collection_name], True #was db.Querizz4
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None, None, None, False

def storing(collection, query, sql, schema, relevance, model_name, query_output=None):
    query_documentation = {
        "query_text": query,
        "sql_result": sql if sql else "Not generated (irrelevant query)",
        "timestamp": datetime.now(),
        "schema_used": schema,
        "is_safe": True if not sql else is_it_safe(sql),
        "is_relevant": relevance,
        "model_used": model_name,
        "query_output": query_output if query_output is not None else "No output or not executed"
    }
    collection.insert_one(query_documentation)

def extract_schema(source, is_mongo_db=True):
    if is_mongo_db:
        schema = {}
        for collection_name in source.list_collection_names():
            if collection_name == db_collection_name: #was querizz4
                continue
            sample = source[collection_name].find_one()
            if sample:
                schema[collection_name] = list(sample.keys())
    else:
        schema = {"csv_data": list(source.columns)}
    return json.dumps(schema)

def is_it_safe(sql):
    return not re.search(r"\b(drop|delete|update|alter|truncate)\b", sql, re.IGNORECASE)

def clean_up(text):
    return re.sub(r"```(sql|sqlite)?", "", text, flags=re.IGNORECASE).strip("`").strip()

def judgy_judge(query, schema, model_name): #to verify if the query is relevant to the schema
    schema_data = json.loads(schema)
    all_fields = []
    for fields in schema_data.values():
        all_fields.extend(fields)

    query_tokens = re.findall(r'\b\w+\b', query.lower())
    field_tokens = [field.lower() for field in all_fields]
    field_match = any(token in field_tokens for token in query_tokens)

    data_patterns = [
        "how many", "how much", "count", "number of", "sum", "total", "average", "mean", "median",
        "maximum", "minimum", "max", "min", "total number", "total amount", "total count",
        "aggregate", "tally", "compute", "calculate", "total value", "accumulate", "overall",
        "highest", "lowest", "biggest", "smallest", "peak", "least", "most", "top", "bottom",
        "per", "percent", "percentage", "ratio", "proportion", "distribution", "frequency",
        "how often"
    ]
    pattern_match = any(pattern in query.lower() for pattern in data_patterns)
    table_match = any(table.lower() in query.lower() for table in schema_data.keys())

    if field_match or (pattern_match and (field_match or table_match)):
        return True

    model = OllamaLLM(model = model_name, temperature = 0.2) #since default is 0.7
    prompt = ChatPromptTemplate.from_template(TEMPLATES["judge"])
    chain = prompt | model
    result = chain.invoke({"query": query, "schema": schema}).strip().upper()
    return "RELEVANT" in result and "NOT_RELEVANT" not in result

def generate_sql(query, schema, model_name):
    model = OllamaLLM(model = model_name, temperature = 0.2) #since default is 0.7
    prompt = ChatPromptTemplate.from_template(TEMPLATES["sql_generator"])
    chain = prompt | model
    sql = chain.invoke({"query": query, "schema": schema})
    return clean_up(sql)

def run_sql_on_df(df, sql):
    conn = sqlite3.connect(":memory:")
    df.to_sql("csv_data", conn, index=False, if_exists="replace")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return result, columns
    except Exception as e:
        return str(e), None

def to_dataframe(result, columns):
    if not result or not columns:
        return None
    return pd.DataFrame(result, columns=columns)

def history(collection):
    with st.expander("Show History"):
        history = list(collection.find({}).sort("timestamp", -1).limit(5))
        for item in history:
            with st.container():
                st.markdown(f"Query:\n{item['query_text']}")
                if item.get('is_relevant', True):
                    st.code(item["sql_result"], language="sql")
                else:
                    st.info(item["sql_result"])
                if "query_output" in item and isinstance(item["query_output"], list) and item["query_output"]:
                    st.markdown("Stored Query Output:")
                    try:
                        st.dataframe(pd.DataFrame(item["query_output"]))
                    except Exception:
                        st.text(item["query_output"])
                st.markdown(f"""    
                Model: {item.get('model_used', 'Unknown')}  
                Relevant: {'Yes' if item.get('is_relevant', True) else 'No'}  
                Safe SQL: {'Yes' if item['is_safe'] else 'No'}
                """)
                st.divider()

def load_uploaded_csv(uploaded_file, db, connected):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.subheader("Data Head")
            st.dataframe(df.head())
            schema = extract_schema(df, is_mongo_db=False)
            return df, schema
        except Exception as e:
            st.error(f"Error loading the data: {e}")
    elif connected:
        schema = extract_schema(db)
        return None, schema
    return None, "{}"

def render_chart(df_out):
    st.subheader("Charts")
    try:
        if df_out.shape[1] >= 2:
            x_col = df_out.columns[1]
            y_col = df_out.columns[0]

            if pd.api.types.is_numeric_dtype(df_out[y_col]):
                fig = px.bar(
                    df_out,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    template="plotly_dark"
                )
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig)
            else:
                st.info("Second column is not numeric — can't generate chart.")
        else:
            st.info("Need at least two columns to generate a chart.")
    except Exception as e:
        st.error(f"Error generating chart: {e}")

def handle_query(query, schema, model_name, df, uploaded_file, query_collection):
    is_relevant = judgy_judge(query, schema, model_name)
    if not is_relevant:
        st.warning("This query doesn't seem to be related to the available data.")
        st.info("SQL generation skipped due to irrelevancy.")
        storing(query_collection, query, None, schema, is_relevant, model_name)
        st.success("Query stored in database for analysis.")
        return

    sql = generate_sql(query, schema, model_name)
    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    if not is_it_safe(sql):
        st.error("Dangerous SQL detected! This query has not been stored in the database.")
        storing(query_collection, query, sql, schema, is_relevant, model_name, query_output="Dangerous query (not executed)")
        return

    if uploaded_file is not None and df is not None:
        result, columns = run_sql_on_df(df, sql)
        if isinstance(result, str):
            st.error(f"SQL Execution Error: {result}")
            storing(query_collection, query, sql, schema, is_relevant, model_name, query_output="SQL execution failed")
        else:
            df_out = to_dataframe(result, columns)
            output_data = df_out.to_dict(orient="records") if df_out is not None else None
            storing(query_collection, query, sql, schema, is_relevant, model_name, query_output=output_data)

            if df_out is not None and not df_out.empty:
                st.subheader("SQL Query Result")
                st.dataframe(df_out)
                render_chart(df_out)
            else:
                st.info("No results returned from query.")
    else:
        storing(query_collection, query, sql, schema, is_relevant, model_name)

def main():
    st.set_page_config(page_title="Text 2 SQL with MongoDB", layout="centered")
    st.title("Text 2 SQL with MongoDB")

    client, db, query_collection, connected = connect_mongodb()
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    df, schema = load_uploaded_csv(uploaded_file, db, connected)
    model_name = st.selectbox("Choose a Model", ["mistral:7b", "llama3.2:3b"])
    query = st.text_area("Enter a question related to the data")

    with st.expander("View Data Schema"):
        st.json(json.loads(schema))

    if connected:
        history(query_collection)

    if st.button("Generate") and query:
        if not connected:
            st.error("Mongo connection failed")
        else:
            handle_query(query, schema, model_name, df, uploaded_file, query_collection)

    if client:
        client.close()

if __name__ == "__main__":
    main()




# # reversed judgy_judge
# def judgy_judge(query, schema, model_name):  # to verify if the query is relevant to the schema
#     # Step 1: Ask the LLM first
#     model = OllamaLLM(model=model_name, temperature=0.2)
#     prompt = ChatPromptTemplate.from_template(TEMPLATES["judge"])
#     chain = prompt | model
#     result = chain.invoke({"query": query, "schema": schema}).strip().upper()

#     if "RELEVANT" in result and "NOT_RELEVANT" not in result:
#         return True  # LLM confirms relevance

#     # Step 2: Fallback to manual heuristics if LLM says NOT_RELEVANT
#     schema_data = json.loads(schema)
#     all_fields = []
#     for fields in schema_data.values():
#         all_fields.extend(fields)

#     query_tokens = re.findall(r'\b\w+\b', query.lower())
#     field_tokens = [field.lower() for field in all_fields]
#     field_match = any(token in field_tokens for token in query_tokens)

#     data_patterns = [
#         "how many", "how much", "count", "number of", "sum", "total", "average", "mean", "median",
#         "maximum", "minimum", "max", "min", "total number", "total amount", "total count",
#         "aggregate", "tally", "compute", "calculate", "total value", "accumulate", "overall",
#         "highest", "lowest", "biggest", "smallest", "peak", "least", "most", "top", "bottom",
#         "per", "percent", "percentage", "ratio", "proportion", "distribution", "frequency",
#         "how often"
#     ]
#     pattern_match = any(pattern in query.lower() for pattern in data_patterns)
#     table_match = any(table.lower() in query.lower() for table in schema_data.keys())

#     return field_match or (pattern_match and (field_match or table_match))

# #not reversed
# def judgy_judge(query, schema, model_name): #to verify if the query is relevant to the schema
#     schema_data = json.loads(schema)
#     all_fields = []
#     for fields in schema_data.values():
#         all_fields.extend(fields)

#     query_tokens = re.findall(r'\b\w+\b', query.lower())
#     field_tokens = [field.lower() for field in all_fields]
#     field_match = any(token in field_tokens for token in query_tokens)

#     data_patterns = [
#         "how many", "how much", "count", "number of", "sum", "total", "average", "mean", "median",
#         "maximum", "minimum", "max", "min", "total number", "total amount", "total count",
#         "aggregate", "tally", "compute", "calculate", "total value", "accumulate", "overall",
#         "highest", "lowest", "biggest", "smallest", "peak", "least", "most", "top", "bottom",
#         "per", "percent", "percentage", "ratio", "proportion", "distribution", "frequency",
#         "how often"
#     ]
#     pattern_match = any(pattern in query.lower() for pattern in data_patterns)
#     table_match = any(table.lower() in query.lower() for table in schema_data.keys())

#     if field_match or (pattern_match and (field_match or table_match)):
#         return True

#     model = OllamaLLM(model = model_name, temperature = 0.2) #since default is 0.7
#     prompt = ChatPromptTemplate.from_template(TEMPLATES["judge"])
#     chain = prompt | model
#     result = chain.invoke({"query": query, "schema": schema}).strip().upper()
#     return "RELEVANT" in result and "NOT_RELEVANT" not in result

# describe() head() tail() value_counts() dtypes() missing_values() maybe a random sample
