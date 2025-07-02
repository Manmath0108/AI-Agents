# Core libraries
import os
import re
import json
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Database libraries
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from sqlalchemy import create_engine, text, inspect

# AI libraries
import openai
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Utility libraries
from datetime import datetime
import time

print("✅ All libraries imported successfully!")

class PostgreSQLConnection:
    """
    A robust PostgreSQL connection handler with error handling and connection management.
    """
    
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.engine = None
        
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            # Create connection string
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            
            # Create SQLAlchemy engine
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                print(f"✅ Connected to PostgreSQL!")
                print(f"📊 Database: {self.database}")
                print(f"🔧 Version: {version[:50]}...")
                
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    
    def execute_query(self, query, return_df=True):
        """Execute a SQL query and return results"""
        try:
            if return_df:
                df = pd.read_sql_query(query, self.engine)
                return df
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    return result.fetchall()
                    
        except Exception as e:
            print(f"❌ Query execution failed: {str(e)}")
            return None
    
    def get_table_info(self):
        """Get information about all tables in the database"""
        try:
            inspector = inspect(self.engine)
            tables_info = {}
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                tables_info[table_name] = {
                    'columns': [col['name'] for col in columns],
                    'column_details': columns
                }
                
            return tables_info
            
        except Exception as e:
            print(f"❌ Failed to get table info: {str(e)}")
            return None

# Initialize database connection
DB_CONFIG = {
    'host': '54.251.218.166',
    'port': 5432,
    'database': 'dummy',
    'user': 'rajesh',
    'password': 'rajesh123'
}

# Create database connection
db = PostgreSQLConnection(**DB_CONFIG)
connection_success = db.connect()

db.execute_query("SELECT * FROM actor a")

# Check if connection was successful and fetch available tables
if connection_success:
     tables_query = """
    SELECT 
        table_name,
        table_schema,
        table_type
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """
     tables_df = db.execute_query(tables_query)
     print("Available Tables")
     print("="*50)
     for idx, row in tables_df.iterrows():
        print(f"  {idx+1}. {row['table_name']} ({row['table_type']})")
     print(f"\n🔢 Total tables found: {len(tables_df)}")


else:
     print("trouble to connect with database")

def explore_table_structure(table_name, limit=5):
    columns_query = f"""
    SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM information_schema.columns 
    WHERE table_name = '{table_name}'
    ORDER BY ordinal_position;
    """
    columns_df = db.execute_query(columns_query)
    print(f"🔍 Table: {table_name}")
    print("=" * 60)
    print("📊 Column Structure:")
    for idx, row in columns_df.iterrows():
        nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
        default = f" DEFAULT {row['column_default']}" if row['column_default'] else ""
        print(f"  • {row['column_name']}: {row['data_type']} ({nullable}){default}")
    # Get sample data
    sample_query = f"SELECT * FROM {table_name} LIMIT {limit};"
    sample_df = db.execute_query(sample_query)
    
    print(f"\n📝 Sample Data (first {limit} rows):")
    if sample_df is not None and not sample_df.empty:
        print(sample_df.to_string())
    else:
        print("  No data found or query failed")
    
    print("\n" + "=" * 60)
    return columns_df, sample_df


if connection_success and not tables_df.empty:
    # Take first few tables to explore
    tables_to_explore = tables_df['table_name'].head(3).tolist()
    
    for table in tables_to_explore:
        try:
            explore_table_structure(table)
            print()
        except Exception as e:
            print(f"❌ Error exploring {table}: {str(e)}")
            print()

# SchemaContextBuilder: A class to build context about database schema for AI models
# This class will help AI models generate accurate SQL queries by providing detailed schema information
class SchemaContextBuilder:
    """
    Builds context about database schema for AI models to generate accurate SQL queries
    """
    def __init__(self, db_connection):
        self.db = db_connection
        self.schema_cache = {}
        self.build_full_schema_context()

    def build_full_schema_context(self):
        """Build complete schema context for all tables"""
        
        # Get all tables
        tables_query = """
        SELECT table_name, table_schema 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        tables_df = self.db.execute_query(tables_query)
        if tables_df is None:
            return
        for _, row in tables_df.iterrows():
            table_name = row['table_name']
            self.schema_cache[table_name] = self.get_table_schema(table_name)

    def get_table_schema(self, table_name):
        """Get detailed schema for a specific table"""

        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
        """
        columns_df = self.db.execute_query(columns_query)

        if columns_df is None:
            return None
        
        # Get foreign key relationships
        fk_query = f"""
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = '{table_name}';
        """

        fk_df = self.db.execute_query(fk_query)

        # Build schema info
        schema_info = {
            'table_name': table_name,
            'columns': [],
            'foreign_keys': []
        }

        for _, col in columns_df.iterrows():
            col_info = {
                'name': col['column_name'],
                'type': col['data_type'],
                'nullable': col['is_nullable'] == 'YES',
                'default': col['column_default'],
                'max_length': col['character_maximum_length']
            }
            schema_info['columns'].append(col_info)

        if fk_df is not None and not fk_df.empty:
            for _, fk in fk_df.iterrows():
                fk_info = {
                    'column': fk['column_name'],
                    'references_table': fk['foreign_table_name'],
                    'references_column': fk['foreign_column_name']
                }
                schema_info['foreign_keys'].append(fk_info)
        
        return schema_info
    

    def get_relevant_tables(self, query_text):
        """Identify tables that might be relevant to the query"""
        query_lower = query_text.lower()
        relevant_tables = []
        
        for table_name in self.schema_cache.keys():
            # Check if table name appears in query
            if table_name.lower() in query_lower:
                relevant_tables.append(table_name)
                continue
                
            # Check if any column names appear in query
            schema = self.schema_cache[table_name]
            if schema:
                for col in schema['columns']:
                    if col['name'].lower() in query_lower:
                        relevant_tables.append(table_name)
                        break
        
        # If no specific tables found, return first few tables
        if not relevant_tables:
            relevant_tables = list(self.schema_cache.keys())[:5]
            
        return relevant_tables
    
    def build_context_for_query(self, query_text):
        """Build focused context for a specific query"""
        relevant_tables = self.get_relevant_tables(query_text)
        
        context = f"""
DATABASE SCHEMA INFORMATION:
Database: {self.db.database}
Relevant Tables for Query: "{query_text}"

"""
        
        for table_name in relevant_tables:
            schema = self.schema_cache.get(table_name)
            if not schema:
                continue
                
            context += f"TABLE: {table_name}\n"
            context += "Columns:\n"
            
            for col in schema['columns']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                context += f"  - {col['name']}: {col['type']} ({nullable})\n"
            
            if schema['foreign_keys']:
                context += "Foreign Keys:\n"
                for fk in schema['foreign_keys']:
                    context += f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n"
            
            context += "\n"
        
        return context
    

# Initialize schema builder
if connection_success:
    schema_builder = SchemaContextBuilder(db)
    print("✅ Schema context builder initialized!")
    print(f"📊 Cached schema for {len(schema_builder.schema_cache)} tables")
else:
    print("❌ Cannot initialize schema builder - no database connection")




# AI Configuration
# You'll need to set your API keys here
# Option 1: Set as environment variables
# export OPENAI_API_KEY="your-openai-key"
# export ANTHROPIC_API_KEY="your-anthropic-key"

# Option 2: Set directly in code (less secure)
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

def get_available_models():
    """Check which AI models are available based on API keys"""
    models = {}
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            models["openai"] = ChatOpenAI(
                model="gpt-4o-mini",  # Cost-effective but powerful
                temperature=0.1,      # Low temperature for consistent SQL generation
                max_tokens=1000
            )
            print("✅ OpenAI GPT-4o-mini available")
        except Exception as e:
            print(f"❌ OpenAI setup failed: {str(e)}")
    
    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            models["anthropic"] = ChatAnthropic(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                temperature=0.1,
                max_tokens=1000
            )
            print("✅ Anthropic Claude available")
        except Exception as e:
            print(f"❌ Anthropic setup failed: {str(e)}")
    
    if not models:
        print("⚠️  No AI models available. Please set your API keys.")
        print("   You can use OpenAI, Anthropic, or other compatible models.")
        print("   For this tutorial, we'll create a mock model for demonstration.")
        
        # Create a mock model for demonstration
        class MockModel:
            def invoke(self, messages):
                # Simple pattern matching for demo
                user_msg = messages[-1].content.lower()
                
                if "count" in user_msg and "table" in user_msg:
                    return type('Response', (), {'content': 'SELECT COUNT(*) FROM your_table_name;'})()
                elif "select" in user_msg or "show" in user_msg:
                    return type('Response', (), {'content': 'SELECT * FROM your_table_name LIMIT 10;'})()
                else:
                    return type('Response', (), {'content': 'SELECT * FROM your_table_name WHERE condition = value;'})()
        
        models["mock"] = MockModel()
        print("✅ Mock model created for demonstration")
    
    return models

# Initialize available models
available_models = get_available_models()
print(f"\n📊 Available models: {list(available_models.keys())}")


# SQLAgent: A class that converts natural language queries to SQL using AI models
class SQLAgent:
    """
    A robust SQL Agent that converts natural language queries to SQL using AI
    """
    
    def __init__(self, db_connection, schema_builder, ai_model, model_name="default"):
        self.db = db_connection
        self.schema_builder = schema_builder
        self.ai_model = ai_model
        self.model_name = model_name
        self.query_history = []

    def create_system_prompt(self):
        """Create a comprehensive system prompt for SQL generation"""
        
        system_prompt = """You are an expert PostgreSQL database analyst. Your job is to convert natural language questions into accurate, efficient SQL queries.

IMPORTANT GUIDELINES:
1. Always use proper PostgreSQL syntax
2. Use appropriate table and column names from the provided schema
3. Include proper JOINs when querying multiple tables
4. Use LIMIT clauses for exploratory queries to avoid large result sets
5. Handle NULL values appropriately
6. Use proper date/time functions for temporal queries
7. Return ONLY the SQL query, no explanations or markdown formatting
8. Make queries efficient and avoid unnecessary complexity

QUERY STRUCTURE:
- Use SELECT statements for data retrieval
- Use appropriate WHERE clauses for filtering
- Use GROUP BY and aggregation functions when needed
- Use ORDER BY for sorting results
- Use proper JOIN syntax for multi-table queries

COMMON PATTERNS:
- For counts: SELECT COUNT(*) FROM table_name WHERE condition
- For lists: SELECT column_name FROM table_name WHERE condition LIMIT 10
- For aggregations: SELECT column_name, AGG_FUNCTION(column) FROM table_name GROUP BY column_name
- For date ranges: WHERE date_column BETWEEN 'start_date' AND 'end_date'

Remember: Return only valid PostgreSQL SQL queries that can be executed directly."""

        return system_prompt
    
    def generate_sql_query(self, natural_language_query):
            
            """Convert natural language to SQL query"""
            
            try:
                # Build context for the query
                schema_context = self.schema_builder.build_context_for_query(natural_language_query)
                
                # Create messages for the AI model
                system_prompt = self.create_system_prompt()
                
                user_prompt = f"""
    {schema_context}

    Convert this natural language question to a PostgreSQL query:
    "{natural_language_query}"

    Return only the SQL query, nothing else.
    """
                
                # Prepare messages
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                # Generate SQL using AI model
                response = self.ai_model.invoke(messages)
                sql_query = response.content.strip()
                
                # Clean up the response (remove markdown formatting if present)
                sql_query = self.clean_sql_response(sql_query)
                
                # Store in history
                self.query_history.append({
                    'natural_language': natural_language_query,
                    'sql_query': sql_query,
                    'timestamp': datetime.now(),
                    'model': self.model_name
                })
                
                return sql_query
            except Exception as e:
                error_msg = f"Error generating SQL: {str(e)}"
                print(f"❌ {error_msg}")
                return None
    
    def clean_sql_response(self, sql_response):
        """Clean up SQL response from AI model"""
        
        # Remove markdown code blocks
        sql_response = re.sub(r'```sql\n', '', sql_response)
        sql_response = re.sub(r'```\n', '', sql_response)
        sql_response = re.sub(r'```', '', sql_response)
        
        # Remove extra whitespace
        sql_response = sql_response.strip()
        
        # Ensure it ends with semicolon
        if not sql_response.endswith(';'):
            sql_response += ';'
            
        return sql_response
    

    def validate_sql_query(self, sql_query):

        """Validate SQL query syntax without executing it"""
        
        try:
            # Use EXPLAIN to validate without executing
            explain_query = f"EXPLAIN {sql_query}"
            with self.db.engine.connect() as conn:
                conn.execute(text(explain_query))
            return True, "Query is valid"
            
        except Exception as e:
            return False, f"Query validation failed: {str(e)}"
        
    def execute_query_safely(self, sql_query, max_rows=100):
        """Execute SQL query with safety limits"""
        
        try:
            # Validate first
            is_valid, validation_msg = self.validate_sql_query(sql_query)
            
            if not is_valid:
                return None, validation_msg
            
            # Add LIMIT if not present for SELECT queries
            if sql_query.upper().strip().startswith('SELECT') and 'LIMIT' not in sql_query.upper():
                sql_query = sql_query.rstrip(';') + f' LIMIT {max_rows};'
            
            # Execute query
            result_df = self.db.execute_query(sql_query)
            
            if result_df is not None:
                return result_df, f"Query executed successfully. Returned {len(result_df)} rows."
            else:
                return None, "Query execution failed"
                
        except Exception as e:
            return None, f"Execution error: {str(e)}"
    


    def query(self, natural_language_query, execute=True, max_rows=100):
        """
        Main method to convert natural language to SQL and optionally execute it
        """
        
        print(f"🤔 Question: {natural_language_query}")
        print("=" * 80)
        
        # Generate SQL
        sql_query = self.generate_sql_query(natural_language_query)
        
        if sql_query is None:
            return None, None
        
        print(f"🔧 Generated SQL:")
        print(sql_query)
        print("-" * 40)
        
        if execute:
            # Execute the query
            result_df, message = self.execute_query_safely(sql_query, max_rows)
            print(f"📊 {message}")
            
            if result_df is not None and not result_df.empty:
                print("\n📋 Results:")
                print(result_df.to_string())
            
            return sql_query, result_df
        else:
            return sql_query, None


# Select a model for the agent
model_name = list(available_models.keys())[0]
selected_model = available_models[model_name]

sql_agent = SQLAgent(
        db_connection=db,
        schema_builder=schema_builder,
        ai_model=selected_model,
        model_name=model_name
    )


# Example usage
sql_agent.query("give me the list of Movies in which Nick worked")