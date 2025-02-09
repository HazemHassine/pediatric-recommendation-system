import pdfplumber  # PDF text extraction library
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time  # For adding sleep between requests

# Load environment variables (API key)
load_dotenv()  # Load environment variables from .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get the Gemini response
def get_gemini_response(input_prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    return response.text

# Prompt template for the input to the model
input_prompt_template = """
Given the following text from a PDF page:

{text}

Please extract the most important key words of this.
Return the result in the following structure:
{{"key_words":"<key_word>"}}
"""

# Load your DataFrame (replace with actual data or file path)
df = pd.read_csv("out_put.csv")

# Ensure the 'key_words' column exists in the DataFrame
if 'key_words' not in df.columns:
    df['key_words'] = None

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    content = row['content']
    
    # Skip rows with missing content or already processed key_words
    if pd.isnull(content) or pd.notnull(row['key_words']):
        continue

    # Print the index of the current request
    print(f"Processing row {index}...")
    
    # Format the input prompt with the content
    input_prompt = input_prompt_template.format(text=content)
    
    # Get the response from Gemini API
    try:
        response = get_gemini_response(input_prompt)
        # Store the response in the DataFrame
        df.at[index, 'key_words'] = response
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df.at[index, 'key_words'] = None  # Store None in case of error

    # Add a sleep time of 5 seconds between requests
    time.sleep(5)

# Save the DataFrame to a CSV file
output_csv_path = "output_with_keywords.csv"
df.to_csv(output_csv_path, index=False)

print(f"Data with keywords saved to {output_csv_path}")
