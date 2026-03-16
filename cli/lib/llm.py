import os
from dotenv import load_dotenv
from google import genai
from numpy import promote_types
from .search_utils import PROMPTS_DIR

load_dotenv()
MODEL = "gemini-2.5-flash"

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def generate_content(prompt, query):
    """
    Generate content using the Gemini LLM based on a provided prompt and query.

    The function formats the `prompt` template with the supplied `query`, sends the combined 
    prompt to the Gemini model specified by `MODEL`, and returns the generated textual response.

    Args:
        prompt (str): A prompt template string, expected to contain a `{query}` placeholder 
                      that will be filled in with the actual search/query terms.
        query (str): The search or question string to insert into the prompt.

    Returns:
        str: Generated text content returned by the Gemini model in response to the prompt.
    """
    prompt = prompt.format(query=query)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


def correct_spelling(query):
    """
    Use an LLM (Gemini) to correct spelling and grammar for a provided query string.

    This function reads a spelling correction prompt template from 'prompts/spelling.md',
    feeds the original query and template to the LLM, and returns the LLM's suggested correction.

    Args:
        query (str): The input query string potentially containing spelling or grammatical errors.

    Returns:
        str: The LLM's suggested spelling/grammar corrected version of the input query.
    """
    with open(PROMPTS_DIR / "spelling.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query)

def rewrite_query(query):
    """
    Use an LLM (Gemini) to rewrite a provided query string.

    This function reads a rewrite prompt template from 'prompts/rewrite.md',
    feeds the original query and template to the LLM, and returns the LLM's suggested rewrite.
    """
    with open(PROMPTS_DIR / "rewrite.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query)