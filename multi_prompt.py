import gradio as gr
import os
import google.generativeai as genai
from openai import OpenAI
import anthropic
import threading

# --- API Key Configuration ---
# It's recommended to set these as environment variables for security.
# Example: export OPENAI_API_KEY='your_openai_key'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- LLM Client Initialization ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using a more recent and available model
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_model = None

try:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    anthropic_client = None

# --- Functions to get responses from each LLM ---

def get_gemini_response(prompt, results):
    """
    Gets a response from the Gemini model.
    """
    try:
        if not gemini_model:
            results['gemini'] = "Gemini client not initialized. Please check your API key."
            return
        response = gemini_model.generate_content(prompt)
        results['gemini'] = response.text
    except Exception as e:
        results['gemini'] = f"Error from Gemini: {e}"

def get_gpt_response(prompt, results):
    """
    Gets a response from the GPT model.
    """
    try:
        if not openai_client:
            results['gpt'] = "OpenAI client not initialized. Please check your API key."
            return
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        results['gpt'] = response.choices[0].message.content
    except Exception as e:
        # Provide a more user-friendly message for quota errors
        if "insufficient_quota" in str(e):
             results['gpt'] = "Error from GPT: You have exceeded your API quota. Please check your OpenAI billing details."
        else:
            results['gpt'] = f"Error from GPT: {e}"

def get_claude_response(prompt, results):
    """
    Gets a response from the Claude model.
    """
    try:
        if not anthropic_client:
            results['claude'] = "Anthropic client not initialized. Please check your API key."
            return
        response = anthropic_client.messages.create(
            # Using a more recent and available model
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        results['claude'] = response.content[0].text
    except Exception as e:
        results['claude'] = f"Error from Claude: {e}"

def get_llm_responses_and_judge(question):
    """
    Main function to get responses from all three LLMs and then have a judge LLM evaluate them.
    """
    if not all([OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY]):
        # Clearer error message for missing keys
        missing_keys = []
        if not OPENAI_API_KEY: missing_keys.append("OpenAI")
        if not GOOGLE_API_KEY: missing_keys.append("Google")
        if not ANTHROPIC_API_KEY: missing_keys.append("Anthropic")
        return f"Missing API key(s) for: {', '.join(missing_keys)}. Please configure them as environment variables.", "", "", ""


    results = {'gemini': '', 'gpt': '', 'claude': ''}

    # Create and start threads for each LLM
    gemini_thread = threading.Thread(target=get_gemini_response, args=(question, results))
    gpt_thread = threading.Thread(target=get_gpt_response, args=(question, results))
    claude_thread = threading.Thread(target=get_claude_response, args=(question, results))

    gemini_thread.start()
    gpt_thread.start()
    claude_thread.start()

    # Wait for all threads to complete
    gemini_thread.join()
    gpt_thread.join()
    claude_thread.join()

    gemini_response = results['gemini']
    gpt_response = results['gpt']
    claude_response = results['claude']

    # --- LLM as a Judge ---
    judge_prompt = f"""
    You are an expert LLM evaluator. You will be given a user's question and three responses from three different LLMs: Gemini, GPT, and Claude.

    Your task is to analyze the three responses and determine which one is the best.

    **User's Question:**
    "{question}"

    **LLM Responses:**

    **Gemini:**
    "{gemini_response}"

    **GPT:**
    "{gpt_response}"

    **Claude:**
    "{claude_response}"

    **Evaluation Criteria:**
    1.  **Helpfulness and Relevance:** Does the response directly answer the user's question?
    2.  **Accuracy:** Is the information provided correct and factual?
    3.  **Clarity and Conciseness:** Is the response easy to understand and to the point?
    4.  **Completeness:** Does the response cover all aspects of the user's question?

    **Your Response:**
    Please provide your final verdict in the following format:

    **Winning Model:** [Name of the winning model: Gemini, GPT, or Claude]
    **Justification:** [A detailed explanation of why you chose the winning model, referencing the evaluation criteria.]
    """

    try:
        # Also check if the judge model itself has an error
        if "Error from" in gemini_response:
             judge_response = "Could not get a verdict as the Judge LLM (Gemini) failed to generate a response."
        else:
            judge_response = gemini_model.generate_content(judge_prompt).text
    except Exception as e:
        judge_response = f"Error from Judge (Gemini): {e}"


    return judge_response, gemini_response, gpt_response, claude_response

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LLM Response Comparator")
    gr.Markdown("Enter a question to get responses from Gemini, GPT, and Claude. A judge LLM will then evaluate the responses.")

    with gr.Row():
        question_input = gr.Textbox(label="Your Question", lines=3, placeholder="e.g., Explain the theory of relativity in simple terms.")

    submit_button = gr.Button("Get Responses and Judge")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Judge's Verdict")
            judge_output = gr.Markdown(label="Judge's Verdict")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Gemini's Response")
            gemini_output = gr.Textbox(label="Gemini", lines=10, interactive=False)
        with gr.Column():
            gr.Markdown("## GPT's Response")
            gpt_output = gr.Textbox(label="GPT", lines=10, interactive=False)
        with gr.Column():
            gr.Markdown("## Claude's Response")
            claude_output = gr.Textbox(label="Claude", lines=10, interactive=False)

    submit_button.click(
        fn=get_llm_responses_and_judge,
        inputs=question_input,
        outputs=[judge_output, gemini_output, gpt_output, claude_output]
    )

if __name__ == "__main__":
    demo.launch()
