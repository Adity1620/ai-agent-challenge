"""
AI Agent for Automated Bank Statement Parser Generation
Fixed version based on TypeScript/JavaScript logic with proper Google GenAI integration
"""

import os
import sys
import argparse
import json
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import TypedDict, Optional
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Agent State ---
class AgentState(TypedDict):
    target_bank: str
    pdf_path: str
    csv_path: str
    pdf_content: str
    csv_content: str
    plan: str
    generated_code: str
    reasoning_for_test: str
    test_passed: bool
    test_output: str
    error: str
    attempts: int
    max_attempts: int

# --- Helper Functions ---
def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from PDF using pdfplumber"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            content = ""
            for page in pdf.pages[:2]:  # First 2 pages for analysis
                if page.extract_text():
                    content += page.extract_text() + "\n\n"
            return content[:3000]  # Limit content for prompt
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_base_prompt(pdf_text: str, csv_text: str, bank_name: str) -> str:
    """Constructs the base prompt matching TypeScript logic"""
    csv_lines = csv_text.splitlines()
    csv_headers = csv_lines[0] if csv_lines else ""

    return f"""You are an expert Python developer specialized in data extraction. Your task is to act as an automated coding agent to write a Python parser for a bank statement PDF.

**Goal:** Create a Python script that parses a given bank statement PDF and outputs a pandas DataFrame with a specific schema.

**Inputs provided to you:**
1. **PDF Content:** Text extracted from a sample PDF for "{bank_name}".
2. **Target CSV:** A sample CSV file showing the exact desired output format, including headers and data types.

**Your task is to perform a single "plan -> generate -> test" cycle and return a JSON object with your results.**

**Constraints & Requirements:**
1. The parser must be a single Python function: `parse(pdf_path: str) -> pd.DataFrame`.
2. The script must use the `pandas` library and `pdfplumber` for PDF parsing.
3. The final DataFrame returned by your function MUST EXACTLY match the schema of the target CSV. The column headers are: `{csv_headers}`. Pay close attention to data types.
4. Your Python code should be complete, executable, and include all necessary imports.
5. Do not include example usage in the code string itself, only the function and imports.

**Here is the content:**

**PDF Text Content:**
---
{pdf_text}
---

**Target CSV Content:**
---
{csv_text}
---

**Final Step: Your Response**
After analyzing everything, you must respond with a single JSON object.

The JSON response MUST contain these four fields:
1. `plan`: A brief, step-by-step plan for how you will write the code.
2. `python_code`: A string containing the complete Python code for the parser.
3. `reasoning_for_test`: Critically evaluate your generated code. Explain your reasoning on whether this test would pass or fail.
4. `test_passed`: A boolean value (true or false) indicating if you predict your code will pass the test."""

def get_refinement_prompt(failed_code: str, error: str) -> str:
    """Constructs refinement prompt for failed attempts"""
    return f"""
The previous attempt to generate the parser failed the self-evaluation test.

**Previous Code (Failed):**
```python
{failed_code}
```

**Reason for Failure / Analysis:**
{error}

**Your Task:**
Analyze the reason for failure and the previous code. Generate a new, corrected version of the Python parser. The goal remains the same: the `parse` function must produce a DataFrame matching the target CSV. Create a new plan and generate the corrected code."""

# --- Agent Nodes ---
def plan_and_generate_code_step(state: AgentState):
    """Plans and generates code using Google GenAI with proper JSON schema"""
    print("--- 📝 PLANNING & 💻 GENERATING CODE ---")

    # Setup Google GenAI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        state['error'] = "GOOGLE_API_KEY or GEMINI_API_KEY not set"
        return state

    genai.configure(api_key=api_key)

    # Build prompt
    if state['attempts'] == 0:
        prompt = get_base_prompt(state['pdf_content'], state['csv_content'], state['target_bank'])
    else:
        base_prompt = get_base_prompt(state['pdf_content'], state['csv_content'], state['target_bank'])
        refinement_prompt = get_refinement_prompt(state['generated_code'], state['error'])
        prompt = f"{base_prompt}\n\n{refinement_prompt}"

    try:
        # Use the correct model name and JSON mode
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"},
                        "python_code": {"type": "string"},
                        "reasoning_for_test": {"type": "string"},
                        "test_passed": {"type": "boolean"}
                    },
                    "required": ["plan", "python_code", "reasoning_for_test", "test_passed"]
                }
            )
        )

        # Parse JSON response
        data = json.loads(response.text)

        state['plan'] = data.get('plan', 'No plan provided')
        state['generated_code'] = data.get('python_code', '')
        state['reasoning_for_test'] = data.get('reasoning_for_test', 'No reasoning provided')
        state['test_passed'] = data.get('test_passed', False)

        print(f"Plan: {state['plan']}")
        print(f"Generated {len(state['generated_code'])} characters of code")
        print(f"AI Prediction: {'PASS' if state['test_passed'] else 'FAIL'}")

    except Exception as e:
        print(f"Error generating code: {e}")
        state['error'] = f"Code generation failed: {str(e)}"
        return state

    # Write generated code to file
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    parser_file = parser_dir / f"{state['target_bank']}_parser.py"

    with open(parser_file, "w", encoding="utf-8") as f:
        f.write(state['generated_code'])

    print(f"📝 Code written to {parser_file}")
    return state

def run_tests_step(state: AgentState):
    """Test the generated parser against expected output"""
    print("--- 🧪 RUNNING TESTS ---")
    state['attempts'] += 1

    if not state.get('generated_code'):
        state['error'] = "No code was generated to test"
        return state

    try:
        # Import and run the parser
        import importlib.util
        import importlib

        parser_path = Path("custom_parsers") / f"{state['target_bank']}_parser.py"

        # Clear cached modules
        module_name = f"parser_{state['target_bank']}_{state['attempts']}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = parser_module
        spec.loader.exec_module(parser_module)

        # Check if parse function exists
        if not hasattr(parser_module, 'parse'):
            state['error'] = "Generated parser missing 'parse' function"
            return state

        # Run the parser
        result_df = parser_module.parse(state['pdf_path'])
        expected_df = pd.read_csv(state['csv_path'])

        print(f"📊 Parser extracted {len(result_df)} rows, expected {len(expected_df)} rows")

        # Validation checks
        checks = {
            'columns_match': list(result_df.columns) == list(expected_df.columns),
            'row_count_match': len(result_df) == len(expected_df),
            'content_match': result_df.equals(expected_df)
        }

        if all(checks.values()):
            print("✅ All Tests Passed!")
            state['test_output'] = "Perfect match achieved"
            state['error'] = ""
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            error_msg = f"Failed checks: {', '.join(failed_checks)}"

            if not checks['row_count_match']:
                error_msg += f"\nRow count: expected {len(expected_df)}, got {len(result_df)}"

            if not checks['content_match'] and len(result_df) > 0:
                # Show first mismatch
                for i in range(min(2, len(result_df), len(expected_df))):
                    if i < len(result_df) and i < len(expected_df):
                        expected_row = expected_df.iloc[i]
                        result_row = result_df.iloc[i]

                        if not expected_row.equals(result_row):
                            error_msg += f"\nRow {i+1} mismatch:"
                            error_msg += f"\nExpected: {expected_row.to_dict()}"
                            error_msg += f"\nGot: {result_row.to_dict()}"
                            break

            print("❌ Tests Failed!")
            state['error'] = error_msg

        # Show AI vs Actual result comparison
        ai_predicted_pass = state.get('test_passed', False)
        actual_pass = not state.get('error')
        if ai_predicted_pass == actual_pass:
            print(f"🎯 AI Prediction was CORRECT: {ai_predicted_pass}")
        else:
            print(f"🎲 AI Prediction was WRONG: predicted {ai_predicted_pass}, actual {actual_pass}")

    except Exception as e:
        print("❌ Parser Execution Failed!")
        state['error'] = f"Parser execution error: {str(e)}"

    return state

def should_continue(state: AgentState):
    """Decide whether to retry or end"""
    if not state.get('error'):
        return "end"  # Success

    if state['attempts'] >= state['max_attempts']:
        print(f"--- 🚫 MAX ATTEMPTS ({state['max_attempts']}) REACHED ---")
        return "end"  # Failure after max retries

    return "retry"  # Continue to refine and retry

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="AI Agent for generating bank statement parsers.")
    parser.add_argument("--target", type=str, required=True, help="Target bank name (e.g., icici)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum retry attempts")

    args = parser.parse_args()

    # Setup paths
    base_dir = Path.cwd()
    target = args.target.lower()

    pdf_path = base_dir / "data" / target / f"{target} sample.pdf"
    csv_path = base_dir / "data" / target / "result.csv"

    # Validate inputs
    if not pdf_path.exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"❌ Error: CSV not found at {csv_path}")
        sys.exit(1)

    # Load file content
    try:
        pdf_content = extract_pdf_content(str(pdf_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🚀 AI Agent Starting for {target.upper()} Bank")
    print(f"📄 PDF: {pdf_path}")
    print(f"📊 CSV: {csv_path}")
    print(f"🔄 Max attempts: {args.max_attempts}")
    print(f"{'='*60}")

    # Initialize state
    initial_state = AgentState(
        target_bank=target,
        pdf_path=str(pdf_path),
        csv_path=str(csv_path),
        pdf_content=pdf_content,
        csv_content=csv_content,
        plan="",
        generated_code="",
        reasoning_for_test="",
        test_passed=False,
        test_output="",
        error="",
        attempts=0,
        max_attempts=args.max_attempts
    )

    # Define workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("generate", plan_and_generate_code_step)
    workflow.add_node("test", run_tests_step)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "test")
    workflow.add_conditional_edges(
        "test",
        should_continue,
        {"retry": "generate", "end": END}
    )

    # Run the agent
    app = workflow.compile()
    final_state = app.invoke(initial_state)

    # Final results
    print(f"\n{'='*60}")
    print("🏁 AGENT RUN COMPLETE")
    print(f"{'='*60}")

    if not final_state['error']:
        print(f"✅ SUCCESS! Parser generated and validated!")
        print(f"📁 Location: custom_parsers/{args.target}_parser.py")
        print(f"🔄 Completed in {final_state['attempts']} attempt(s)")

        if final_state.get('reasoning_for_test'):
            print(f"\n🤖 AI Self-Evaluation:")
            print(f"   {final_state['reasoning_for_test']}")

        return 0
    else:
        print(f"❌ FAILED after {final_state['attempts']} attempts")
        print(f"\n📋 Final Error: {final_state['error']}")

        if final_state.get('reasoning_for_test'):
            print(f"\n🤖 AI Self-Evaluation:")
            print(f"   {final_state['reasoning_for_test']}")

        if final_state.get('generated_code'):
            print(f"\n📝 Last Generated Code:")
            print("--- Code Start ---")
            print(final_state['generated_code'][:800] + "..." if len(final_state['generated_code']) > 800 else final_state['generated_code'])
            print("--- Code End ---")

        return 1

if __name__ == "__main__":
    sys.exit(main())
