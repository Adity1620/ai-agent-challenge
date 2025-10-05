# AI Agent for Automated Bank Statement Parser Generation 🤖📄

This project features a sophisticated AI agent designed to automate the creation of Python parsers for bank statements. Leveraging the power of Google's Gemini API and the LangGraph framework, the agent analyzes a sample PDF and a target CSV format to autonomously write, test, and refine a custom parser for any given bank.

## ✨ Features

- **Autonomous Code Generation**: Automatically writes Python code using pdfplumber and pandas.
- **LLM-Powered**: Integrates with Google's Gemini API for intelligent code planning and generation.
- **Self-Correcting Loop**: Implements a "plan -> generate -> test" cycle, allowing the agent to retry and fix its own code upon failure.
- **Extensible**: Easily add support for new banks by providing a sample PDF and a target CSV.
- **Built-in Testing**: Includes a separate script to validate the accuracy of any generated parser.

## ⚙️ How It Works

The agent follows a stateful, cyclical workflow managed by LangGraph:

1. **Analyze**: It reads the text content from a sample PDF and the structure of a target CSV file.
2. **Plan & Generate**: It sends this context to the Gemini model, instructing it to create a plan and generate a Python function to parse the PDF into the desired CSV format.
3. **Test**: The agent executes the generated code, running it against the sample PDF and comparing the output DataFrame with the target CSV.
4. **Refine (or Finish)**: If the test fails, the agent feeds the error and the failed code back into the model for another attempt. If the test succeeds, the process completes, and the final parser is saved.

![AI Agent Workflow Diagram](Karbon_ai_challenge(agent workflow).png)

_Diagram: AI Agent Workflow Phases - Generate, Test, and Decision Loop_

## 🚀 Getting Started

Follow these steps to set up and run the AI agent on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Adity1620/ai-agent-challenge.git
cd ai-agent-challenge
```

### 2. Set Up Environment Variables

You'll need a Google Gemini API key.

- Rename the `.env.example` file to `.env`.
- Open the `.env` file and add your Google Gemini API key:

```bash
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## USAGE

### Generate a New Parser

Run the agent by specifying the target bank. The bank name must correspond to a folder inside the `data/` directory.

```bash
python agent.py --target icici
```

- The agent will start the generation and testing process.
- On success, the final parser will be saved in the `custom_parsers/` directory (e.g., `custom_parsers/icici_parser.py`).

### Test a Generated Parser

You can manually test any generated parser using the `test_parser.py` script. This is useful for verifying the parser's output against the expected result.

```bash
python test_parser.py --parser-module custom_parsers.icici_parser --pdf-path data/icici/"icici sample.pdf" --csv-path data/icici/result.csv
```

**Arguments:**

- `--parser-module`: The Python module path to the generated parser.
- `--pdf-path`: The path to the sample PDF statement.
- `--csv-path`: The path to the ground-truth CSV file for comparison.

## 📂 Project Structure

```
.
├── custom_parsers/      # Stores the successfully generated parsers
├── data/                # Contains sample data for each bank
│   └── icici/
│       ├── icici sample.pdf
│       └── result.csv
├── .env                 # For your API keys
├── agent.py             # The main AI agent script
├── test_parser.py       # Script to manually test a parser
└── requirements.txt     # Project dependencies
```

## 🏦 How to Add a New Bank

It's simple to extend the agent to support a new bank:

1. **Create a Directory**: Create a new folder inside the `data/` directory. For example, for "HDFC Bank", create `data/hdfc`.
2. **Add Sample PDF**: Place a sample bank statement PDF in the new folder (e.g., `data/hdfc/hdfc_sample.pdf`).
3. **Add Target CSV**: Create a `result.csv` file in the same folder. This file is crucial as it defines the exact headers and data format you expect the parser to extract from the PDF.
4. **Run the Agent**: Execute the agent with the new target name.

```bash
python agent.py --target hdfc
```
