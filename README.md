# aegis - AI Powered Security Analysis Tool

aegis is a AI powered Static Application Security Testing (SAST) tool that uses Large Language Models (LLMs) via Ollama to analyze source code for security vulnerabilities. It provides comprehensive vulnerability detection with detailed reports.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) running locally

### Installation

1. Clone the repository:

```bash
git clone https://github.com/canoztas/aegis
cd aegis
```

2. Install dependencies using uv:

```bash
uv sync
```

3. Make sure you have logged in to Ollama:

```bash
ollama signin
```

Follow the prompts to complete the login process.

4. Run the application:

```bash
uv run python app.py
```

5. Open your browser and go to `http://localhost:7766`

## Usage

1. **Upload Source Code**: Create a ZIP file containing your source code
2. **Drag & Drop**: Use the web interface to upload your ZIP file
3. **Analysis**: aegis will automatically extract and analyze your code
4. **Review Results**: View detailed vulnerability reports with:
   - File paths and line numbers
   - Severity levels (Low, Medium, High, Critical)
   - CVSS-like scores (0.0 to 10.0)
   - Detailed descriptions
   - Remediation recommendations
   - Code snippets showing vulnerable code

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Model to use for analysis (default: `gpt-oss:120b-cloud`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `7766`)
- `FLASK_ENV`: Set to `development` for debug mode

### Supported File Types

aegis analyzes the following file types:

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)
- C# (`.cs`)
- PHP (`.php`)
- Ruby (`.rb`)
- Go (`.go`)
- Rust (`.rs`)
- SQL (`.sql`)
- Shell scripts (`.sh`, `.bash`)
- HTML (`.html`)
- CSS (`.css`)

## API Usage

aegis also provides a REST API for programmatic access:

```bash
curl -X POST -F "file=@your-code.zip" http://localhost:7766/api/analyze
```

Response format:

```json
{
  "results": [
    {
      "file_path": "src/app.py",
      "line_number": 42,
      "severity": "high",
      "score": 8.5,
      "category": "SQL Injection",
      "description": "Potential SQL injection vulnerability...",
      "recommendation": "Use parameterized queries...",
      "code_snippet": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
    }
  ]
}
```

## Acknowledgments

- Built with Flask and modern web technologies
- Powered by Ollama and Large Language Models
- Inspired by industry-standard SAST tools
