<div align="center">
  <img src="aegis/static/img/aegis-logo.svg" alt="aegis logo" width="200"/>
  <h1>aegis - AI Powered Security Analysis Tool</h1>
</div>

aegis is an AI-powered Static Application Security Testing (SAST) tool that uses Large Language Models (LLMs) via Ollama to analyze source code for security vulnerabilities. It provides comprehensive vulnerability detection with detailed reports.

## Overview

aegis leverages multiple AI models working in parallel to identify security vulnerabilities in source code. By utilizing consensus algorithms, it combines findings from multiple models to reduce false positives and improve detection accuracy. The tool supports various LLM providers including Ollama (local), OpenAI, Anthropic, Azure, and HuggingFace.

## Features

- **Multi-Model Analysis**: Execute scans using multiple AI models simultaneously for comprehensive coverage
- **Consensus Engine**: Combines findings from multiple models using various consensus strategies (union, majority vote, weighted vote, judge model)
- **CWE-Aware Detection**: Automatically focuses on relevant Common Weakness Enumeration (CWE) vulnerability types based on the programming language
- **Web Interface**: User-friendly interface with dark mode support and integrated code snippet visualization for vulnerability context
- **Code Context Display**: Shows vulnerable code snippets with context lines and severity-based highlighting
- **Industry-Standard Exports**: Export results in SARIF format for integration with CI/CD pipelines
- **Flexible Model Support**: Compatible with local (Ollama) and cloud-based LLM providers

## Installation

### Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`
- Ollama (optional, for local model execution)

### Setup

####  Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/canoztas/aegis
   cd aegis
   ```

2. Install dependencies:
   
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Or using `pip`:
   ```bash
   pip install -e .
   ```

3. Configure models in `config/models.yaml` (see Configuration section)

4. Start the application:
   ```bash
   uv run python app.py
   # or
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`

### Docker Commands

- **Build the image:**
  ```bash
  docker build -t aegis .
  ```

- **Run the container:**
  ```bash
  docker run -d -p 5000:5000 \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/data:/app/data \
    aegis
  ```

- **View logs:**
  ```bash
  docker-compose logs -f aegis
  ```

- **Stop the services:**
  ```bash
  docker-compose down
  ```

## Usage

### Web Interface

1. Navigate to the main page and upload a ZIP file containing your source code
2. Select one or more AI models from the available options
3. Choose a consensus strategy (optional)
4. Initiate the scan and wait for results
5. Review findings with code snippets and detailed vulnerability information
6. Export results in SARIF or CSV format if needed

### API

The application provides RESTful API endpoints for programmatic access:

- `POST /api/scan` - Create a new security scan
- `GET /api/scan/<scan_id>` - Retrieve scan results
- `GET /api/scan/<scan_id>/export/sarif` - Export results as SARIF
- `GET /api/scan/<scan_id>/export/csv` - Export results as CSV
- `GET /api/models` - List available models
- `GET /api/models/<model_id>/health` - Check model health status

## Configuration

Model configuration is managed through `config/models.yaml`. Each model entry specifies:

- Model identifier
- Adapter type (ollama, openai, anthropic, azure, huggingface, classic)
- Provider-specific settings (API keys, endpoints, model names)
- Enabled status

Example configuration:

```yaml
models:
  - id: llama3.1
    type: ollama
    enabled: true
    config:
      model_name: llama3.1:8b
      base_url: http://localhost:11434
```

## Consensus Strategies

aegis supports multiple consensus strategies to combine findings from different models:

- **Union**: Returns all unique findings from all models
- **Majority Vote**: Includes findings detected by more than half of the models
- **Weighted Vote**: Uses model-specific confidence weights
- **Judge Model**: Employs a separate LLM to evaluate and merge findings

## Supported File Types

The tool analyzes common source code file types including:
- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- Go (.go)
- Ruby (.rb)
- PHP (.php)
- And other text-based source files

## Architecture

aegis is built with a modular architecture:

- **Adapters**: Standardized interfaces for different LLM providers
- **Model Registry**: Centralized model configuration and management
- **Prompt Builder**: Constructs prompts with CWE context and structured output schemas
- **Consensus Engine**: Implements various strategies for combining model findings
- **Multi-Model Runner**: Orchestrates parallel execution of multiple models
- **Export Module**: Generates SARIF and CSV reports

## Development

### Project Structure

```
aegis/
├── adapters/          # LLM adapter implementations
├── consensus/         # Consensus engine strategies
├── static/           # Frontend assets (CSS, JS, images)
├── templates/        # HTML templates
├── config/           # Configuration files
├── data/             # CWE data and templates
└── routes.py         # Flask application routes
```

### Running Tests

```bash
uv run pytest
# or
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please ensure that your code follows the project's style guidelines and includes appropriate tests.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
