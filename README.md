# LangChain + Bedrock + HF Embeddings — Student Starter (v2)

## Supported Bedrock models in this starter
- **Meta Llama 3 8B Instruct**: `meta.llama3-8b-instruct-v1:0` (default)
- **Mistral 7B Instruct**: `mistral.mistral-7b-instruct-v0:2`
- **Mixtral 8x7B Instruct**: `mistral.mixtral-8x7b-instruct-v0:1`

> Set `BEDROCK_MODEL_ID` in `.env` to switch.

## Quick Start
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   # .venv\Scripts\activate
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and fill in your AWS + Hugging Face values.
4. Run the app (CLI mode or browser):
   ```bash
   # CLI usage
   python main.py "What is LangChain and why use a vector store?"
   # Web app usage
   python main.py
   # then open http://127.0.0.1:8000/ in your browser
   ```

### Notes
- If Bedrock isn't configured, the script still runs the **retriever + Chroma** pipeline and prints the top chunks.
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2` (fast, local, no API key required).
- Delete `chroma_db` to rebuild the vector store.
- New endpoints available:
  - `/docs` for Swagger UI
  - `/api/reindex` and `/api/clear` for document management
