# DeepSearch: An AI-Powered Research Assistant

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

DeepSearch is an intelligent research assistant designed to understand complex queries, formulate comprehensive research plans, execute web searches, synthesize information from multiple sources, and deliver well-structured, insightful answers. This project leverages **PlanAI** for graph-based processing, **Flask** for backend communication, and **Svelte** for a modern, reactive frontend.

## Features

- **Intelligent Research Planning:** Automatically generates multi-phase research plans based on user queries.
- **Web Search Integration:** Performs targeted web searches for each phase of the research plan.
- **Information Synthesis:** Summarizes and synthesizes information from multiple sources into a coherent narrative.
- **Real-time Feedback:** Provides users with thinking/progress updates during processing.
- **Markdown Support:** Delivers final responses in well-formatted Markdown.
- **Session Management:** Supports restoring previous sessions and automatic cleanup of stale sessions.
- **Debugging and Replay:** Includes tools for capturing and replaying sessions for development and testing.
- **Multiple LLM Support:** Supports OpenAI, Anthropic, and Ollama (local models) with dynamic validation and model selection.
- **Audio Player:** Includes a built-in audio player (currently playing classical music - a placeholder for future functionality).

## Project Structure

The project consists of the following main components:

- **`deepsearch/`**: Backend logic (Python, Flask, PlanAI).
  - **`debug.py`**: Debugging utilities (capture/replay sessions).
  - **`deepsearch.py`**: Main backend application, Flask setup, WebSocket handling, task queue management.
  - **`graph.py`**: Definition of the research task graph and workers (plan creation, search, summarization, response generation).
  - **`session.py`**: Session management (creation, retrieval, update, deletion, SID mapping, metadata handling).
- **`frontend/`**: Frontend application (Svelte, JavaScript).
  - **`lib/components/`**:
    - **`AudioPlayer.svelte`**:  Component for playing and controlling audio tracks.
    - **`ChatInterface.svelte`**: Chat interface, message display, loading indicators, markdown rendering.
    - **`SessionManager.svelte`**: WebSocket connection management, session restoration, backend communication.
    - **`SettingsSidebar.svelte`**: Sidebar component to configure API keys, select LLM providers/models, and persist settings.
  - **`lib/stores/`**:
    - **`messageBus.svelte.js`**: Centralized event bus for frontend communication.
    - **`sessionStore.svelte.js`**: Stores session ID and connection status.
    - **`trackStore.js`**: Manages the playlist and current track for the audio player.
  - **`routes/`**:
    - **`+layout.svelte`**: Main application layout.
    - **`+page.svelte`**: Main page, integrates `SessionManager`, `ChatInterface`, and `AudioPlayer`.
- **`tests/`**: Unit tests for session management (`test_session.py`).

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn
- [Playwright](https://playwright.dev/docs/intro) (for web scraping):

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/provos/planai.git](https://github.com/provos/planai.git)
   cd planai/examples/deepsearch
   ```

2.  **Install backend dependencies:**

    ```bash
    poetry install
    poetry run playwright install
    ```

3.  **Install frontend dependencies:**

    ```bash
    cd frontend
    npm install
    ```

4.  **Configure API Keys (Optional):**

      - Create a `.env.local` file in the root of the `deepsearch` directory.
      - Add your API keys for Serper, OpenAI, and/or Anthropic (see `.env.local` example).
      - **Note:** Serper API key is required for web search functionality unless you already have a key stored in your settings (persisted across sessions). OpenAI and Anthropic keys are optional, required only if you want to use their respective LLMs.

    <!-- end list -->

    ```
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    SERPER_API_KEY=your_serper_api_key
    ```

### Running the Application

1.  **Start the backend server:**

    ```bash
    cd .. # (If you're in the `frontend` directory)
    poetry run python deepsearch/deepsearch.py
    ```

2.  **Start the frontend development server:**

    ```bash
    cd frontend
    npm run dev
    ```

3.  **Access the application:**

    Open your web browser and navigate to `http://localhost:5173`.

## Usage

1.  **Enter a research query** in the chat input field and press Enter or click the Send button.
2.  **Observe the research process:** DeepSearch will generate a research plan, perform web searches, and synthesize the results. You'll see real-time updates in the chat interface.
3.  **Receive the final response:** Once the research is complete, you'll receive a well-structured answer in Markdown format.
4.  **Configure Settings (Optional):** Click the gear icon to open the settings sidebar. Here you can:
      - Enter API keys for Serper, OpenAI, and Anthropic.
      - Specify the Ollama host address if you are running a local model.
      - Select your preferred LLM provider and model from the available options.
      - Save your settings, which will be persisted across sessions.

## Debugging and Replay

DeepSearch includes tools for capturing and replaying sessions, useful for development and testing.

  - **Capture mode:** Records interactions and function calls for a session.
  - **Replay mode:** Simulates a previously captured session.

**Enable debugging:**

```bash
poetry run python deepsearch/deepsearch.py --debug
```

**Enable replay mode (with debugging):**

```bash
poetry run python deepsearch/deepsearch.py --debug --replay
```

**Adjust replay delay:**

```bash
poetry run python deepsearch/deepsearch.py --debug --replay --replay-delay 0.5
```

(This sets a 0.5-second delay between replayed events).

**Debug Output:** Captured sessions are stored in the `debug_output/` directory.

## Testing

To run the at the moment limited unit tests:

```bash
poetry run pytest
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

  - **PlanAI:** [https://github.com/provos/planai](https://github.com/provos/planai)
  - **Flask:** [https://flask.palletsprojects.com/](https://www.google.com/url?sa=E&source=gmail&q=https://flask.palletsprojects.com/)
  - **Svelte:** [https://svelte.dev/](https://svelte.dev/)
  - **Socket.IO:** [https://socket.io/](https://socket.io/)
  - **Marked:** [https://marked.js.org/](https://www.google.com/url?sa=E&source=gmail&q=https://marked.js.org/)