# DeepSearch: An AI-Powered Research Assistant

This is an example project for PlanAI that aims to create a similar experience as demonstrated by Google's Gemini DeepResearch experiment. DeepSearch is designed to be an intelligent research assistant capable of understanding complex queries, formulating comprehensive research plans, executing web searches, synthesizing information from multiple sources, and delivering well-structured, insightful answers to users.

## Project Structure

The project is organized into several key components:

-   **`deepsearch/`**: Contains the core backend logic, including the graph-based processing, session management, and debugging tools.
-   **`frontend/`**: Houses the Svelte-based frontend application, providing a user-friendly chat interface for interaction.
-   **`tests/`**: Includes unit tests to ensure the reliability and correctness of the session management functionality.

### Backend (`deepsearch/`)

The backend is built using Python and leverages the Flask framework for communication with the frontend. It utilizes **PlanAI** as a graph-based processing system to manage the complex workflow of research tasks.

#### Key Files:

-   **`debug.py`**: Implements debugging functionalities, allowing for capturing and replaying sessions for development and testing.
-   **`deepsearch.py`**: The main backend file. It sets up the Flask application, handles WebSocket connections, manages the task queue, and orchestrates the interaction between the frontend and the research graph.
-   **`graph.py`**: Defines the task graph and workers responsible for plan creation, search query generation, information retrieval, and response synthesis.

### Frontend (`frontend/`)

The frontend is a Svelte application that provides a clean and intuitive chat interface. It communicates with the backend via WebSockets to send user queries and receive real-time updates and responses.

#### Key Components:

-   **`lib/components/`**:
    -   **`ChatInterface.svelte`**: Implements the chat interface, message display, loading indicators, and markdown rendering.
    -   **`SessionManager.svelte`**: Manages the WebSocket connection, session restoration, and communication with the backend.
-   **`lib/stores/`**:
    -   **`messageBus.svelte.js`**: A central message bus for handling events and updates within the frontend.
    -   **`sessionStore.svelte.js`**: Stores session-related state, such as the current session ID and connection status.
-   **`routes/`**:
    -   **`+layout.svelte`**: Defines the main layout of the application.
    -   **`+page.svelte`**: The main page component, integrating the `SessionManager` and `ChatInterface`.

## Getting Started

### Prerequisites

-   Python 3.10+
-   Node.js 18+
-   npm or yarn

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/provos/planai.git
    cd planai/examples/deepsearch
    ```

2.  **Install backend dependencies:**

    ```bash
    pip install -e .
    ```

3.  **Install frontend dependencies:**

    ```bash
    cd frontend
    npm install
    ```

### Running the Application

1.  **Start the backend server:**

    ```bash
    cd ..
    python deepsearch/deepsearch.py
    ```

2.  **Start the frontend development server:**

    ```bash
    cd frontend
    npm run dev
    ```

3.  **Access the application:**

    Open your web browser and navigate to `http://localhost:5173`.

## Debugging and Replay

The `debug.py` module provides tools for capturing and replaying sessions.

-   **Capture mode:** Records all interactions and function calls for a session, allowing for detailed analysis and debugging.
-   **Replay mode:** Simulates a previously captured session, replaying the events with configurable delays. This is useful for testing and development.

To enable debugging, run the backend with the `--debug` flag. To enable replay mode, also include the `--replay` flag.

```bash
python deepsearch/deepsearch.py --debug --replay
```