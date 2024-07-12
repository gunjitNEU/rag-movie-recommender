# Chatbot-LLM-Interaction

## Overview
The primary objective of this project is to develop a chatbot that leverages Large Language Models (LLM) to assist users in planning their travels. The chatbot aims to provide personalized recommendations, detailed itineraries, and useful travel tips, making the travel planning process easier and more efficient.

## Technology Stack:
- **Frontend**: Streamlit for building the user interface.
- **Backend**: Ollama LLM model running on a local machine for natural language processing and query handling.

## Project Structure
-  Contains Python Streamlit-based chatbot script, data generator python notebook and base data used.
- **video/**: Contains [video](video) presentation. You can also watch the video on [YouTube](https://www.youtube.com/watch?v=W0BGfIb-dZs).

## Dependencies
- Python 3.7+
- streamlit
- request
- ollama

## Usage
1. Clone the repository: `git clone https://github.com/gunjitNEU/rag-movie-recommender.git`
2. Navigate to the project directory: `cd rag-movie-recommender`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Streamlit application: `streamlit run finance_chatbot.py`
5. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).
6. Interact with the chatbot by typing messages and receiving responses from the local LLM service.

## Setting up LLMs
Ensure your local language model service is running:

### Installation Instructions:
### macOS
1.	Download the [Ollama installer for macOS](https://ollama.com/download/Ollama-darwin.zip).
2.	Extract the downloaded ZIP file.
3.	Open Terminal and navigate to the extracted folder.
4.	Run the following command to install Ollama: `./install.sh`

### Windows Preview
1.	Download the [Ollama for Windows](https://ollama.com/download/OllamaSetup.exe) setup executable.
2.	Run the downloaded executable file and follow the on-screen instructions to complete the installation.

### Linux
1.	Open a terminal window.
2.	Run the following command to install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`

For further instructions, visit the [Ollama GitHub page](https://github.com/ollama/ollama).

### After installing Ollama, start the service

Start your local instance of the LLM service (e.g., llama). Example command to start the service: `ollama serve --model llama3`

## Acknowledgment
This LLM model was originally published by [Ollama](https://github.com/ollama/ollama).