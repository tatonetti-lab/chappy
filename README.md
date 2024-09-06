# Chappy: A Conversational Assistant

Chappy is a versatile conversational assistant designed to facilitate various tasks including chat interactions, quick analysis script writing, graphic generation, and monitoring model statuses. Built with privacy and security in mind, Chappy is safe for handling PHI (Protected Health Information), making it suitable for use in sensitive environments.

## Features

### Chat
- Engage in conversations with Chappy using predefined or custom prompts.
- Edit and save conversation histories.
- Download conversation histories as text files.
- Upload and process past conversation files.

### Quick Analysis Script Writer
- Upload CSV files and generate Python scripts based on user queries.
- Auto-fill descriptions and suggested analyses based on the dataset.
- Customize the analysis by entering specific requests.
- Download the generated Python script.

### Graphic Generation
- Upload CSV files and request graphical data representations.
- Auto-fill descriptions and suggest plots based on the dataset.
- Generate and display plots using Plotly based on user or suggested queries.
- Download generated plots as PDF files.

### Model Status
- Monitor the status of various models including their availability and time since last use.
- Refresh model status to get real-time updates.

## User Interface
- **Sidebar**: Contains tools and settings like model selection, prompt selection, and user authentication status.
- **Main Area**: Depending on the selected functionality, it displays chat interface, script generation tools, or model status.
- **Feedback**: Users can leave feedback directly through a GitHub link provided in the sidebar.

## Authentication
- Supports token-based authentication for secure access.
- In development mode, authentication can be bypassed for testing purposes.

## Configuration
- Users can select different models for tasks, adjust parameters like token count and temperature for model responses.
- Configuration settings are read from a `.gitignore/config.ini` file ensuring sensitive information is not exposed.

## Safety and Compliance
- Ensures compliance with PHI handling protocols.
- Uses encryption for sensitive data like usernames.

## Technology Stack
- Built with Streamlit for the frontend.
- Utilizes Plotly for generating interactive graphics.
- Employs OpenAI's Azure API for natural language processing tasks.


## Contribution and Support
- Users can contribute by submitting pull requests or issues on GitHub.
- For help, users can access a dedicated help section linked in the application.
