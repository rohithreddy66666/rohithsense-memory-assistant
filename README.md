# rohithsense-memory-assistant
RohithSense: An intelligent AI memory assistant built with OpenVINO™ that remembers conversation context, tracks user interactions over time, and provides personalized responses based on conversation history. Featuring speech recognition, context-aware dialogue, and conversation summarization capabilities.
# rohithsense-memory-assistant

🧠 **An Intelligent Context-Aware Memory Assistant built with OpenVINO™ Toolkit**

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

RohithSense is a specialized AI memory assistant designed to engage in natural conversations while tracking user interactions and recalling past topics to provide more contextually relevant responses. Built on the foundation of OpenVINO™ and state-of-the-art language models, this application offers a seamless voice-activated interface that developers can easily integrate and deploy.

> **Repository Description**: RohithSense: An intelligent AI memory assistant built with OpenVINO™ that remembers conversation context, tracks user interactions over time, and provides personalized responses based on conversation history. Featuring speech recognition, context-aware dialogue, and conversation summarization capabilities.

## Key Features

- **Conversation Memory**: Tracks and recalls past interactions to provide contextually relevant responses
- **Voice-Activated Interface**: Natural speech recognition for hands-free interaction
- **Conversation Summaries**: Generate structured summaries of past discussions
- **Progress Tracking**: Helps users track learning progress over time
- **Context Awareness**: Understands long-term conversations and suggests relevant follow-ups

## Technology Stack

- **OpenVINO™ Toolkit**: For optimized neural network inference
- **LLaMA Models**: State-of-the-art language models for natural conversation
- **Distil-Whisper**: For efficient speech recognition
- **Gradio**: For the interactive web interface

## Getting Started

Follow these steps to set up and run RohithSense on your machine. We recommend using Ubuntu for the best experience.

### Installing Prerequisites

This project requires Python 3.10 or higher and several libraries:

```bash
sudo apt install git gcc python3-venv python3-dev
```

> **NOTE**: Windows users may also need to install Microsoft Visual C++ Redistributable.

### Setting Up Your Environment

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_REPOSITORY/rohithsense-memory-assistant.git
cd rohithsense-memory-assistant
```

#### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # For Unix-based systems (Linux/macOS)
# For Windows: venv\Scripts\activate
```

#### 3. Install Required Packages

```bash
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

### Accessing Required Models

#### LLaMA Models Authentication

1. **Meta AI Access**: Visit [Meta AI's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and complete the registration form.
2. **Hugging Face Setup**:
   - Create a [Hugging Face account](https://huggingface.co/join)
   - Authenticate with the same email used on Meta AI's website
   - Login using the Hugging Face CLI:
     ```bash
     huggingface-cli login
     ```
     (When prompted to add the token as a git credential, respond with 'n')

### Model Conversion and Optimization

> **NOTE**: This process requires significant bandwidth, disk space (>8GB), and memory (>32GB) for the initial conversion. Subsequent runs will be faster.

#### 1. Convert Automatic Speech Recognition (ASR) Model

For CPU (INT8 precision):
```bash
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3 --precision int8
```

For GPU (FP16 precision):
```bash
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3
```

> **⚠️ Warning**: Windows users may see a "Permission Error" message due to an export function bug. The model will export successfully, but you may need to clear the temp directory manually.

For Chinese language support:
```bash
python convert_and_optimize_asr.py --asr_model_type belle-distilwhisper-large-v2-zh --precision int8
```

#### 2. Convert Chat Model

For desktop/server processors:
```bash
python convert_and_optimize_chat.py --chat_model_type llama3.1-8B --precision int4
```

For AI PC or edge devices:
```bash
python convert_and_optimize_chat.py --chat_model_type llama3.2-3B --precision int4
```

For Chinese language support:
```bash
python convert_and_optimize_chat.py --chat_model_type qwen2-7B --precision int4
```

### Running RohithSense

> **NOTE**: This application requires substantial memory (>16GB) due to the size of the models, especially the chatbot.

```bash
python app.py --asr_model_dir path/to/asr_model --chat_model_dir path/to/chat_model
```

Add `--public` flag to make it publicly accessible.

### Accessing the Web Interface

After running the script, Gradio will provide a local URL (typically http://127.0.0.1:XXXX) which you can open in your web browser to start interacting with RohithSense.

## Using RohithSense

1. Navigate to the provided Gradio URL in your web browser
2. Use the microphone to record your voice queries
3. Click "Submit Recording" to process your query
4. RohithSense will respond based on your query and the conversation context
5. Use the "Generate Conversation Summary" button to create a summary of your interaction

RohithSense remembers previous interactions and uses this context to provide more relevant responses over time.

## Key Capabilities

- **Memory Tracking**: Recalls key points from past conversations
- **Follow-up Suggestions**: Offers relevant follow-ups based on conversation history
- **Progress Monitoring**: Helps track learning or research progress
- **Information Organization**: Assists in organizing information effectively
- **Context Preservation**: Maintains conversation context throughout sessions

## Additional Resources

- [Learn more about OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- [Explore OpenVINO's documentation](https://docs.openvino.ai/latest/index.html)
- [LLaMA model information](https://ai.meta.com/llama/)

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE file for details.
