import argparse
import logging as log
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional, Set

import gradio as gr
import librosa
import numpy as np
import openvino as ov
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedTokenizer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are RohithSense - an intelligent and context-aware AI memory assistant. "
    "Your role is to engage in conversations, track user interactions, and recall past topics to provide better responses. "
    "You focus on understanding long-term conversations and help users learn, research, track progress, or organize information effectively. "
    "You must recall key points from past interactions and suggest follow-ups based on previous discussions. "
    "You should NEVER reset memory during a session but can summarize key insights if asked. "
    "You must keep responses engaging, helpful, and contextually relevant based on past interactions. "
    "Do not ask for or store personal details like age, name, gender, or contact information. "
    "If the user asks for a summary, provide a clear and structured summary of past conversations."
)
GREET_THE_CUSTOMER = "Hi! I'm RohithSense, your AI memory assistant. I remember past interactions and can help track your progress. What can I assist you with today?"
SUMMARIZE_THE_CUSTOMER = (
    "You are now required to summarize the key points from our past conversations. "
    "Summarize in a structured format, focusing on recurring themes or topics discussed. "
    "Provide a quick recap that helps the user recall past discussions easily."
)

# Initialize Model variables
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[PreTrainedTokenizer] = None
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None


def get_available_devices() -> Set[str]:
    """
    List all devices available for inference

    Returns:
        Set of available devices
    """
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_asr_model(model_dir: Path) -> None:
    """
    Load automatic speech recognition model and assign it to a global variable

    Params:
        model_dir: dir with the ASR model
    """
    global asr_model, asr_processor

    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return

    # create a distil-whisper model and its processor
    device = "GPU" if "GPU" in get_available_devices() and ov.__version__ < "2024.3" else "CPU"
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device)
    asr_processor = AutoProcessor.from_pretrained(model_dir)


def load_chat_model(model_dir: Path) -> None:
    """
    Load chat model and assign it to a global variable

    Params:
        model_dir: dir with the chat model
    """
    global chat_model, chat_tokenizer

    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return

    # load llama model and its tokenizer
    device = "GPU" if "GPU" in get_available_devices() else "AUTO"
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device=device, config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)


def respond(prompt: str, streamer: BaseStreamer | None = None) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
        streamer: if not None will use it to stream tokens
    Returns:
        The chat's response
    """
    start_time = time.time()  # Start time
    # tokenize input text
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    # generate response tokens
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, streamer=streamer)
    tokens = outputs[0, input_length:]
    end_time = time.time()  # End time

    # 75 words ~= 100 tokens
    processing_time = end_time - start_time
    log.info(f"Chat model response time: {processing_time:.2f} seconds ({len(tokens) / processing_time:.2f} tokens/s)")

    # decode tokens into text
    return chat_tokenizer.decode(tokens, skip_special_tokens=True)


def get_conversation(history: List[List[str]]) -> str:
    """
    Combines all messages into one string

    Params:
        history: history of the messages (conversation) so far
    Returns:
        All messages combined into one string
    """
    # the conversation must be in that format to use chat template
    conversation = [
        {"role": "system", "content": SYSTEM_CONFIGURATION},
        {"role": "user", "content": GREET_THE_CUSTOMER}
    ]
    # add prompts to the conversation
    for user_prompt, assistant_response in history:
        if user_prompt:
            conversation.append({"role": "user", "content": user_prompt})
        if assistant_response:
            conversation.append({"role": "assistant", "content": assistant_response})

    # use a template specific to the model
    return chat_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    conv = get_conversation([[None, None]])
    return respond(conv)


def chat(history: List[List[str]]) -> List[List[str]]:
    """
    Chat function. It generates response based on a prompt

    Params:
        history: history of the messages (conversation) so far
    Returns:
        History with the latest chat's response (yields partial response)
    """
    # convert list of message to conversation string
    conversation = get_conversation(history)

    # use streamer to show response word by word
    chat_streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)

    # generate response for the conversation in a new thread to deliver response token by token
    thread = Thread(target=respond, args=[conversation, chat_streamer])
    thread.start()

    # get token by token and merge to the final response
    history[-1][1] = ""
    for partial_text in chat_streamer:
        history[-1][1] += partial_text
        # "return" partial response
        yield history

    # wait for the thread
    thread.join()


def transcribe(audio: Tuple[int, np.ndarray], conversation: List[List[str]]) -> List[List[str]]:
    """
    Transcribe audio to text

    Params:
        audio: audio to transcribe text from
        conversation: conversation history with the chatbot
    Returns:
        User prompt as a text
    """
    start_time = time.time()  # Start time for ASR process

    sample_rate, audio = audio
    # the whisper model requires 16000Hz, not 44100Hz
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_AUDIO_SAMPLE_RATE).astype(np.int16)

    # get input features from the audio
    input_features = asr_processor(audio, sampling_rate=TARGET_AUDIO_SAMPLE_RATE, return_tensors="pt").input_features

    # use streamer to show transcription word by word
    text_streamer = TextIteratorStreamer(asr_processor, skip_prompt=True, skip_special_tokens=True)

    # transcribe in the background to deliver response token by token
    thread = Thread(target=asr_model.generate, kwargs={"input_features": input_features, "streamer": text_streamer})
    thread.start()

    conversation.append(["", None])
    # get token by token and merge to the final response
    for partial_text in text_streamer:
        conversation[-1][0] += partial_text
        # "return" partial response
        yield conversation

    end_time = time.time()  # End time for ASR process
    log.info(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

    # wait for the thread
    thread.join()

    return conversation


def summarize(conversation: List) -> str:
    """
    Summarize the patient case

    Params
        conversation: history of the messages so far
    Returns:
        Summary
    """
    conversation.append([SUMMARIZE_THE_CUSTOMER, None])
    for partial_summary in chat(conversation):
        yield partial_summary[-1][1]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface with modern design

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(
        title="RohithSense - Your AI Memory Companion",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
    ) as demo:
        with gr.Row():
            gr.Markdown("""
                # 🧠 RohithSense
                ### Your Intelligent Memory Companion
                
                Engage in natural conversations while I track and remember our interactions.
            """)
            
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot_ui = gr.Chatbot(
                    value=[[None, initial_message]], 
                    label="Conversation History",
                    height=500,
                    bubble_full_width=False,
                    show_label=True,
                    elem_classes="chat-display"
                )
                
            with gr.Column(scale=1):
                # Controls panel
                with gr.Group():
                    gr.Markdown("### Voice Input")
                    input_audio_ui = gr.Audio(
                        sources=["microphone"],
                        label="Record your message",
                        elem_classes="audio-input"
                    )
                    submit_audio_btn = gr.Button(
                        "🎙️ Submit Recording",
                        variant="primary",
                        interactive=False,
                        size="lg"
                    )
                
                with gr.Group():
                    gr.Markdown("### Memory Features")
                    summarize_button = gr.Button(
                        "📝 Generate Conversation Summary",
                        variant="secondary",
                        interactive=False,
                        size="lg"
                    )
                    summary_ui = gr.Textbox(
                        label="Conversation Summary",
                        interactive=False,
                        lines=5,
                        elem_classes="summary-box"
                    )

        # Custom CSS for better styling
        gr.Markdown("""
            <style>
            .chat-display {
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .audio-input {
                border-radius: 8px;
                padding: 8px;
            }
            .summary-box {
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            </style>
        """)

        # Events (keeping the same logic but with updated component references)
        input_audio_ui.change(
            lambda x: gr.Button(interactive=False) if x is None else gr.Button(interactive=True),
            inputs=input_audio_ui,
            outputs=submit_audio_btn
        )

        submit_audio_btn.click(
            lambda: gr.Button(interactive=False),
            outputs=submit_audio_btn
        ).then(
            lambda: gr.Button(interactive=False),
            outputs=summarize_button
        ).then(
            transcribe,
            inputs=[input_audio_ui, chatbot_ui],
            outputs=chatbot_ui
        ).then(
            chat,
            chatbot_ui,
            chatbot_ui
        ).then(
            lambda: None,
            inputs=[],
            outputs=[input_audio_ui]
        ).then(
            lambda: gr.Button(interactive=True),
            outputs=summarize_button
        )

        summarize_button.click(
            lambda: gr.Button(interactive=False),
            outputs=summarize_button
        ).then(
            summarize,
            inputs=chatbot_ui,
            outputs=summary_ui
        ).then(
            lambda: gr.Button(interactive=True),
            outputs=summarize_button
        )

    return demo


def run(asr_model_dir: Path, chat_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the assistant application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        chat_model_dir: dir with the chat model
        public_interface: whether UI should be available publicly
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_dir)
    # load chat model
    load_chat_model(chat_model_dir)

    if chat_model is None or asr_model is None:
        log.error("Required models are not loaded. Exiting...")
        return

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)

    log.info("Demo is ready!")
    # launch demo
    demo.queue().launch(share=public_interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr_model_dir', type=str, default="model/distil-whisper-large-v3-FP16", help="Path to the automatic speech recognition model directory")
    parser.add_argument('--chat_model_dir', type=str, default="model/llama3.2-3B-INT4", help="Path to the chat model directory")
    parser.add_argument('--public', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model_dir), Path(args.chat_model_dir), args.public)
