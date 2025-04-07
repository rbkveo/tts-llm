import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional, Dict

import streamlit as st
from pydub import AudioSegment
from transformers import pipeline
import ollama
import google.generativeai as genai

# Optional: Suppress some warnings for cleaner console
warnings.filterwarnings("ignore", category=FutureWarning)

# Increase the Hugging Face download timeout if needed
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# Whisper model configurations
WHISPER_MODELS: Dict[str, Dict[str, str]] = {
    "tiny": {
        "name": "openai/whisper-tiny",
        "description": "Fastest, lowest accuracy. Good for testing. ~1GB VRAM required.",
        "params": "39M parameters"
    },
    "base": {
        "name": "openai/whisper-base",
        "description": "Fast, decent accuracy. Good for short clips. ~1GB VRAM required.",
        "params": "74M parameters"
    },
    "small": {
        "name": "openai/whisper-small",
        "description": "Balanced speed/accuracy. Good for general use. ~2GB VRAM required.",
        "params": "244M parameters"
    },
    "medium": {
        "name": "openai/whisper-medium",
        "description": "Better accuracy, slower. Good for multiple languages. ~5GB VRAM required.",
        "params": "769M parameters"
    },
    "large": {
        "name": "openai/whisper-large",
        "description": "Best accuracy, slowest. Recommended for production. ~10GB VRAM required.",
        "params": "1.5B parameters"
    }
}

# Initialize Gemini (will be configured with API key later)
def init_gemini(api_key: str):
    """Initialize Gemini with the provided API key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def analyze_with_gemini(text: str, model, system_prompt: str) -> str:
    """Use Gemini API to analyze text"""
    try:
        response = model.generate_content([system_prompt, text])
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return ""

def perform_multi_step_analysis(
    text: str,
    analysis_type: Literal["none", "ollama", "gemini"],
    model_name: Optional[str] = None,
    gemini_model = None,
    system_prompts: dict = None
) -> dict:
    """
    Perform multi-step analysis of the text using the specified model.
    Returns a dictionary with results from each step.
    """
    results = {
        "raw_text": text,
        "summary": None,
        "tasks": None,
        "key_points": None
    }
    
    if analysis_type == "none":
        return results
        
    # Define default prompts if not provided
    if system_prompts is None:
        system_prompts = {
            "summary": (
                "You are a helpful assistant who summarizes text. "
                "Please produce a concise, structured summary in clear paragraphs."
            ),
            "tasks": (
                "Based on the summary provided, create a list of actionable tasks "
                "for the team. Format each task with priority (High/Medium/Low), "
                "estimated effort (in hours), and clear description."
            ),
            "key_points": (
                "Extract the key discussion points and decisions from the text. "
                "Format them as bullet points with categories."
            )
        }

    try:
        # Step 1: Generate Summary
        if analysis_type == "ollama":
            results["summary"] = summarize_text_with_ollama(
                text=text,
                model_name=model_name,
                system_prompt=system_prompts["summary"]
            )
        else:  # gemini
            results["summary"] = analyze_with_gemini(
                text=text,
                model=gemini_model,
                system_prompt=system_prompts["summary"]
            )

        # Step 2: Generate Tasks (based on summary)
        if results["summary"]:
            if analysis_type == "ollama":
                results["tasks"] = summarize_text_with_ollama(
                    text=results["summary"],
                    model_name=model_name,
                    system_prompt=system_prompts["tasks"]
                )
            else:  # gemini
                results["tasks"] = analyze_with_gemini(
                    text=results["summary"],
                    model=gemini_model,
                    system_prompt=system_prompts["tasks"]
                )

        # Step 3: Extract Key Points
        if results["summary"]:
            if analysis_type == "ollama":
                results["key_points"] = summarize_text_with_ollama(
                    text=results["summary"],
                    model_name=model_name,
                    system_prompt=system_prompts["key_points"]
                )
            else:  # gemini
                results["key_points"] = analyze_with_gemini(
                    text=results["summary"],
                    model=gemini_model,
                    system_prompt=system_prompts["key_points"]
                )

    except Exception as e:
        st.error(f"Error in multi-step analysis: {e}")
        print(f"[ANALYSIS] Error: {e}")

    return results

# --- ASR Pipeline (Whisper) ---
@st.cache_resource
def load_asr_pipeline(model_size: str = "large"):
    """
    Load the Whisper model for automatic-speech-recognition (ASR).
    Args:
        model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
    """
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Invalid model size. Choose from: {', '.join(WHISPER_MODELS.keys())}")
    
    model_name = WHISPER_MODELS[model_size]["name"]
    print(f"[INIT] Loading Whisper {model_size} ASR pipeline...")
    
    asr = pipeline("automatic-speech-recognition", model=model_name)
    
    # Disable timestamp generation
    asr.model.generation_config.return_timestamps = False
    if not hasattr(asr.model.generation_config, "no_timestamps_token_id"):
        try:
            no_ts_id = asr.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            asr.model.generation_config.no_timestamps_token_id = no_ts_id
        except Exception as e:
            print("[WARN] Failed to set no_timestamps_token_id:", e)
            asr.model.generation_config.no_timestamps_token_id = None
    
    print(f"[INIT] Whisper {model_size} pipeline loaded successfully.")
    return asr

asr_pipeline = load_asr_pipeline()


# --- Ollama Helpers ---
@st.cache_data
def list_ollama_models():
    """
    Return a list of model references from Ollama (using the ListResponse object).
    Each model has a `model` field, e.g. "evilfreelancer/o1_gigachat:latest".
    """
    try:
        print("[OLLAMA] Listing local models via ollama.list() ...")
        list_response = ollama.list()  # Returns a ListResponse
        # Each element in list_response.models has `model`, e.g. "evilfreelancer/o1_gigachat:latest"
        model_refs = [m.model for m in list_response.models]
        print("[OLLAMA] Models found:", model_refs)
        return model_refs
    except Exception as e:
        st.error(f"Failed to list Ollama models: {e}")
        print("[OLLAMA] ERROR listing models:", e)
        return []


def summarize_text_with_ollama(text, model_name, system_prompt):
    """
    Use the chosen Ollama model to produce a summary (structured, shortened text).
    The system_prompt is passed from the user input.
    """
    if not text.strip():
        print("[SUMMARIZE] No text to summarize, returning empty.")
        return ""

    user_prompt = (
        "Please summarize the following text:\n\n"
        f"{text}\n\n"
        "### End of text ###\n"
    )

    print(f"[SUMMARIZE] Summarizing text with model='{model_name}'...")
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        total_time = time.time() - start_time
        print(f"[SUMMARIZE] Summarization finished in {total_time:.2f}s.")
        return response["message"]["content"].strip()
    except ollama.ResponseError as e:
        st.error(f"Ollama returned an error: {e.error}")
        print("[SUMMARIZE] Ollama ResponseError:", e.error)
        return ""
    except Exception as e:
        st.error(f"Failed to summarize with Ollama: {e}")
        print("[SUMMARIZE] Exception:", e)
        return ""


# --- Chunked Transcription Logic ---
def process_chunk(segment, idx, total, asr_pipeline, language=None):
    """
    Process a single audio segment (chunk) and return transcription along with the chunk index.
    """
    chunk_start_t = time.time()
    print(f"[TRANSCRIBE] Processing chunk {idx+1}/{total} ...")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        segment.export(tmp.name, format="mp3")
        if language:
            result = asr_pipeline(tmp.name, language=language)
        else:
            result = asr_pipeline(tmp.name)
        text = result.get("text", "")
    elapsed_chunk = time.time() - chunk_start_t
    print(f"[TRANSCRIBE] Chunk {idx+1}/{total} done in {elapsed_chunk:.2f}s.")
    return idx, text


def transcribe_long_audio_parallel(file_path, chunk_length_ms=29000, language=None, max_workers=4, asr_pipeline=None):
    """
    Splits an audio file into chunks and processes each chunk in parallel (ThreadPoolExecutor).
    Args:
        file_path: Path to the audio file
        chunk_length_ms: Length of each chunk in milliseconds
        language: Optional language code to force specific language
        max_workers: Number of parallel workers
        asr_pipeline: The ASR pipeline to use for transcription
    """
    if asr_pipeline is None:
        raise ValueError("ASR pipeline must be provided")

    print(f"[TRANSCRIBE] Loading audio from {file_path} ...")
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    print(f"[TRANSCRIBE] Audio duration: {duration_ms/1000:.2f} seconds.")

    # Create chunks
    segments = []
    if duration_ms <= chunk_length_ms:
        segments.append(audio)
    else:
        for i in range(0, duration_ms, chunk_length_ms):
            segments.append(audio[i : i + chunk_length_ms])

    total_chunks = len(segments)
    print(f"[TRANSCRIBE] Total chunks to process: {total_chunks}")

    start_time = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, segment in enumerate(segments):
            futures.append(
                executor.submit(
                    process_chunk, 
                    segment, 
                    idx, 
                    total_chunks, 
                    asr_pipeline,
                    language
                )
            )
        for f in as_completed(futures):
            idx, chunk_text = f.result()
            results[idx] = chunk_text

    full_text = " ".join(results[i] for i in sorted(results.keys())).strip()
    total_time = time.time() - start_time
    print(f"[TRANSCRIBE] Finished all chunks in {total_time:.2f}s.")
    return full_text


def extract_audio_from_video(input_file, output_ext=".mp3"):
    """
    Extracts audio from a video file using pydub/ffmpeg. Returns a path to a temporary audio file.
    """
    print(f"[EXTRACT] Extracting audio from video file: {input_file}")
    tmp_dir = tempfile.TemporaryDirectory()
    output_path = os.path.join(tmp_dir.name, f"extracted_audio{output_ext}")
    audio = AudioSegment.from_file(input_file)
    audio.export(output_path, format=output_ext.replace(".", ""))
    print(f"[EXTRACT] Audio extracted to: {output_path}")
    return output_path, tmp_dir


# --- Streamlit Frontend ---
def main():
    st.title("Audio/Video to Text + Multi-Step Analysis")

    # Initialize session state for storing results
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'gemini_model' not in st.session_state:
        st.session_state.gemini_model = None
    if 'asr_pipeline' not in st.session_state:
        st.session_state.asr_pipeline = None
    if 'current_whisper_model' not in st.session_state:
        st.session_state.current_whisper_model = "large"

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Audio Processing Settings
        st.subheader("Audio Processing")
        
        # Whisper Model Selection
        whisper_col1, whisper_col2 = st.columns([3, 1])
        with whisper_col1:
            selected_whisper = st.selectbox(
                "Whisper Model",
                options=list(WHISPER_MODELS.keys()),
                index=list(WHISPER_MODELS.keys()).index(st.session_state.current_whisper_model),
                help="Select the Whisper model size for transcription"
            )
        with whisper_col2:
            st.markdown(f"<div style='margin-top: 35px;'><em>{WHISPER_MODELS[selected_whisper]['params']}</em></div>", unsafe_allow_html=True)
        
        # Show model description
        st.info(WHISPER_MODELS[selected_whisper]['description'])
        
        # Load new ASR pipeline if model changed
        if selected_whisper != st.session_state.current_whisper_model:
            with st.spinner(f"Loading Whisper {selected_whisper} model..."):
                st.session_state.asr_pipeline = load_asr_pipeline(selected_whisper)
                st.session_state.current_whisper_model = selected_whisper
        
        chunk_length_ms = st.slider(
            "Chunk length (milliseconds)", 
            5000, 60000, 29000, 
            step=1000,
            help="Length of audio chunks for processing. Smaller chunks use less memory but may reduce accuracy."
        )
        force_language = st.text_input(
            "Force language code",
            help="Optional: Force a specific language (e.g., 'en' for English, 'es' for Spanish). Leave blank for automatic detection."
        )

        # Analysis Settings
        st.subheader("Analysis Settings")
        analysis_type = st.radio(
            "Choose Analysis Method",
            ["none", "ollama", "gemini"],
            format_func=lambda x: {
                "none": "No Analysis (Raw Text Only)",
                "ollama": "Ollama (Local Models)",
                "gemini": "Google Gemini API"
            }[x],
            help="Select the method for analyzing the transcribed text"
        )

        # Model Selection based on analysis type
        selected_model_name = None
        if analysis_type == "ollama":
            all_ollama_models = list_ollama_models()
            if all_ollama_models:
                selected_model_name = st.selectbox(
                    "Choose Ollama Model",
                    all_ollama_models,
                    help="Select a local Ollama model for text analysis"
                )
            else:
                st.error("No Ollama models found.")
        elif analysis_type == "gemini":
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Google Gemini API key for cloud-based analysis"
            )
            if gemini_api_key and not st.session_state.gemini_model:
                try:
                    st.session_state.gemini_model = init_gemini(gemini_api_key)
                    st.success("Gemini API configured successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize Gemini: {e}")

        # Custom Prompts
        if analysis_type != "none":
            st.subheader("Analysis Prompts")
            show_custom_prompts = st.checkbox(
                "Customize Analysis Prompts",
                help="Customize the prompts used for each analysis step"
            )
            
            if show_custom_prompts:
                system_prompts = {
                    "summary": st.text_area(
                        "Summary Prompt",
                        value="You are a helpful assistant who summarizes text. Please produce a concise, structured summary in clear paragraphs.",
                        height=100,
                        help="Prompt for generating the initial summary"
                    ),
                    "tasks": st.text_area(
                        "Tasks Prompt",
                        value="Based on the summary provided, create a list of actionable tasks for the team. Format each task with priority (High/Medium/Low), estimated effort (in hours), and clear description.",
                        height=100,
                        help="Prompt for extracting actionable tasks"
                    ),
                    "key_points": st.text_area(
                        "Key Points Prompt",
                        value="Extract the key discussion points and decisions from the text. Format them as bullet points with categories.",
                        height=100,
                        help="Prompt for identifying key points and decisions"
                    )
                }
            else:
                system_prompts = None

    # Get the current ASR pipeline
    if st.session_state.asr_pipeline is None:
        with st.spinner(f"Loading Whisper {st.session_state.current_whisper_model} model..."):
            st.session_state.asr_pipeline = load_asr_pipeline(st.session_state.current_whisper_model)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio or video file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "mov", "avi", "mpeg", "mkv"],
        help="Select an audio or video file for transcription"
    )

    if uploaded_file is not None:
        # Only process if we haven't already processed this file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}_{st.session_state.current_whisper_model}"
        if 'last_processed_file' not in st.session_state or st.session_state.last_processed_file != file_key:
            # Save to a temp file
            print(f"[UPLOAD] Received file: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(suffix=uploaded_file.name, delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_input_path = tmp_file.name
            print(f"[UPLOAD] Saved upload to temp file: {temp_input_path}")

            # Determine if it's a video or an audio container
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext in [".mp4", ".mov", ".avi", ".mpeg", ".mkv"]:
                st.write("Extracting audio from video...")
                audio_path, tmp_dir_ref = extract_audio_from_video(temp_input_path, output_ext=".mp3")
            else:
                audio_path = temp_input_path
                tmp_dir_ref = None

            st.write("**Transcribing...**")
            with st.spinner(f"Running Whisper {st.session_state.current_whisper_model} transcription..."):
                st.session_state.transcription = transcribe_long_audio_parallel(
                    file_path=audio_path,
                    chunk_length_ms=chunk_length_ms,
                    language=force_language if force_language.strip() else None,
                    max_workers=4,
                    asr_pipeline=st.session_state.asr_pipeline
                )

            st.success("Transcription complete!")

            # Perform analysis if selected
            if analysis_type != "none":
                st.write(f"**Performing {analysis_type.title()} Analysis...**")
                with st.spinner("Running analysis..."):
                    st.session_state.analysis_results = perform_multi_step_analysis(
                        text=st.session_state.transcription,
                        analysis_type=analysis_type,
                        model_name=selected_model_name,
                        gemini_model=st.session_state.gemini_model,
                        system_prompts=system_prompts
                    )
            else:
                st.session_state.analysis_results = {
                    "raw_text": st.session_state.transcription,
                    "summary": None,
                    "tasks": None,
                    "key_points": None
                }

            # Clean up
            if tmp_dir_ref:
                tmp_dir_ref.cleanup()
                print("[CLEANUP] Temporary directory cleaned up.")
            
            # Store the file key to prevent reprocessing
            st.session_state.last_processed_file = file_key

        # Display results from session state
        st.markdown("### Raw Transcription")
        st.write(st.session_state.transcription)

        if analysis_type != "none" and st.session_state.analysis_results:
            # Create tabs for different analysis results
            tabs = st.tabs(["Summary", "Tasks", "Key Points"])
            
            with tabs[0]:
                st.markdown("### Summary")
                if st.session_state.analysis_results["summary"]:
                    st.write(st.session_state.analysis_results["summary"])
            
            with tabs[1]:
                st.markdown("### Tasks")
                if st.session_state.analysis_results["tasks"]:
                    st.write(st.session_state.analysis_results["tasks"])
            
            with tabs[2]:
                st.markdown("### Key Points")
                if st.session_state.analysis_results["key_points"]:
                    st.write(st.session_state.analysis_results["key_points"])

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Raw Transcription",
                st.session_state.transcription.encode("utf-8"),
                file_name="transcription_raw.txt",
                mime="text/plain",
            )
        
        if analysis_type != "none" and st.session_state.analysis_results:
            with col2:
                # Combine all analysis results into one formatted text
                analysis_text = "# Analysis Results\n\n"
                if st.session_state.analysis_results["summary"]:
                    analysis_text += "## Summary\n" + st.session_state.analysis_results["summary"] + "\n\n"
                if st.session_state.analysis_results["tasks"]:
                    analysis_text += "## Tasks\n" + st.session_state.analysis_results["tasks"] + "\n\n"
                if st.session_state.analysis_results["key_points"]:
                    analysis_text += "## Key Points\n" + st.session_state.analysis_results["key_points"]
                
                st.download_button(
                    "Download Analysis Results",
                    analysis_text.encode("utf-8"),
                    file_name="analysis_results.md",
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()
