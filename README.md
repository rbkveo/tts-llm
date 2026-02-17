# Audio/Video Transcription and Analysis Tool

A powerful Streamlit application that combines OpenAI's Whisper models for audio transcription with advanced text analysis capabilities using either local Ollama models or Google's Gemini API.

## Features

### Transcription
- Support for multiple audio/video formats (MP3, WAV, M4A, OGG, FLAC, MP4, MOV, AVI, MPEG, MKV)
- Multiple Whisper model options:
  - **Tiny** (39M parameters): Fastest, good for testing, ~1GB VRAM
  - **Base** (74M parameters): Fast, decent accuracy, ~1GB VRAM
  - **Small** (244M parameters): Balanced speed/accuracy, ~2GB VRAM
  - **Medium** (769M parameters): Better accuracy, good for multiple languages, ~5GB VRAM
  - **Large** (1.5B parameters): Best accuracy, production quality, ~10GB VRAM
- Automatic language detection or forced language mode
- **Voice Activity Detection (VAD)**: Use Silero VAD to split audio based on speech segments, improving accuracy by avoiding splits in the middle of sentences.
- Parallel processing of audio segments for improved performance
- Video audio extraction support

### Analysis
Three analysis options:
1. **Raw Text Only**: Just transcription without analysis
2. **Ollama (Local Models)**: Use local LLMs for analysis
3. **Google Gemini API**: Cloud-based analysis using Google's Gemini Pro

### Multi-Step Analysis Pipeline
When using Ollama or Gemini, the tool performs three analysis steps:
1. **Summary Generation**: Creates a concise, structured summary
2. **Task Extraction**: Generates actionable tasks with priorities and effort estimates
3. **Key Points**: Identifies main discussion points and decisions
4. **Hallucination Detection**: Verifies factual consistency between the summary and the raw transcription, flagging potential inconsistencies.

### Additional Features
- Customizable analysis prompts for each step
- **Text-to-Speech (TTS)**: Listen to generated summaries using high-quality AI voices.
- Adjustable audio chunk size for memory optimization
- Progress indicators and detailed feedback
- Download options for both raw transcription and analysis results
- Markdown-formatted output for analysis results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transcriber
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- CUDA-compatible GPU recommended for faster transcription
- Ollama (if using local models)
- Gemini API key (if using Google Gemini)

### Dependencies
```
streamlit
transformers
torch
pydub
google-generativeai
ollama
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface (typically http://localhost:8501)

3. Configure settings in the sidebar:
   - Select Whisper model size
   - Choose analysis method
   - Configure language settings
   - Set up model-specific options

4. Upload an audio/video file and wait for processing

5. View and download results

## Configuration

### Whisper Models
- Select based on your needs and hardware capabilities
- Larger models require more VRAM but provide better accuracy
- Model loading occurs once per session unless changed

### Analysis Options

#### Ollama
- Requires Ollama installed and running locally
- Available models are automatically detected
- Custom prompts available for each analysis step

#### Google Gemini
- Requires API key
- Generally faster than local models
- Better for complex analysis tasks

### Customization
- Adjust chunk size for memory/speed trade-off
- Customize analysis prompts for each step
- Force specific language for transcription
- Configure worker count for parallel processing

## Output Formats

### Raw Transcription
- Plain text format
- Downloaded as .txt file

### Analysis Results
- Markdown formatted
- Includes:
  - Summary section
  - Prioritized tasks
  - Key discussion points
- Downloaded as .md file

## Best Practices

1. **Model Selection**:
   - Start with smaller models for testing
   - Use larger models for production/important transcriptions
   - Consider VRAM requirements

2. **Audio Processing**:
   - Use smaller chunk sizes for limited memory
   - Ensure good audio quality for better results
   - Consider pre-processing audio for noisy recordings

3. **Analysis**:
   - Use Ollama for privacy-sensitive content
   - Use Gemini for complex analysis needs
   - Customize prompts for specific use cases

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce chunk size
   - Use a smaller Whisper model
   - Reduce number of parallel workers

2. **Slow Processing**:
   - Check GPU availability
   - Adjust chunk size
   - Consider model size trade-offs

3. **Model Loading Errors**:
   - Ensure sufficient disk space
   - Check internet connection
   - Verify CUDA installation (if using GPU)

### Error Messages

- "No Ollama models found": Install Ollama and download required models
- "Failed to initialize Gemini": Check API key and internet connection
- "CUDA out of memory": Reduce model size or chunk size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 