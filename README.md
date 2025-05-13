🦜 Parakeet-TDT-0.6b-v2 Transcription Tool
A user-friendly interface for transcribing audio files and YouTube videos using NVIDIA's Parakeet-TDT-0.6b-v2 speech recognition model.
Overview
This tool provides a web interface for transcribing speech from various audio sources:

Upload audio files (WAV, MP3, FLAC, etc.)
Transcribe directly from YouTube video URLs
Automatically generate timestamps
Download transcription files with a single click

The system uses NVIDIA's Parakeet-TDT-0.6b-v2, a high-performance automatic speech recognition model capable of generating accurate transcriptions with punctuation, capitalization, and precise timestamps.
Installation
Prerequisites

Python 3.8 or higher
NVIDIA GPU with CUDA support (strongly recommended)
At least 4GB of VRAM (8GB+ recommended for longer audio files)

Setup

Clone or download this repository:
bashgit clone https://github.com/yourusername/parakeet-transcription.git
cd parakeet-transcription

Create a virtual environment (recommended):
bash# Using conda
conda create -n parakeet python=3.10
conda activate parakeet

# Or using venv
python -m venv parakeet-env
source parakeet-env/bin/activate  # On Windows: parakeet-env\Scripts\activate

Install dependencies:
bash# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install nemo_toolkit[all] gradio librosa soundfile

# Install yt-dlp for YouTube support
pip install yt-dlp


Usage

Start the application:
bashpython transcription_app.py

Access the web interface:
Open your browser and go to: http://127.0.0.1:7860
Using the interface:

Upload Audio tab: Upload audio files or record directly
YouTube tab: Enter YouTube video URLs for transcription
Configure options:

Include timestamps: Add word-level timing information
Optimize for long audio: Reduce memory usage for longer files
Attention Window Size: Adjust memory optimization level




After transcription:

View the transcription results in the text box
Download the transcript via the automatic download link
Transcripts are saved as text files with timestamps in the filename



Features

Audio Format Support: Automatically converts various audio formats to the required mono 16kHz WAV format
YouTube Integration: Download and transcribe audio directly from YouTube video URLs
Memory Optimization: Special options for handling longer audio files on limited GPU memory
Timestamps: Option to include detailed segment-level timestamps
Automatic Downloads: Transcripts are automatically available for download after processing
Error Handling: Detailed error reporting and troubleshooting information

Options Explained

Include timestamps: When enabled, generates timing information for each segment of speech
Optimize for long audio: Reduces memory usage by applying local attention mechanisms, recommended for files longer than 5 minutes
Attention Window Size: Controls the size of the attention window when optimization is enabled:

Larger values (128-256): Better accuracy but uses more memory
Smaller values (32-64): Uses less memory but may slightly reduce accuracy



Troubleshooting
Common Issues

"CUDA out of memory" errors:

Enable "Optimize for long audio"
Reduce the "Attention Window Size"
Try processing a shorter audio clip
Close other GPU-intensive applications


YouTube download fails:

Ensure yt-dlp is installed and up to date: pip install -U yt-dlp
Check that the YouTube URL is valid and the video is publicly accessible
Try downloading with a different tool and then uploading the audio file


Slow transcription:

The model requires a GPU for reasonable performance
CPU-only operation is possible but will be very slow
Ensure your GPU drivers are up to date


Audio format issues:

If audio conversion fails, install FFmpeg:
bash# Linux
sudo apt-get install ffmpeg

# Windows (with chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg




System Requirements

Minimum: NVIDIA GPU with 4GB VRAM, 8GB RAM, 5GB disk space
Recommended: NVIDIA GPU with 8GB+ VRAM, 16GB RAM, 10GB disk space
CPU-only: Possible but transcription will be very slow

Performance Benchmarks

Short audio clips (1-2 minutes): ~10-20 seconds processing time
Medium-length audio (5-10 minutes): ~1-2 minutes processing time
Long-form audio (30+ minutes): May require 5+ minutes processing time

Performance varies greatly based on GPU capabilities.
About the Parakeet-TDT-0.6b-v2 Model
Parakeet-TDT-0.6b-v2 is a 600-million-parameter automatic speech recognition model developed by NVIDIA:

Designed for high-quality English transcription
Features support for punctuation, capitalization, and accurate timestamp prediction
Optimized for Nvidia GPUs using TensorRT for fast inference
Released under CC-BY-4.0 license (permissive for commercial use)
Based on the FastConformer architecture with a TDT decoder

More information: Parakeet-TDT-0.6b-v2 on Hugging Face
License
This tool is provided under the MIT License. The Parakeet-TDT-0.6b-v2 model is licensed by NVIDIA under CC-BY-4.0.
Acknowledgments

NVIDIA for creating and open-sourcing the Parakeet-TDT-0.6b-v2 model
Hugging Face for hosting the model
The NeMo team for their work on ASR frameworks
Gradio team for the web interface framework


For issues, feature requests, or contributions, please open an issue or pull request on the GitHub repository.
