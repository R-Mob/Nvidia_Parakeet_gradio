import gradio as gr
import nemo.collections.asr as nemo_asr
import time
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import subprocess
import sys
from datetime import datetime

# Global variable to track if transcription is in progress
is_transcribing = False

def convert_to_mono_wav(input_file, output_file=None):
    """Convert any audio file to mono WAV format at 16kHz"""
    if output_file is None:
        # Create a temporary file in the same directory
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file), f"{base_name}_mono.wav")
    
    print(f"Converting {input_file} to mono WAV format...")
    
    try:
        # Load audio file with librosa (handles various formats)
        audio, sr = librosa.load(input_file, sr=16000, mono=True)
        
        # Save as WAV file
        sf.write(output_file, audio, sr, subtype='PCM_16')
        print(f"Converted audio saved to {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return None

def check_audio_format(audio_path):
    """Check audio format and convert if needed"""
    try:
        # Get audio information
        info = sf.info(audio_path)
        channels = info.channels
        sample_rate = info.samplerate
        
        print(f"Audio file: {audio_path}")
        print(f"Channels: {channels}, Sample rate: {sample_rate} Hz")
        
        # Check if conversion is needed
        needs_conversion = channels > 1 or sample_rate != 16000
        
        if needs_conversion:
            print("Audio needs conversion (mono 16kHz required)")
            converted_file = convert_to_mono_wav(audio_path)
            return converted_file
        else:
            print("Audio format is compatible (mono, 16kHz)")
            return audio_path
    except Exception as e:
        print(f"Error checking audio format: {str(e)}")
        print("Attempting conversion anyway...")
        return convert_to_mono_wav(audio_path)

def get_default_output_file(audio_path):
    """Generate default output filename based on input audio file"""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), f"{base_name}_transcript_{timestamp}.txt")

def transcribe_audio(audio_path, use_timestamps=False, optimize_for_long_audio=False, 
                     output_file=None, attention_window=128, keep_temp_files=False):
    """Transcribe audio with automatic format conversion and save to text file"""
    
    # Set default output file if none specified
    if output_file is None:
        output_file = get_default_output_file(audio_path)
    
    # Check GPU
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  GPU {i}: {gpu_name} ({total_mem:.2f} GB)")
    else:
        print("No GPU found. Running on CPU (not recommended - will be very slow)")
    
    # Check and convert audio format if needed
    compatible_audio = check_audio_format(audio_path)
    
    if compatible_audio is None:
        print("Error: Could not prepare audio for transcription")
        return None, None
    
    # Load the model (will download if not already cached)
    print("\nLoading Parakeet-TDT-0.6b-v2 model (this may take a minute the first time)...")
    start_load = time.time()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    end_load = time.time()
    print(f"Model loaded in {end_load - start_load:.2f} seconds")
    
    # Optimize for long audio if requested
    if optimize_for_long_audio:
        print("Applying optimizations for long audio files...")
        asr_model.change_attention_model("rel_pos_local_attn", [attention_window, attention_window])
        asr_model.change_subsampling_conv_chunking_factor(1)
        print(f"Set attention window to {attention_window}")
    
    # Print audio file info
    file_size_mb = os.path.getsize(compatible_audio) / (1024 * 1024)
    print(f"\nProcessing: {compatible_audio} ({file_size_mb:.2f} MB)")
    
    # Transcribe
    print("Beginning transcription...")
    start_time = time.time()
    transcript_text = ""
    
    try:
        output = asr_model.transcribe([compatible_audio], timestamps=use_timestamps)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        print("\n----- TRANSCRIPTION RESULTS -----")
        print(output[0].text)
        print(f"\nTranscription completed in {execution_time:.2f} seconds")
        
        # Format transcript
        transcript_text = f"# Transcription of: {os.path.basename(audio_path)}\n"
        transcript_text += f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        transcript_text += f"# Duration: {execution_time:.2f} seconds\n"
        transcript_text += f"# Model: Parakeet-TDT-0.6b-v2\n\n"
        transcript_text += output[0].text
        
        # Add timestamps if requested
        if use_timestamps:
            transcript_text += "\n\n----- SEGMENT TIMESTAMPS -----\n"
            segment_timestamps = output[0].timestamp['segment']
            for i, stamp in enumerate(segment_timestamps):
                transcript_text += f"{i+1}. {stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['segment']}\n"
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
                    
        print(f"Transcript saved to: {output_file}")
                
    except RuntimeError as e:
        error_msg = str(e)
        print(f"\nError during transcription: {error_msg}")
        
        if "CUDA out of memory" in error_msg:
            transcript_text = "Error: CUDA out of memory. Try:\n"
            transcript_text += "  1. Use 'Optimize for long audio' option\n"
            transcript_text += "  2. Use a smaller attention window\n"
            transcript_text += "  3. Process a shorter audio file\n"
            transcript_text += "  4. Use a GPU with more memory"
        else:
            transcript_text = f"Error during transcription: {error_msg}"
            
        # Try to save error log
        error_log = f"{os.path.splitext(output_file)[0]}_error.log"
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"Error during transcription: {error_msg}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio file: {audio_path}\n")
            if has_gpu:
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            else:
                f.write("Running on CPU\n")
        print(f"Error log saved to: {error_log}")
        
        # Return None for the file path if an error occurred
        output_file = None
            
    except Exception as e:
        error_msg = str(e)
        print(f"\nError during transcription: {error_msg}")
        transcript_text = f"Error during transcription: {error_msg}"
        output_file = None
    
    # Clean up temporary files if needed
    if not keep_temp_files and compatible_audio != audio_path and compatible_audio is not None:
        print(f"Removing temporary file: {compatible_audio}")
        try:
            os.remove(compatible_audio)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")
            
    return transcript_text, output_file

def download_youtube_audio(youtube_url, output_dir="youtube_downloads"):
    """Download audio from YouTube video"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"youtube_{timestamp}.wav")
    
    try:
        # Check if yt-dlp is installed
        try:
            subprocess.run(["yt-dlp", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Installing yt-dlp...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=True)
        
        # Command to download just the audio and convert to WAV
        cmd = [
            "yt-dlp",
            "-x",                         # Extract audio
            "--audio-format", "wav",      # Convert to WAV
            "--audio-quality", "0",       # Best quality
            "--postprocessor-args", "-ac 1 -ar 16000",  # Convert to mono 16kHz
            "-o", output_file,            # Output file
            youtube_url                   # YouTube URL
        ]
        
        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if file exists or has a different extension
        if os.path.exists(output_file):
            return output_file
        
        # Sometimes yt-dlp adds an extension even though format is specified
        if os.path.exists(output_file + ".wav"):
            return output_file + ".wav"
        
        # Search the output directory for the file
        for file in os.listdir(output_dir):
            if file.startswith(os.path.basename(output_file).split('.')[0]) and file.endswith(".wav"):
                return os.path.join(output_dir, file)
        
        return None
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        print(f"Command output: {e.stdout.decode() if hasattr(e, 'stdout') else ''}")
        print(f"Command error: {e.stderr.decode() if hasattr(e, 'stderr') else ''}")
        return None

def process_upload(audio_file, include_timestamps, optimize_long, attention_window):
    """Process uploaded audio file and return transcript + download link"""
    global is_transcribing
    
    if audio_file is None:
        return "Please upload an audio file.", None
    
    if is_transcribing:
        return "Transcription already in progress. Please wait.", None
    
    is_transcribing = True
    
    try:
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcript_{timestamp}.txt"
        
        # Run transcription
        transcript, file_path = transcribe_audio(
            audio_path=audio_file,
            use_timestamps=include_timestamps,
            optimize_for_long_audio=optimize_long,
            output_file=output_file,
            attention_window=attention_window,
            keep_temp_files=False
        )
        
        if file_path and os.path.exists(file_path):
            return transcript, file_path
        else:
            return transcript, None
    
    except Exception as e:
        return f"Error during transcription: {str(e)}", None
    
    finally:
        is_transcribing = False

def process_youtube(youtube_url, include_timestamps, optimize_long, attention_window):
    """Process YouTube URL and return transcript + download link"""
    global is_transcribing
    
    if not youtube_url or not youtube_url.strip():
        return "Please enter a valid YouTube URL.", None
    
    if is_transcribing:
        return "Transcription already in progress. Please wait.", None
    
    is_transcribing = True
    
    try:
        # First download the audio
        print(f"Downloading audio from: {youtube_url}")
        audio_file = download_youtube_audio(youtube_url)
        
        if not audio_file or not os.path.exists(audio_file):
            return f"Error: Failed to download audio from {youtube_url}", None
        
        print(f"Downloaded audio file: {audio_file}")
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"youtube_transcript_{timestamp}.txt"
        
        # Run transcription
        transcript, file_path = transcribe_audio(
            audio_path=audio_file,
            use_timestamps=include_timestamps,
            optimize_for_long_audio=optimize_long,
            output_file=output_file,
            attention_window=attention_window,
            keep_temp_files=False
        )
        
        if file_path and os.path.exists(file_path):
            return transcript, file_path
        else:
            return transcript, None
    
    except Exception as e:
        return f"Error processing YouTube URL: {str(e)}", None
    
    finally:
        is_transcribing = False
        # Clean up downloaded file
        try:
            if 'audio_file' in locals() and os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass

# Create a very simple Gradio interface
with gr.Blocks(title="Parakeet Transcription Tool") as demo:
    gr.Markdown(
        """
        # 🦜 Parakeet-TDT-0.6b-v2 Transcription Tool
        
        This simplified interface transcribes audio files or YouTube videos.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Upload Audio File"
                    )
                    
                    with gr.Row():
                        timestamps_checkbox1 = gr.Checkbox(
                            label="Include timestamps",
                            value=True
                        )
                        
                        optimize_checkbox1 = gr.Checkbox(
                            label="Optimize for long audio",
                            value=True
                        )
                    
                    attention_slider1 = gr.Slider(
                        minimum=32,
                        maximum=256,
                        value=128,
                        step=32,
                        label="Attention Window Size"
                    )
                    
                    transcribe_btn1 = gr.Button("Transcribe Audio", variant="primary")
            
            with gr.Row():
                transcript_output1 = gr.Textbox(
                    label="Transcription Result",
                    lines=15
                )
                
            with gr.Row():
                download_output1 = gr.File(
                    label="Download Transcript"
                )
        
        with gr.TabItem("YouTube"):
            with gr.Row():
                with gr.Column():
                    youtube_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=..."
                    )
                    
                    with gr.Row():
                        timestamps_checkbox2 = gr.Checkbox(
                            label="Include timestamps",
                            value=True
                        )
                        
                        optimize_checkbox2 = gr.Checkbox(
                            label="Optimize for long audio",
                            value=True
                        )
                    
                    attention_slider2 = gr.Slider(
                        minimum=32,
                        maximum=256,
                        value=128,
                        step=32,
                        label="Attention Window Size"
                    )
                    
                    transcribe_btn2 = gr.Button("Transcribe from YouTube", variant="primary")
            
            with gr.Row():
                transcript_output2 = gr.Textbox(
                    label="Transcription Result",
                    lines=15
                )
                
            with gr.Row():
                download_output2 = gr.File(
                    label="Download Transcript"
                )
    
    # Set up event handlers
    transcribe_btn1.click(
        fn=process_upload,
        inputs=[audio_input, timestamps_checkbox1, optimize_checkbox1, attention_slider1],
        outputs=[transcript_output1, download_output1]
    )
    
    transcribe_btn2.click(
        fn=process_youtube,
        inputs=[youtube_url, timestamps_checkbox2, optimize_checkbox2, attention_slider2],
        outputs=[transcript_output2, download_output2]
    )
    
    # Add troubleshooting info
    with gr.Accordion("Usage Tips", open=False):
        gr.Markdown(
            """
            ## Usage Tips
            
            1. **Upload Audio Tab**:
               - Supports most audio formats (WAV, MP3, FLAC, etc.)
               - For longer audio files, use the "Optimize for long audio" option
            
            2. **YouTube Tab**:
               - Enter a complete YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)
               - The system will download, convert, and transcribe the audio
            
            3. **Options**:
               - "Include timestamps" adds timing information to the transcript
               - "Optimize for long audio" reduces memory usage but might slightly reduce accuracy
               - "Attention Window Size" controls the optimization level (smaller values use less memory)
            
            4. **Transcripts**:
               - After transcription completes, a download link will appear automatically
               - Files are named with timestamps to avoid overwriting
            
            ## Troubleshooting
            
            If you encounter issues:
            
            1. Check the terminal for detailed error messages
            2. Start with a short audio file to verify functionality
            3. For memory errors, reduce the Attention Window Size
            4. For YouTube errors, make sure yt-dlp is installed (`pip install yt-dlp`)
            """
        )

# Launch the interface
if __name__ == "__main__":
    # Launch with minimal configuration
    demo.launch(server_name="127.0.0.1", server_port=7860)