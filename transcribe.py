import os
import json
from datetime import timedelta
import whisperx
import torch

# Input and output paths
AUDIO_FOLDER = "/Users/saeed/Desktop/call_recordings"
OUTPUT_FOLDER = os.path.join(AUDIO_FOLDER, "transcripts")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# HuggingFace token for diarization
HF_TOKEN = "YOUR_TOKEN"

def process_audio_file(audio_path):
    """Process a single audio file with transcription and speaker identification."""
    try:
        print("Loading WhisperX model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float32"  # Use float32 instead of float16
        
        # Load model
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        
        # Transcribe audio
        print("Transcribing audio...")
        result = model.transcribe(audio_path, batch_size=16)
        
        # Align whisper output
        print("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
        
        # Get speaker diarization
        print("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio_path)
        
        # Assign speaker labels
        print("Assigning speaker labels...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format output
        formatted_segments = []
        for segment in result["segments"]:
            # Map speaker IDs to AGENT/CUSTOMER (SPEAKER_00 is usually the first speaker)
            speaker = "CUSTOMER" if segment.get("speaker", "SPEAKER_00") == "SPEAKER_00" else "AGENT"
            
            formatted_segments.append({
                "speaker": speaker,
                "start": str(timedelta(seconds=segment["start"])),
                "end": str(timedelta(seconds=segment["end"])),
                "text": segment["text"].strip()
            })
        
        return formatted_segments
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        raise

def main():
    # Process single file
    audio_path = "/Users/saeed/Desktop/call_recordings/James Gillies.mp3"
    output_path = os.path.join(OUTPUT_FOLDER, "James Gillies.json")
    
    try:
        print(f"Processing: {os.path.basename(audio_path)}")
        
        # Process the audio file
        transcript = process_audio_file(audio_path)
        
        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Transcription saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 