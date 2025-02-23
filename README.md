# Call Recording Transcription with Speaker Detection

This script transcribes audio recordings of business calls and identifies speakers (Agent/Customer) using WhisperX, which combines Whisper's speech recognition with speaker diarization.

## Features

- Accurate transcription using WhisperX (based on OpenAI's Whisper model)
- Speaker diarization to distinguish between Agent and Customer
- Timestamps for each segment of speech
- JSON output format for easy integration

## Prerequisites

- Python 3.12 or higher
- HuggingFace account and authentication token
- Sufficient disk space (about 3GB for model files)

## Installation

1. Install the required packages:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Get your HuggingFace authentication token:

   - Go to [HuggingFace](https://huggingface.co/)
   - Create an account or sign in
   - Go to Settings -> Access Tokens
   - Create a new token
   - Accept the license agreement for pyannote/speaker-diarization

3. Replace the `HF_TOKEN` in `transcribe.py` with your token

## Usage

1. Place your audio files in the `/Users/username/Desktop/call_recordings/` directory
   (or update the `AUDIO_FOLDER` path in the script)

2. Run the script:
   ```bash
   python3 transcribe.py
   ```

The script will:

1. Load the WhisperX model (first run will download ~3GB of model files)
2. Transcribe the audio
3. Perform speaker diarization
4. Save the results in JSON format

## Output Format

The script creates a JSON file with this structure:

```json
[
  {
    "speaker": "CUSTOMER",
    "start": "0:00:01",
    "end": "0:00:05",
    "text": "Hello, James speaking."
  },
  {
    "speaker": "AGENT",
    "start": "0:00:05",
    "end": "0:00:10",
    "text": "Hi James, this is Kathy from Future Solutions."
  }
]
```

## Notes

- Processing time can be significant on CPU (30+ minutes for short calls)
- First run will download necessary model files (~3GB)
- Uses float32 computation for compatibility
- Automatically detects and labels speakers as AGENT or CUSTOMER
- Transcripts are saved in the `transcripts` subdirectory
# whisper_diarization
