## Video Lecture Transcription Tool

<img width="1036" height="624" alt="image" src="https://github.com/user-attachments/assets/2b78d9d6-8464-42f4-9371-29acfdd9c43a" />

### Overview

This script is designed to transcribe video lectures (.mp4) into text files using OpenAI Whisper.
It processes videos in chunks, converts speech to text, and outputs:

* `.txt` — readable transcript
* `.srt` — timestamped transcript (one sentence per line)

The system supports parallel processing of multiple videos to improve throughput.

* Designed specifically for lecture transcription
* Optimized for long videos (1–2 hours)
* Works best with clear speech audio

---

### Features

* 🎧 Extracts audio from video using FFmpeg (no intermediate files)
* 🧠 Transcribes speech using Whisper (medium model by default)
* ✂️ Splits audio into manageable chunks (streaming processing)
* 📝 Outputs clean sentence-based transcription
* ⚡ Parallel processing of multiple videos
* 📊 Supports folder input

---

### Configuration (Constants)

#### Core Parameters

* CHUNK_SECONDS = 60
    Duration of each audio chunk (in seconds).
    Smaller values → better accuracy, higher overhead.
* bytes_per_second = 16000 * 2
    Audio format:
    * 16kHz sample rate
    * 16-bit (2 bytes per sample)
        Used to calculate chunk size in bytes.
* BUFFER_SIZE = 20 * 1024 * 1024 (20MB)
    Buffer size for FFmpeg stdout pipe.
    Prevents blocking between FFmpeg and Python.
  
---

### Model

* MODEL = whisper.load_model("medium")
    Whisper model used for transcription.

Options:

* `tiny` → fastest, least accurate
* `base` → faster
* `medium` → balanced (recommended)
* `large` → best quality, very slow

---

### Input Folder

* default `input/output` **FOLDER** = `/videos`
    Directory containing `.mp4` lecture videos.
    Output files are written to the same folder (or you can change it)

---

### Transcription Settings

#### Inside `transcribe_stream()`:

* task=`transcribe`
    Ensures speech is transcribed, not translated.
* language=`ru`
    Forces language detection (can be changed to `en`, etc.).
* temperature=`0.2`
    Controls randomness:
    * `0.0` → strict, literal transcription
    * higher → more flexible / creative
* `condition_on_previous_text=False`
    Prevents Whisper from rewriting earlier text across chunks.
* `verbose=False`
    Disables Whisper console logs (cleaner output).

---

#### Processing Pipeline

1. FFmpeg stream
    * Extracts audio from video
    * Converts to mono, 16kHz PCM
2. Chunking
    * Audio is read in fixed-size chunks (60 seconds)
3. Normalization
    * audio = int16 → float32 → scaled to [-1, 1]
4. Transcription
    * Each chunk is processed independently
5. Sentence reconstruction
    * Segments are buffered
    * Regex detects sentence endings (. ! ?)
    * Ensures one line per full sentence
6. Output
    * `.txt` → plain text
    * `.srt` → timestamped sentences

---

Usage

1. Place `.mp4` lecture videos in the target folder:
    * `/videos` (or somewhere else)

2. Run the script:
    * `python viseo_transcribing.py`

3. Outputs:
    * `video1.txt`
    * `video1.srt`

