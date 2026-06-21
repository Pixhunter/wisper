import subprocess
import numpy as np
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import whisper
import logging

CHUNK_SECONDS = 60  # 1 minute
bytes_per_second = 16000 * 2  # 16kHz * 2 bytes (int16)
BUFFER_SIZE = 20 * 1024 * 1024 # 20MB
MAX_WORKERS = 3
MODEL = whisper.load_model("medium")
FOLDER = '/videos'


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)


def read_audio_chunks(process): 
    chunk_size = bytes_per_second * CHUNK_SECONDS
    while True:
        raw_audio = process.stdout.read(chunk_size)
        if not raw_audio:
            break

        audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32)
        audio = audio_float32 / 32768.0 # normalization for int16 range ≈ [-1.0, 1.0]

        yield audio
        
def create_ffmpeg_stream(video_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # drop video stream
        "-ac", "1",              # mono audio
        "-ar", "16000",          # 16kHz (Whisper-friendly)
        "-acodec", "pcm_s16le",  # raw PCM
        "-f", "s16le",           # raw output format
        "-loglevel", "error",    # clean logs
        "-"
    ]

    process = subprocess.Popen( 
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        bufsize=BUFFER_SIZE
    )

    return process

class TranscriptWriter:
    def __init__(self, txt_path, srt_path):
        self.txt_file = open(txt_path, "a", encoding="utf-8")
        self.srt_file = open(srt_path, "a", encoding="utf-8")
        self.srt_index = 1

        self.buffer = ""
        self.buffer_start = None


    def format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02}:{minutes:02}:{secs:02}"


    def write_segment(self, start, end, text):
        text = text.strip()
        if not text:
            return

        if self.buffer_start is None:
            self.buffer_start = start

        self.buffer += " " + text

        # Detect full sentences
        sentences = re.split(r'(?<=[.!?])\s+', self.buffer)

        # Keep last unfinished part in buffer
        complete_sentences = sentences[:-1]
        self.buffer = sentences[-1]

        for sentence in complete_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # ---- Plain readable transcript ----
            self.txt_file.write("- " + text.strip() + "\n")
            self.txt_file.flush()

            start_ts = self.format_timestamp(self.buffer_start)
            end_ts = self.format_timestamp(end)

            # ---- Write ONE LINE PER SENTENCE ----
            line = f"[{start_ts} - {end_ts}] {sentence}\n"

            self.srt_file.write(line)
            self.srt_file.flush()

            self.srt_index += 1

            # reset start for next sentence
            self.buffer_start = end
        

    def close(self):
        if self.buffer.strip():
            line = f"[{self.format_timestamp(self.buffer_start)} - ?] {self.buffer.strip()}\n"
            self.txt_file.write(line)
            self.srt_file.write(line)

        self.txt_file.close()
        self.srt_file.close()


def transcribe_stream(video_path):
    offset = 0
    process = create_ffmpeg_stream(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    writer = TranscriptWriter(
        txt_path= FOLDER + "/" + filename + ".txt",
        srt_path= FOLDER + "/" + filename + ".srt",
    )

    for chunk in read_audio_chunks(process):
        result = MODEL.transcribe(
            chunk,
            task="transcribe",
            language="ru",
            temperature=0.2,
            condition_on_previous_text=False,
            verbose=False
        )

        for seg in result["segments"]:
            start = seg["start"] + offset
            end = seg["end"] + offset
            text = seg["text"]

            writer.write_segment(start, end, text)

        offset += CHUNK_SECONDS

    logging.info(f"File succesfully transcribed {filename}")
    writer.close()


def get_video_files(folder):
    return list(Path(folder).glob("*.mp4"))


def process_video(video_path):
    start = time.time()
    transcribe_stream(video_path)
    duration = time.time() - start

    return str(video_path), duration


def process_folder(folder):
    files = get_video_files(folder)
    logging.info(f"Total files to work {len(files)}: {files}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, f) for f in files]
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                file, duration = future.result()
                logging.info(f"Done ({completed}/{len(files)}): {file} in {duration:.1f}s")

            except Exception as e:
                logging.info(f"Failed ({completed}/{len(files)}): {e}")


logging.info(f"start work with {MAX_WORKERS}")
if __name__ == "__main__":
    process_folder(FOLDER)
