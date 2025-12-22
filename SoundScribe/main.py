import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import time
import os

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 8000
DURATION = 0.5
THRESHOLD = 0.015
SILENT_THRESHOLD = 3
silence_duration = 0

model = whisper.load_model("base")

def transcribe():
    result = model.transcribe('recording.wav')
    print(result['text'])
    return result['text']

# Open output file for writing
output_file = sf.SoundFile('recording.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)
# Start recording audio
with sd.InputStream(channels=CHANNELS, blocksize=BLOCKSIZE, samplerate=SAMPLE_RATE) as stream:
    print("RECORDING NOW")
    while True:
        audio_data, _ = stream.read(BLOCKSIZE)

        # Append audio to output file
        output_file.write(audio_data)

        # Wait for half a second
        time.sleep(0.5)

        # Append half a second of audio to output file
        audio_data, _ = stream.read(int(DURATION * SAMPLE_RATE))
        output_file.write(audio_data)

        # Check if audio is not quiet and transcribe it
        print(np.abs(audio_data).mean())
        if float(np.abs(audio_data).mean()) > THRESHOLD:
            print('Audio detected! Transcribing...')
            transcribe()
            # Call OpenAI Whisper library to transcribe audio from output file
            # Code to transcribe audio using Whisper goes here
        elif float(np.abs(audio_data).mean()) < THRESHOLD:
            silence_duration += BLOCKSIZE / float(SAMPLE_RATE)
            if silence_duration >= SILENT_THRESHOLD:
                print("You are done speaking")
                silence_duration = 0
                output_file.close()
                os.remove('recording.wav')
                audio_data = None
                output_file = sf.SoundFile('recording.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)