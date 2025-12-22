import sounddevice as sd
import soundfile as sf
import time
import os

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 8000
DURATION = 0.01
minutes=0.1

# Open output file for writing
output_file = sf.SoundFile('data.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)
# Start recording audio
start = time.time()
with sd.InputStream(channels=CHANNELS, blocksize=BLOCKSIZE, samplerate=SAMPLE_RATE) as stream:
    while True:
        audio_data, _ = stream.read(BLOCKSIZE)

        # Append audio to output file
        output_file.write(audio_data)

        # Wait for half a second
        time.sleep(DURATION)

        # Append half a second of audio to output file
        audio_data, _ = stream.read(int(DURATION * SAMPLE_RATE))
        output_file.write(audio_data)
        end = time.time()

        if end-start >= minutes*60:
            output_file.close()
            os.system("mv data.wav data.pth")
            break