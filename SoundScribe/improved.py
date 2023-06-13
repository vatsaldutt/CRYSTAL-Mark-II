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

# Load the model
model = whisper.load_model("base")


def transcribe(file_path):
    result = model.transcribe(file_path)
    print(result['text'])
    return result['text']

# Open output file for writing


def create_output_file():
    return sf.SoundFile('recording.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)


def delete_output_file(file_path):
    os.remove(file_path)


def main():
    output_file = create_output_file()
    silence_duration = 0
    is_recording = False

    # Start recording audio
    with sd.InputStream(channels=CHANNELS, blocksize=BLOCKSIZE, samplerate=SAMPLE_RATE) as stream:
        print("RECORDING NOW")
        while True:
            try:
                audio_data, _ = stream.read(BLOCKSIZE)
                output_file.write(audio_data)

                # Check if audio is not quiet and transcribe it
                if float(np.abs(audio_data).mean()) > THRESHOLD:
                    if not is_recording:
                        is_recording = True
                    print('Audio detected! Transcribing...')
                    transcribe('recording.wav')
                elif is_recording:
                    silence_duration += BLOCKSIZE / float(SAMPLE_RATE)
                    if silence_duration >= SILENT_THRESHOLD:
                        print("You are done speaking")
                        is_recording = False
                        output_file.close()
                        delete_output_file('recording.wav')
                        output_file = create_output_file()
                        silence_duration = 0

            except KeyboardInterrupt:
                print("Recording stopped by user.")
                break

        # Close the output file if not already closed
        if not output_file.closed:
            output_file.close()

        # Delete the output file if it exists
        if os.path.exists('recording.wav'):
            delete_output_file('recording.wav')


if __name__ == "__main__":
    main()
