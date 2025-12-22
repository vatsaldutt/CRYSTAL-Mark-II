from SoundScribe.speakerID import find_user
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import time

model = whisper.load_model("base")

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()


def transcribe():
    result = model.transcribe('recording.wav')
    transcription = result['text']
    print(transcription)
    user = find_user("recording.wav")
    if user != "Crystal":
        with open(f'{folder_location}database/input.txt', 'w') as write_to:
            write_to.write(transcription[1:])
    return transcription


def listen(model):
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCKSIZE = 8000
    DURATION = 0.5
    THRESHOLD = 0.015
    SILENT_THRESHOLD = 3
    silence_duration = 0
    output_file = sf.SoundFile('recording.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)

    with sd.InputStream(channels=CHANNELS, blocksize=BLOCKSIZE, samplerate=SAMPLE_RATE) as stream:
        print("RECORDING NOW")
        while True:
            audio_data, _ = stream.read(BLOCKSIZE)

            output_file.write(audio_data)

            time.sleep(0.5)

            audio_data, _ = stream.read(int(DURATION * SAMPLE_RATE))
            output_file.write(audio_data)
            if float(np.abs(audio_data).mean()) > THRESHOLD:
                print('Audio detected! Transcribing...')
                transcribe()

            elif float(np.abs(audio_data).mean()) < THRESHOLD:
                silence_duration += BLOCKSIZE / float(SAMPLE_RATE)
                if silence_duration >= SILENT_THRESHOLD:
                    print("You are done speaking")
                    silence_duration = 0
                    output_file.close()
                    audio_data = None
                    output_file = sf.SoundFile(
                        'recording.wav', mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS)


if __name__ == "__main__":
    listen(model)
