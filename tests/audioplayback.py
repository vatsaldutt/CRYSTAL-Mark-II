import sounddevice as sd

def record_and_playback():
    # Set up the audio parameters
    CHUNK = 1024
    RATE = 44100
    CHANNELS = 1

    print("Listening... (Press Ctrl+C to stop)")

    try:
        # Initialize the recording and playback streams
        record_stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, blocksize=CHUNK)
        playback_stream = sd.OutputStream(samplerate=RATE, channels=CHANNELS, blocksize=CHUNK)

        with record_stream, playback_stream:
            while True:
                # Read audio data from the microphone
                data, overflowed = record_stream.read(CHUNK)

                # Play back the recorded audio
                playback_stream.write(data)

    except KeyboardInterrupt:
        # If Ctrl+C is pressed, stop the loop
        print("Stopped by user.")

if __name__ == "__main__":
    record_and_playback()
