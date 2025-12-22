import sounddevice as sd
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

# Set up the audio parameters
CHUNK = 1024
RATE = 44100
CHANNELS = 1

# Initialize the recording and playback streams
record_stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, blocksize=CHUNK)
playback_stream = sd.OutputStream(samplerate=RATE, channels=CHANNELS, blocksize=CHUNK)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_audio')
def start_audio():
    print("Listening...")

    def audio_generator():
        with record_stream:
            while True:
                # Read audio data from the microphone
                data, overflowed = record_stream.read(CHUNK)

                # Convert audio data to bytes and emit it to the client
                emit('audio_data', {'data': data.tobytes()})

    socketio.start_background_task(audio_generator)

if __name__ == "__main__":
    socketio.run(app, debug=True)
