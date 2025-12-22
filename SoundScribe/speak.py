from SoundScribe.translation import translate, reverse_translate
from playsound import playsound as ps
from gtts import gTTS
import time

with open("pwd.txt", 'r') as folder_location:
    folder_location = folder_location.read()

with open(f"{folder_location}database/lang.txt", 'r') as language:
    language = language.read()

def speak(text):
    text = reverse_translate(text)
    tts = gTTS(text, lang=language)
    print(text)
    tts.save(f'{folder_location}audio.mp3')
    ps(f'{folder_location}audio.mp3')
    time.sleep(4)
    with open(f"{folder_location}database/recognition.txt", 'w') as recognition:
        recognition.write('')
