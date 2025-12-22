from googletrans import Translator

with open("pwd.txt", 'r') as folder_location:
    folder_location = folder_location.read()

with open(f"{folder_location}database/lang.txt", 'r') as language:
    language = language.read()

def translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src=language, dest='en').text
    return translate_text


def reverse_translate(text):
    translator = Translator()
    translate_text = translator.translate(text, src='en', dest=language).text
    return translate_text