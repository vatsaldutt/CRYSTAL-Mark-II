from flask import Flask, request, jsonify, render_template, send_file
from flask_ngrok import run_with_ngrok
import threading
import CRYSTAL

app = Flask(__name__)
run_with_ngrok(app)   


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reply.txt')
def serve_reply():
    return send_file('reply.txt')

@app.route('/write-file', methods=['POST'])
def write_file():
    data = request.get_json()
    filename = data.get('filename')
    file_content = data.get('data')

    try:
        with open(filename, 'w') as file:
            file.write(file_content)
        return 'File written successfully', 200
    except Exception as e:
        return str(e), 500

@app.route('/process-query', methods=['POST'])
def process_query():
    query = request.json['query']

    answer = 'This is the answer for the query: ' + query

    return jsonify({'answer': answer})

def flask_app_runner():
    app.run()

def answer():
    while True:
        with open('query.txt', 'r') as query_file:
            query_data = query_file.read()
        if query_data != "":
            CRYSTAL.ask_crystal(query_data)
            with open("query.txt", "w") as clear_file:
                clear_file.write("")
        else:
            pass



if __name__ == '__main__':
    flask_thread = threading.Thread(target=flask_app_runner)
    flask_thread.start()

    answer()
