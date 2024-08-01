from flask import Flask, request, jsonify
from utils import analyze_frame, generate_narration, text_to_speech
from waitress import serve

app = Flask(__name__)

@app.route('/process-frame', methods=['POST'])
def process_frame():
    frame = request.data
    detection_result = analyze_frame(frame)
    narration = generate_narration(detection_result)
    audio_file = text_to_speech(narration)
    return jsonify({'audioFile': audio_file})

if __name__ == '__main__':
    # Use Waitress to serve the app
    serve(app, host='0.0.0.0', port=5000)
