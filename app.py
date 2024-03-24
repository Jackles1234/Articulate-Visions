from flask import Flask, request, jsonify, send_file
from main import update_variables
import sys
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)

@app.route('/api/images')
def get_images():
    image_names = os.listdir('imgs/')
    images = [{'name': name, 'url': f'/api/images/{name}'} for name in image_names]
    return jsonify(images)

@app.route('/api/images/<image_name>')
def get_image(image_name):
    requested_path = os.path.abspath(os.path.join('imgs/', image_name))
    if not requested_path.startswith(os.path.abspath('imgs/')):
        return "Forbidden", 403
    return send_file(requested_path, mimetype='image/png')


@app.route('/api/submit', methods=['GET', 'POST'])
def submit_data():
    if request.method == 'POST':
        try:
            data = request.get_json()
            print("Received data:", data)

            textbox1 = data.get('textbox1')
            textbox2 = data.get('textbox2')
            textbox3 = data.get('textbox3')

            # Do something with the received data
            print("Textbox 1:", textbox1)
            print("Textbox 2:", textbox2)
            print("Textbox 3:", textbox3)
            update_variables(int(textbox1), int(textbox2), int(textbox3))
            return jsonify({'message': 'Data received successfully'}), 200
        except Exception as e:
            print("Error:", e)
            return jsonify({'message': 'Error processing data'}), 500
    else:
        return jsonify({'error': 'Request must contain JSON data'}), 400
if __name__ == '__main__':
    app.run(debug=True)