from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from flask_cors import CORS
try:
    from model.main import generate_image  # This function needs to be defined in main.py
except ImportError:
    from backend.model.main import generate_image

app = Flask(__name__)
CORS(app)
# Define the directory to save and serve images
IMAGE_DIR = 'backend/imgs/'  # Adjust this as necessary


@app.route('/')
def home():
    return '<html><body><h1>test success</h1></body></html>'


@app.route('/api/images')
def get_images():
    image_names = os.listdir(IMAGE_DIR)
    images = [{'name': name, 'url': f'/api/images/{name}'} for name in image_names]
    return jsonify(images)


@app.route('/api/images/<image_name>')
def get_image(image_name):
    if not os.path.exists(os.path.join(IMAGE_DIR, image_name)):
        return "Image not found", 404
    return send_from_directory(IMAGE_DIR, image_name)


@app.route('/api/submit', methods=['GET', 'POST'])
def submit_data():
    if request.method == 'POST':
        data = request.get_json()
        prompt = data.get('prompt')
        batch_size = data.get('batch_size', 1)  # Default to 1 if not specified
        guidance_scale = data.get('guidance_scale', 1.0)  # Default to 1.0 if not specified
        head_channels = data.get('head_channels', 8)
        xf_heads = data.get('xf_heads', 64)

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        try:
            batch_size = int(batch_size)
            guidance_scale = float(guidance_scale)
            head_channels = int(head_channels)
            xf_heads = int(xf_heads)
        except ValueError:
            return jsonify({'error': 'Invalid batch size or guidance scale'}), 400

        # Call the model script to generate images
        image_paths = generate_image(prompt, batch_size, guidance_scale, head_channels, xf_heads)

        return send_file(image_paths, mimetype='image/png')


def main():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
