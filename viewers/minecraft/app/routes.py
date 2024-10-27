from flask import Blueprint, send_from_directory, jsonify, render_template, abort
import os
import logging

main = Blueprint('main', __name__)

FILES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'static', 'schematics', 'static')
EXP_FILES_DIRECTORY = '../../../exp/icl_0512/outputs'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@main.route('/')
def home():
    """Serve the main HTML page."""
    return render_template('index.html')

def find_json_files(directory):
    """Recursively find all .json files in the directory."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                json_files.append(relative_path)
    return json_files

@main.route('/api/files')
def list_files():
    """API endpoint to return a list of .json files."""
    files = find_json_files(FILES_DIRECTORY)
    return jsonify(files)

@main.route('/api/file-directory')
def get_file_directory():
    """API endpoint to return the main file directory to search for .litematic files."""
    return jsonify(directory=FILES_DIRECTORY)

@main.route('/files/<filename>')
def serve_file(filename):
    """Serve a specific .json file."""
    if filename.endswith('.json') and os.path.exists(os.path.join(FILES_DIRECTORY, filename)):
        return send_from_directory(FILES_DIRECTORY, filename)
    abort(404)


