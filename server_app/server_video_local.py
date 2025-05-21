from flask import Flask, render_template_string, send_from_directory, abort
import os

app = Flask(__name__)

# HTML template for directory listing
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Storage Directory</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .item { padding: 10px; border-bottom: 1px solid #eee; }
        .item:hover { background-color: #f5f5f5; }
        a { text-decoration: none; color: #333; }
        .folder { color: #2c5282; }
        .file { color: #2d3748; }
    </style>
</head>
<body>
    <h1>Storage Directory Contents</h1>
    <div class="container">
        {% if current_path != '' %}
        <div class="item">
            <a href="{{ '../' if current_path else '/' }}">..</a>
        </div>
        {% endif %}
        {% for item in items %}
        <div class="item">
            <a href="{{ item.path }}" class="{{ 'folder' if item.is_dir else 'file' }}">
                {{ '📁' if item.is_dir else '📄' }} {{ item.name }}
            </a>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

@app.route('/')
@app.route('/<path:path>')
def browse(path=''):
    # Construct absolute path
    abs_path = os.path.join('storage', path)
    
    try:
        # If path is a file, serve it
        if os.path.isfile(abs_path):
            return send_from_directory(
                os.path.abspath('storage'),
                path,
                as_attachment=False
            )
        
        # If path is a directory, show listing
        if os.path.isdir(abs_path):
            items = []
            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                items.append({
                    'name': item,
                    'path': os.path.join(path, item) if path else item,
                    'is_dir': os.path.isdir(item_path)
                })
            
            # Sort items (directories first, then files)
            items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
            
            return render_template_string(HTML_TEMPLATE, items=items, current_path=path)
        
        abort(404)
    except Exception as e:
        abort(500, str(e))

if __name__ == '__main__':
    # Create storage directory if it doesn't exist
    storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
    os.makedirs(storage_path, exist_ok=True)
    
    # Run the server
    app.run(host='0.0.0.0', port=7780, debug=False)
