from flask import Flask, request, render_template, send_from_directory, url_for
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'storage'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No video file uploaded', 400
        
        video = request.files['video']
        if video.filename == '':
            return 'No selected file', 400
        
        if video:
            filename = video.filename
            video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return f'Video uploaded successfully. View at: /video/{filename}'
    
    # Show upload form for GET requests
    return '''
    <html>
    <head>
        <title>Upload Video</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            form { display: flex; flex-direction: column; gap: 10px; }
            input[type="submit"] { background: #4CAF50; color: white; padding: 10px; border: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h2>Upload Video</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

@app.route('/video/<filename>')
def video_page(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_exists = os.path.exists(video_path)
    
    return render_template('video.html', 
                         filename=filename,
                         video_exists=video_exists,
                         download_url=url_for('download_video', filename=filename) if video_exists else None)

@app.route('/storage/<filename>')
def download_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7781, debug=True)
