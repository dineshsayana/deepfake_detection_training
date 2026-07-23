# Unified Real-Time Deepfake Detector — Flask Edition

A Flask rewrite of the original Gradio app, with user accounts, a dashboard, and
persistent scan history stored in SQLite.

## Features
- **Register / Login / Logout** — session-based auth via Flask-Login, passwords hashed with Werkzeug.
- **Dashboard** — tabbed UI (Image / Audio / Video) to upload a file and run deepfake detection.
- **History** — every scan is saved per-user (type, result, confidence, explanation, output preview) with a **Clear History** option.
- **SQLite** — zero-config database file (`app_data.db`), created automatically on first run.
- **Light Bootstrap 5 theme** — clean, minimal styling in `static/css/style.css`.

## Project structure
```
deepfake_flask_app/
├── app.py              # Flask routes (auth, dashboard, history)
├── database.py         # SQLite helper functions
├── detection.py         # Image/audio/video inference logic (ported from gradoapp.py)
├── requirements.txt
├── models/              # put your .h5 model files here (optional — falls back to mock mode)
├── static/
│   ├── css/style.css
│   ├── uploads/          # uploaded files land here
│   └── outputs/          # annotated images / re-encoded videos land here
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    └── history.html
```

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Add your trained models**
   Place `deepfake_image_model.h5`, `deepfake_audio_model.h5`, and `deepfake_video_model.h5`
   inside the `models/` folder. If they're missing, the app automatically runs in
   **mock mode** (returns REAL with 0% confidence) so the UI is still fully testable.

4. **FFmpeg** (only needed for video re-encoding to browser-compatible H.264)
   Install it and make sure it's on your PATH:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: download from ffmpeg.org and add to PATH

5. **Run the app**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## Notes
- The SQLite database (`app_data.db`) is created automatically on first run — no manual migration needed.
- `MAX_CONTENT_LENGTH` is set to 100 MB for uploads; adjust in `app.py` if needed.
- Change `app.config['SECRET_KEY']` (or set the `SECRET_KEY` env var) before deploying anywhere beyond localhost.
- For production, run behind a WSGI server (e.g. `gunicorn app:app`) rather than `app.run(debug=True)`.
