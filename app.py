import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import database as db
import detection

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_IMAGE = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_AUDIO = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB uploads

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


class User(UserMixin):
    def __init__(self, row):
        self.id = row['id']
        self.username = row['username']


@login_manager.user_loader
def load_user(user_id):
    row = db.get_user_by_id(user_id)
    return User(row) if row else None


def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not username or not password:
            flash('Username and password are required.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif db.get_user_by_username(username):
            flash('That username is already taken.', 'danger')
        else:
            db.create_user(username, generate_password_hash(password))
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        row = db.get_user_by_username(username)

        if row and check_password_hash(row['password_hash'], password):
            login_user(User(row))
            flash(f'Welcome back, {row["username"]}!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    result = None
    active_tab = request.form.get('media_type', 'image')

    if request.method == 'POST':
        media_type = request.form.get('media_type')
        file = request.files.get('media_file')

        if not file or file.filename == '':
            flash('Please choose a file to upload.', 'warning')
            return redirect(url_for('dashboard'))

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(UPLOAD_DIR, unique_name)

        if media_type == 'image' and allowed_file(filename, ALLOWED_IMAGE):
            file.save(upload_path)
            out_path, label, confidence, explanation = detection.predict_image(upload_path, OUTPUT_DIR)
            rel_output = os.path.relpath(out_path, STATIC_DIR).replace(os.sep, '/') if out_path else None
            db.add_history(current_user.id, 'image', filename, rel_output, label, confidence, explanation)
            result = {'type': 'image', 'label': label, 'confidence': confidence,
                      'explanation': explanation, 'output': rel_output}

        elif media_type == 'audio' and allowed_file(filename, ALLOWED_AUDIO):
            file.save(upload_path)
            label, confidence, explanation = detection.predict_audio(upload_path)
            db.add_history(current_user.id, 'audio', filename, None, label, confidence, explanation)
            result = {'type': 'audio', 'label': label, 'confidence': confidence, 'explanation': explanation}

        elif media_type == 'video' and allowed_file(filename, ALLOWED_VIDEO):
            file.save(upload_path)
            out_path, label, confidence, explanation = detection.process_video(upload_path, OUTPUT_DIR)
            rel_output = os.path.relpath(out_path, STATIC_DIR).replace(os.sep, '/') if out_path else None
            db.add_history(current_user.id, 'video', filename, rel_output, label, confidence, explanation)
            result = {'type': 'video', 'label': label, 'confidence': confidence,
                      'explanation': explanation, 'output': rel_output}
        else:
            flash('Unsupported file type for the selected category.', 'danger')

        active_tab = media_type

    return render_template('dashboard.html', result=result, active_tab=active_tab)


@app.route('/history')
@login_required
def history():
    rows = db.get_history(current_user.id)
    return render_template('history.html', rows=rows)


@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    db.clear_history(current_user.id)
    flash('Your scan history has been cleared.', 'success')
    return redirect(url_for('history'))


if __name__ == '__main__':
    db.init_db()
    app.run(host='127.0.0.1', port=5000, debug=True)
