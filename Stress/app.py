# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load models

def load_models():
    if not os.path.exists('stress_model.pkl') or not os.path.exists('label_encoder.pkl'):
        raise FileNotFoundError("Model files not found. Please run train_model.py first.")
    try:
        model = joblib.load('stress_model.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    except Exception as e:
        raise RuntimeError(f"Error loading model files: {str(e)}")

try:
    model, le = load_models()
except Exception as e:
    print(str(e))
    exit(1)

# Database setup

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists('database.db'):
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE stress_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                study_hours INTEGER NOT NULL,
                sleep_hours INTEGER NOT NULL,
                physical_activity INTEGER NOT NULL,
                social_support INTEGER NOT NULL,
                stress_level TEXT NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()
        conn.close()

init_db()

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        email = request.form['email']
        age = request.form.get('age')
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password, email, age) VALUES (?, ?, ?, ?)',
                        (username, password, email, age))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    stress_history = conn.execute('''SELECT stress_level, COUNT(*) as count FROM stress_records WHERE user_id = ? GROUP BY stress_level''', (session['user_id'],)).fetchall()
    conn.close()
    return render_template('profile.html', user=user, stress_history=stress_history)

@app.route('/stress_test', methods=['GET', 'POST'])
def stress_test():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            study_hours = int(request.form['study_hours'])
            sleep_hours = int(request.form['sleep_hours'])
            physical_activity = int(request.form['physical_activity'])
            social_support = int(request.form['social_support'])
            if not (0 <= study_hours <= 24 and 0 <= sleep_hours <= 24 and 1 <= physical_activity <= 5 and 1 <= social_support <= 5):
                flash('Invalid input values.', 'danger')
                return redirect(url_for('stress_test'))
            input_data = pd.DataFrame([[age, study_hours, sleep_hours, physical_activity, social_support]],
                                      columns=['age', 'study_hours', 'sleep_hours', 'physical_activity', 'social_support'])
            prediction = model.predict(input_data)
            stress_level = le.inverse_transform(prediction)[0]
            conn = get_db_connection()
            conn.execute('''INSERT INTO stress_records (user_id, study_hours, sleep_hours, physical_activity, social_support, stress_level) VALUES (?, ?, ?, ?, ?, ?)''',
                         (session['user_id'], study_hours, sleep_hours, physical_activity, social_support, stress_level))
            conn.commit()
            conn.close()
            flash(f'Your stress level is: {stress_level}', 'info')
            return redirect(url_for('dashboard'))
        except ValueError:
            flash('Please enter valid numbers.', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    return render_template('stress_test.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        stress_records = conn.execute('SELECT * FROM stress_records WHERE user_id = ? ORDER BY recorded_at DESC LIMIT 5', (session['user_id'],)).fetchall()
        conn.close()
        recommendations = get_recommendations(stress_records[0]['stress_level'] if stress_records else None)
        return render_template('dashboard.html', user=user, recommendations=recommendations)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/manage')
def manage():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        conn = get_db_connection()
        all_records = conn.execute('SELECT * FROM stress_records WHERE user_id = ? ORDER BY recorded_at DESC', (session['user_id'],)).fetchall()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        dates = [record['recorded_at'] for record in all_records]
        stress_levels = [record['stress_level'] for record in all_records]
        study_hours = [record['study_hours'] for record in all_records]
        sleep_hours = [record['sleep_hours'] for record in all_records]
        recommendations = get_recommendations(stress_levels[0] if stress_levels else None)
        return render_template('manage.html', user=user, dates=dates, stress_levels=stress_levels, study_hours=study_hours, sleep_hours=sleep_hours, recommendations=recommendations)
    except Exception as e:
        flash(f'Error loading data: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

def get_recommendations(stress_level):
    if not stress_level:
        return None
    if stress_level == 'High':
        return {
            'music': [{'title': 'Weightless', 'artist': 'Marconi Union', 'link': '#'}],
            'quotes': ["You are stronger than you think."],
            'activities': ["Practice deep breathing"]
        }
    elif stress_level == 'Medium':
        return {
            'music': [{'title': 'Lofi Chill', 'artist': 'Various Artists', 'link': '#'}],
            'quotes': ["Keep going, you're halfway there!"],
            'activities': ["Take a short walk"]
        }
    else:
        return {
            'music': [{'title': 'Here Comes the Sun', 'artist': 'The Beatles', 'link': '#'}],
            'quotes': ["Keep up the good work!"],
            'activities': ["Continue your routine"]
        }

if __name__ == '__main__':
    app.run(debug=True)
