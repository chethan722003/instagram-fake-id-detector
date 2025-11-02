import os
import pandas as pd
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
import instaloader
from urllib.parse import urlparse
import joblib

app = Flask(__name__)

# Model path (compatible with Vercel and local)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

def extract_username(url):
    """Extract the Instagram username from a profile URL."""
    return urlparse(url).path.strip('/').split('/')[0]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        username = extract_username(url)
        if not username:
            return jsonify({'error': 'Invalid Instagram URL'}), 400

        # Load profile
        L = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(L.context, username)

        # Extract features
        features = {
            'profile pic': 1 if profile.profile_pic_url else 0,
            'nums/length username': sum(c.isdigit() for c in profile.username) / len(profile.username),
            'fullname words': len(profile.full_name.split()),
            'nums/length fullname': sum(c.isdigit() for c in profile.full_name) / (len(profile.full_name) or 1),
            'name==username': int(profile.full_name.lower() == profile.username.lower()),
            'description length': len(profile.biography),
            'external URL': 1 if profile.external_url else 0,
            'private': int(profile.is_private),
            '#posts': profile.mediacount,
            '#followers': profile.followers,
            '#follows': profile.followees
        }

        # Prediction
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        result = 'Fake Account' if prediction == 1 else 'Real Account'

        # Profile picture as base64
        img_base64 = None
        try:
            response = requests.get(profile.profile_pic_url)
            img_base64 = base64.b64encode(response.content).decode('utf-8')
            img_base64 = f"data:image/jpeg;base64,{img_base64}"
        except Exception:
            pass

        # Profile data
        profile_data = {
            'username': profile.username,
            'full_name': profile.full_name,
            'biography': profile.biography,
            'external_url': profile.external_url,
            'is_private': profile.is_private,
            'is_verified': profile.is_verified,
            'profile_pic_url': img_base64,
            '#posts': profile.mediacount,
            '#followers': profile.followers,
            '#follows': profile.followees
        }

        return jsonify({'result': result, 'profile': profile_data})

    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Serve frontend for testing (optional for local)
@app.route('/')
def home():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_path, 'index.html')


# ❌ Don’t use app.run() — Vercel handles that automatically.
if __name__ == '__main__':
    app.run(debug=True)
