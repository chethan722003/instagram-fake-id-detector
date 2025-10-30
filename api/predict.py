import os
import pandas as pd
import base64
import requests
import instaloader
from urllib.parse import urlparse
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/rf_model.pkl')
model = joblib.load(MODEL_PATH)

def extract_username(url):
    return urlparse(url).path.strip('/').split('/')[0]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data['url']
        username = extract_username(url)

        L = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(L.context, username)

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

        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        result = 'Fake Account' if prediction == 1 else 'Real Account'

        profile_data = {
            'username': profile.username,
            'full_name': profile.full_name,
            'biography': profile.biography,
            'external_url': profile.external_url,
            'is_private': profile.is_private,
            'is_verified': profile.is_verified,
            'profile_pic_url': profile.profile_pic_url,
            '#posts': profile.mediacount,
            '#followers': profile.followers,
            '#follows': profile.followees
        }

        return jsonify({'result': result, 'profile': profile_data})
    except Exception as e:
        return jsonify({'error': str(e)})

# Required for Vercel
def handler(event, context):
    return app(event, context)
