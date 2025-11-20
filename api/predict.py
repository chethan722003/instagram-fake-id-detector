import os
import pandas as pd
import base64
import requests
import instaloader
from urllib.parse import urlparse
import joblib
import json

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

def extract_username(url):
    return urlparse(url).path.strip('/').split('/')[0]

def handler(request):
    """This function replaces Flask for Vercel."""
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    try:
        body = json.loads(request.body.decode())
        url = body.get("url")

        if not url:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No URL provided"})
            }

        username = extract_username(url)
        if not username:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid Instagram URL"})
            }

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

        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        result = "Fake Account" if prediction == 1 else "Real Account"

        # Profile picture base64
        img_base64 = None
        try:
            resp = requests.get(profile.profile_pic_url)
            img_base64 = base64.b64encode(resp.content).decode("utf-8")
            img_base64 = f"data:image/jpeg;base64,{img_base64}"
        except:
            pass

        profile_data = {
            "username": profile.username,
            "full_name": profile.full_name,
            "biography": profile.biography,
            "external_url": profile.external_url,
            "is_private": profile.is_private,
            "is_verified": profile.is_verified,
            "profile_pic_url": img_base64,
            "#posts": profile.mediacount,
            "#followers": profile.followers,
            "#follows": profile.followees
        }

        return {
            "statusCode": 200,
            "body": json.dumps({
                "result": result,
                "profile": profile_data
            }),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
