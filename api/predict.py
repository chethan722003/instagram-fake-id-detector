from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ✅ Correct model path (works on Vercel)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

def to_int_or_default(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

@app.route("/api/predict_manual", methods=["POST"])
def predict_manual():
    try:
        # ✅ Extract form data
        username = (request.form.get("username") or "").strip()
        full_name = (request.form.get("full_name") or "").strip()
        biography = request.form.get("biography") or ""
        external_url = request.form.get("external_url") or ""
        is_private = request.form.get("is_private") == "on"
        is_verified = request.form.get("is_verified") == "on"
        profile_pic_url = request.form.get("profile_pic_url") or ""
        posts = to_int_or_default(request.form.get("posts"), 0)
        followers = to_int_or_default(request.form.get("followers"), 0)
        follows = to_int_or_default(request.form.get("follows"), 0)

        username_len = len(username) or 1
        fullname_len = len(full_name) or 1

        features = {
            'profile pic': 1 if profile_pic_url else 0,
            'nums/length username': sum(1 for c in username if c.isdigit()) / username_len,
            'fullname words': len(full_name.split()) if full_name else 0,
            'nums/length fullname': sum(1 for c in full_name if c.isdigit()) / fullname_len,
            'name==username': int(full_name.lower() == username.lower()) if username and full_name else 0,
            'description length': len(biography),
            'external URL': 1 if external_url else 0,
            'private': int(is_private),
            '#posts': posts,
            '#followers': followers,
            '#follows': follows
        }

        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        result = "Fake Account" if int(prediction) == 1 else "Real Account"

        return jsonify({
            "result": result,
            "username": username,
            "full_name": full_name,
            "followers": followers,
            "follows": follows,
            "posts": posts
        }), 200

    except Exception as e:
        # ✅ Always return JSON, even on failure
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ For local testing
if __name__ == "__main__":
    app.run(debug=True)
