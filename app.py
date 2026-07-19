import os
import warnings
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
import database as db
from database import (init_db,get_db_connection,get_cursor,get_current_user,is_admin_user,row_to_dict,DB_AVAILABLE,DB_TYPE,)
init_db()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "plant-secret-key")

model_path = "model/plant_model.keras"
class_names_path = "class_names.npy"

db.init_db()
DB_AVAILABLE = db.DB_AVAILABLE
DB_TYPE = db.DB_TYPE
get_db_connection = db.get_db_connection
get_cursor = db.get_cursor
row_to_dict = db.row_to_dict
get_current_user = db.get_current_user
is_admin_user = db.is_admin_user

def extract_plant_disease(class_name):
    if class_name.startswith("Tomato_"):
        plant = "Tomato"
        disease = class_name[len("Tomato_"):].lstrip("_")
    else:
        plant = "Unknown"
        disease = class_name

    disease = disease.replace("_", " ")
    disease = disease.title()
    return plant, disease


PLANT_KEYWORDS = [
    "leaf", "plant", "veget", "flower", "tree", "corn", "cabbage",
    "cauliflower", "broccoli", "artichoke", "cardoon", "squash",
    "cucumber", "pepper", "mushroom", "greenhouse", "hay", "rapeseed",
    "daisy", "fungus", "agaric", "moss", "fern", "bush", "shrub",
    "herb", "stem", "bud", "seedling", "sprout", "garden", "fruit",
    "berry", "gourd", "pumpkin", "tomato", "potato", "bell pepper"
]

DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "severity": "High",
        "treatment": "Apply copper bactericide.",
        "prevention": "Avoid handling wet plants."
    },
    "Tomato_Early_blight": {
        "severity": "Medium",
        "treatment": "Apply fungicide and remove infected leaves.",
        "prevention": "Mulch and stake plants."
    },
    "Tomato_Late_blight": {
        "severity": "Critical",
        "treatment": "Apply metalaxyl immediately.",
        "prevention": "Destroy infected plants."
    },
    "Tomato_Leaf_Mold": {
        "severity": "Medium",
        "treatment": "Apply copper fungicide.",
        "prevention": "Reduce humidity and improve airflow."
    },
    "Tomato_Septoria_leaf_spot": {
        "severity": "Medium",
        "treatment": "Remove infected leaves and use fungicide.",
        "prevention": "Avoid overhead watering."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Medium",
        "treatment": "Apply neem oil or insecticidal soap.",
        "prevention": "Maintain proper humidity."
    },
    "Tomato_healthy": {
        "severity": "Healthy",
        "treatment": "No treatment needed.",
        "prevention": "Maintain regular watering and nutrition."
    },

    "Tomato__Target_Spot": {
        "severity": "High",
        "treatment": "Use fungicide and remove infected leaves.",
        "prevention": "Improve field sanitation."
    },
    "Tomato__Tomato_mosaic_virus": {
        "severity": "High",
        "treatment": "No cure. Remove infected plants.",
        "prevention": "Disinfect tools and hands."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "Critical",
        "treatment": "Remove infected plants immediately.",
        "prevention": "Control whitefly populations."
    }
}


if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}. Please run train.py first to train and save the model."
    )

if not os.path.exists(class_names_path):
    raise FileNotFoundError(
        f"Class names not found at {class_names_path}. Please run train.py first."
    )

print("Loading model and class names...")
model = None
class_names = None
gate_model = None

tomato_indices = None
try:
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    class_names = np.load(class_names_path, allow_pickle=True)
    tomato_indices = np.array(
        [i for i, name in enumerate(class_names) if str(name).startswith("Tomato_")]
    )
    if len(tomato_indices) == 0:
        print("Warning: no Tomato_ classes found in class_names.npy")
    print("✓ Model and class names loaded successfully")
except Exception as exc:
    print(f"Warning: Unable to load TensorFlow Keras model now: {exc}")

print("Loading gate model (generic ImageNet classifier for leaf detection)...")
try:
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input as gate_preprocess_input,
        decode_predictions as gate_decode_predictions,
    )
    gate_model = MobileNetV2(weights="imagenet", include_top=True)
    print("✓ Gate model loaded successfully")
except Exception as exc:
    gate_model = None
    print(f"Warning: Unable to load gate model: {exc}")


def looks_like_plant(pil_img, top_k=5):
    """Return True if the photo's top ImageNet predictions look plant/leaf
    related. If the gate model isn't available, default to True (skip the
    check) rather than blocking predictions entirely."""
    if gate_model is None:
        return True

    img_resized = pil_img.resize((224, 224))
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = gate_preprocess_input(arr)

    preds = gate_model.predict(arr, verbose=0)
    decoded = gate_decode_predictions(preds, top=top_k)[0]

    for (_, label, _score) in decoded:
        label_lower = label.replace("_", " ").lower()
        if any(keyword in label_lower for keyword in PLANT_KEYWORDS):
            return True
    return False


@app.route("/")
def landing():
    user = get_current_user()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("landing.html", user=user, db_status=DB_AVAILABLE)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            flash("Please fill in all fields.", "danger")
            return render_template("register.html", user=get_current_user(), db_status=DB_AVAILABLE)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html", user=get_current_user(), db_status=DB_AVAILABLE)

        if not DB_AVAILABLE:
            flash("Database is not available right now. Please try again later.", "danger")
            return render_template("register.html", user=get_current_user(), db_status=DB_AVAILABLE)

        conn = get_db_connection()
        cursor = get_cursor(conn, dictionary=(DB_TYPE == "mysql"))
        if DB_TYPE == "mysql":
            cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        else:
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            conn.close()
            flash("That username or email already exists.", "danger")
            return render_template("register.html", user=get_current_user(), db_status=DB_AVAILABLE)

        if DB_TYPE == "mysql":
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                (username, email, generate_password_hash(password), "user"),
            )
        else:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (username, email, generate_password_hash(password), "user"),
            )
        conn.commit()
        conn.close()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", user=get_current_user(), db_status=DB_AVAILABLE)


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter your username and password.", "danger")
            return render_template("login.html", user=get_current_user(), db_status=DB_AVAILABLE)

        if not DB_AVAILABLE:
            flash("Database is not available right now. Please try again later.", "danger")
            return render_template("login.html", user=get_current_user(), db_status=DB_AVAILABLE)

        conn = get_db_connection()
        cursor = get_cursor(conn, dictionary=(DB_TYPE == "mysql"))
        if DB_TYPE == "mysql":
            cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
        else:
            cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
        user = row_to_dict(cursor.fetchone())
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            flash("Login successful.", "success")
            if str(user["username"]).lower() == "admin" or str(user["role"]).lower() == "admin":
                session["role"] = "admin"
                return redirect(url_for("dashboard"))
            session["role"] = "user"
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html", user=get_current_user(), db_status=DB_AVAILABLE)


@app.route("/dashboard")
@app.route("/index")
def dashboard():
    user = get_current_user()
    if not user:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html", db_status=DB_AVAILABLE)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/admin")
def admin_dashboard():
    user = get_current_user()
    if not user:
        flash("Please log in to access the admin panel.", "warning")
        return redirect(url_for("login"))

    if not is_admin_user(user):
        flash("Access denied. Admin only.", "danger")
        return redirect(url_for("dashboard"))

    if not DB_AVAILABLE:
        flash("Database is not available right now.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cursor = get_cursor(conn, dictionary=(DB_TYPE == "mysql"))
    if DB_TYPE == "mysql":
        cursor.execute("SELECT id, username, email, role FROM users ORDER BY id")
        users = cursor.fetchall()
        cursor.execute(
            """
            SELECT reports.id, users.username, reports.plant, reports.disease, reports.confidence,
                   reports.severity, reports.treatment, reports.prevention, reports.created_at
            FROM reports
            JOIN users ON users.id = reports.user_id
            ORDER BY reports.created_at DESC
            """
        )
        report_rows = cursor.fetchall()
    else:
        cursor.execute("SELECT id, username, email, role FROM users ORDER BY id")
        users = [row_to_dict(row) for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT reports.id, users.username, reports.plant, reports.disease, reports.confidence,
                   reports.severity, reports.treatment, reports.prevention, reports.created_at
            FROM reports
            JOIN users ON users.id = reports.user_id
            ORDER BY reports.created_at DESC
            """
        )
        report_rows = [row_to_dict(row) for row in cursor.fetchall()]
    conn.close()

    return render_template("admin.html", user=user, users=users, reports=report_rows, db_status=DB_AVAILABLE)


@app.route("/reports")
def report():
    user = get_current_user()
    if not user:
        flash("Please log in to view your reports.", "warning")
        return redirect(url_for("login"))

    if not DB_AVAILABLE:
        flash("Database is not available right now.", "danger")
        return redirect(url_for("landing"))

    conn = get_db_connection()
    cursor = get_cursor(conn, dictionary=(DB_TYPE == "mysql"))
    if is_admin_user(user):
        cursor.execute(
            """
            SELECT reports.id, users.username, reports.plant, reports.disease, reports.confidence,
                   reports.severity, reports.treatment, reports.prevention, reports.created_at
            FROM reports
            JOIN users ON users.id = reports.user_id
            ORDER BY reports.created_at DESC
            """
        )
    else:
        if DB_TYPE == "mysql":
            cursor.execute(
                "SELECT id, plant, disease, confidence, severity, treatment, prevention, created_at FROM reports WHERE user_id = %s ORDER BY created_at DESC",
                (user["id"],),
            )
        else:
            cursor.execute(
                "SELECT id, plant, disease, confidence, severity, treatment, prevention, created_at FROM reports WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
    report_rows = [row_to_dict(row) for row in cursor.fetchall()]
    conn.close()

    return render_template("report.html",user=user,reports=report_rows, db_status=DB_AVAILABLE)

@app.route("/report")
def reports():
    return redirect(url_for("report"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user = get_current_user()
        if not user:
            flash("Please sign in to detect plant diseases.", "warning")
            return redirect(url_for("login"))

        file = request.files.get("file")
        if not file or file.filename == '':
            flash("No file selected.", "warning")
            return redirect(url_for("dashboard"))

        if model is None or class_names is None:
            flash("Model is not available. Please ensure TensorFlow is installed and the model file exists.", "danger")
            return redirect(url_for("landing"))

        if tomato_indices is None or len(tomato_indices) == 0:
            flash("Tomato classes not found in the model. Please retrain with Tomato data.", "danger")
            return redirect(url_for("landing"))

        img = Image.open(file).convert("RGB")

      
        if not looks_like_plant(img):
            flash("This doesn't look like a leaf photo. Please upload a clear photo of a tomato leaf.", "warning")
            return redirect(url_for("dashboard"))

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]

       
        tomato_scores = prediction[tomato_indices]
        best_local_index = int(np.argmax(tomato_scores))
        index = int(tomato_indices[best_local_index])
        confidence = float(tomato_scores[best_local_index] * 100)

        if confidence < 50:
            flash("Not confident this is a tomato leaf disease. Please try a clearer, closer photo.", "warning")
            return redirect(url_for("dashboard"))

        predicted_class = str(class_names[index])
        plant, disease = extract_plant_disease(predicted_class)

        info = DISEASE_INFO.get(
            predicted_class,
            {
                "severity": "Unknown",
                "treatment": "Consult an agricultural expert.",
                "prevention": "Monitor plant regularly."
            }
        )

        result = {
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2),
            "severity": info["severity"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        }

        if user and DB_AVAILABLE:
            conn = get_db_connection()
            cursor = get_cursor(conn)
            if DB_TYPE == "mysql":
                cursor.execute(
                    "INSERT INTO reports (user_id, plant, disease, confidence, severity, treatment, prevention) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        user["id"],
                        result["plant"],
                        result["disease"],
                        result["confidence"],
                        result["severity"],
                        result["treatment"],
                        result["prevention"],
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO reports (user_id, plant, disease, confidence, severity, treatment, prevention) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user["id"],
                        result["plant"],
                        result["disease"],
                        result["confidence"],
                        result["severity"],
                        result["treatment"],
                        result["prevention"],
                    ),
                )
            conn.commit()
            conn.close()
            flash("Prediction saved to your reports.", "success")

        return render_template("dashboard.html", user=user, result=result, db_status=DB_AVAILABLE)

    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)