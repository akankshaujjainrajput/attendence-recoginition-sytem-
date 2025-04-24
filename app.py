

from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_mysqldb import MySQL
from datetime import datetime, timedelta
import face_recognition
import pandas as pd
import numpy as np
import cv2
import pickle
import os
import threading
import imghdr  # For image validation
import time
import traceback  # ✅ Add this to capture full error logs
import csv
from flask_mail import Mail, Message



app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'test@123#@'
app.config['MYSQL_DB'] = 'face_recognition'
mysql = MySQL(app)


# ✅ Flask-Mail Configuration (Gmail)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kumarraushan620257@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'sidl kcfo jhzg xemk'  # Use your generated App Password
app.config['MAIL_DEFAULT_SENDER'] = 'kumarraushan620257@gmail.com'  # ✅ Add this line
mail = Mail(app)

# Model Paths
MODEL_DIR = os.path.join(os.getcwd(), "models")
DEPLOY_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Ensure the model files exist
if not os.path.exists(DEPLOY_PROTO) or not os.path.exists(CAFFE_MODEL):
    raise FileNotFoundError("Model files missing! Ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' are in the 'models' folder.")

# Load the Caffe face detection model
face_detector = cv2.dnn.readNetFromCaffe(DEPLOY_PROTO, CAFFE_MODEL)
print("✅ Face detection model loaded successfully!")

# Folder Paths
BASE_FOLDER = os.getcwd()
EMPLOYEE_FOLDER = os.path.join(BASE_FOLDER, "images", "employees")
STUDENT_FOLDER = os.path.join(BASE_FOLDER, "images", "students")
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)
os.makedirs(STUDENT_FOLDER, exist_ok=True)



EMPLOYEE_ENCODE_FILE = os.path.join(BASE_FOLDER, "EmployeeEncodeFile.p")
STUDENT_ENCODE_FILE = os.path.join(BASE_FOLDER, "StudentEncodeFile.p")

# Ensure Encoding Files Exist
for file in [EMPLOYEE_ENCODE_FILE, STUDENT_ENCODE_FILE]:
    if not os.path.exists(file):
        pickle.dump([[], []], open(file, "wb"))


# Load Encodings
def load_encodings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return [], []


employeeEncodeListKnown, employeeIDs = load_encodings(EMPLOYEE_ENCODE_FILE)
studentEncodeListKnown, studentIDs = load_encodings(STUDENT_ENCODE_FILE)

# Save Encodings Function
def save_encodings(file_path, encode_list, ids):
    with open(file_path, 'wb') as file:
        pickle.dump([encode_list, ids], file)



#============================== For laptop Internal camera  part using======= ================================#

# Camera Handling
camera = None
camera_lock = threading.Lock()


def initialize_camera():
    """Ensure the camera is initialized before capturing frames."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(1)
            if not camera.isOpened():
                raise RuntimeError("⚠️ Camera initialization failed!")


def close_camera():
    """Release the camera if it's open."""
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None


@app.route('/video_feed')
def video_feed():
    """Start video streaming."""
    initialize_camera()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera when navigating to another page."""
    close_camera()
    return jsonify({"message": "Camera stopped"}), 200


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue  

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


#============================== External camra  ================================#

# # Camera Handling
# camera = None
# camera_lock = threading.Lock()


# def initialize_camera():
#     """Ensure the camera is initialized before capturing frames from the USB camera."""
#     global camera
#     with camera_lock:
#         if camera is None or not camera.isOpened():
#             camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # ✅ USB Camera (Index 1)
#             time.sleep(1)
#             if not camera.isOpened():
#                 raise RuntimeError("⚠️ USB Camera initialization failed!")


# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             print("❌ Frame capture failed, retrying...")
#             continue  

#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("❌ Encoding frame failed, retrying...")
#             continue  

#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# def close_camera():
#     """Release the USB camera if it's open."""
#     global camera
#     with camera_lock:
#         if camera and camera.isOpened():
#             camera.release()
#             camera = None


# @app.route('/video_feed')
# def video_feed():
#     """Start video streaming."""
#     initialize_camera()
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/stop_camera', methods=['POST'])
# def stop_camera():
#     """Stop the camera when navigating to another page."""
#     close_camera()
#     return jsonify({"message": "Camera stopped"}), 200



# ========================================== Face Recognition Logic with Timer ============================================= #

@app.route('/match_auto', methods=['GET'])
def match_auto():
    """Detect faces and determine whether to enable IN/OUT buttons."""
    initialize_camera()
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    detected_user = None
    detected_type = None
    face_found = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.85:  # ✅ Improved accuracy
            continue

        face_found = True
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(face_rgb)

        if not encodes:
            return jsonify({"error": "Face detected but not recognized. Try adjusting lighting."}), 400

        for face_encode in encodes:
            # 🔹 Check against employee list
            matches_emp = face_recognition.compare_faces(employeeEncodeListKnown, face_encode, tolerance=0.45)
            face_dis_emp = face_recognition.face_distance(employeeEncodeListKnown, face_encode)

            if True in matches_emp:
                best_match_index = np.argmin(face_dis_emp)
                detected_user = employeeIDs[best_match_index]
                detected_type = "employee"
                break

            # 🔹 Check against student list if not found in employees
            matches_stu = face_recognition.compare_faces(studentEncodeListKnown, face_encode, tolerance=0.45)
            face_dis_stu = face_recognition.face_distance(studentEncodeListKnown, face_encode)

            if True in matches_stu:
                best_match_index = np.argmin(face_dis_stu)
                detected_user = studentIDs[best_match_index]
                detected_type = "student"
                break

    if not face_found:
        return jsonify({"error": "No face detected"}), 204

    if detected_user and detected_type:
        cursor = mysql.connection.cursor()

        if detected_type == "employee":
            cursor.execute("SELECT `IN`, `OUT` FROM employee_attendance WHERE user_id = %s ORDER BY `IN` DESC LIMIT 1", (detected_user,))
        else:
            cursor.execute("SELECT `IN`, `OUT` FROM student_attendance WHERE user_id = %s ORDER BY `IN` DESC LIMIT 1", (detected_user,))

        record = cursor.fetchone()
        cursor.close()

        if record:
            last_in = record[0]
            last_out = record[1]

            if last_in and not last_out:
                status = "IN"  # Last status was IN, so enable OUT button
            else:
                status = "OUT"  # Last status was OUT, enable IN button
        else:
            status = "OUT"  # First-time user, enable IN

        return jsonify({"message": "Face matched", "user_id": detected_user, "user_type": detected_type, "status": status}), 200
    else:
        return jsonify({"error": "Unknown Person"}), 404



# =================== Mark Attendance API Route with Email Notification ===================================== #
import threading

def send_email_async(subject, body, recipient):
    """Send email in a background thread with Flask application context."""
    with app.app_context():
        try:
            msg = Message(subject, sender=app.config['MAIL_DEFAULT_SENDER'], recipients=[recipient])
            msg.body = body
            mail.send(msg)
            print(f"📧 Email sent to {recipient}")
        except Exception as e:
            print(f"❌ Email sending failed: {str(e)}")


@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """Mark IN/OUT attendance and send email asynchronously."""
    data = request.json
    user_id = data.get('user_id')
    user_type = data.get('user_type')
    status = data.get('status')

    if not user_id or not user_type or status not in ["IN", "OUT"]:
        return jsonify({"error": "Invalid request"}), 400

    cursor = mysql.connection.cursor()

    if user_type == "employee":
        cursor.execute("SELECT first_name, last_name, email FROM employees WHERE user_id = %s", (user_id,))
        table_name = "employee_attendance"
    else:
        cursor.execute("SELECT first_name, last_name, email FROM students WHERE user_id = %s", (user_id,))
        table_name = "student_attendance"

    person = cursor.fetchone()

    if not person:
        return jsonify({"error": f"{user_type.capitalize()} not found"}), 404

    person_name = f"{person[0]} {person[1]}"
    email = person[2]  
    current_time = datetime.now()

    cursor.execute(f"SELECT id, `IN`, `OUT` FROM {table_name} WHERE user_id = %s ORDER BY `IN` DESC LIMIT 1", (user_id,))
    record = cursor.fetchone()

    if status == "IN":
        cursor.execute(
            f"INSERT INTO {table_name} (name, user_id, `IN`, `Status`) VALUES (%s, %s, %s, %s)",
            (person_name, user_id, current_time, "Present")
        )
        response_message = "✅ You are in office!"
        email_subject = "Attendance Marked: IN ✅"
        email_body = f"Hello {person_name},\n\nYou have successfully marked IN at {current_time.strftime('%Y-%m-%d %H:%M:%S')}.\n\nRegards,\nNetparam Technologies Pvt. Ltd."
    
    elif status == "OUT":
        if not record or not record[1]:  
            return jsonify({"error": "Cannot mark OUT without a matching IN"}), 400

        if record[2]:  
            return jsonify({"error": "OUT already marked for the latest IN"}), 400

        cursor.execute(
            f"UPDATE {table_name} SET `OUT` = %s WHERE id = %s",
            (current_time, record[0])
        )
        response_message = "🚪 You are out of office!"
        email_subject = "Attendance Marked: OUT 🚪"
        email_body = f"Hello {person_name},\n\nYou have successfully marked OUT at {current_time.strftime('%Y-%m-%d %H:%M:%S')}.\n\nRegards,\nNetparam Technologies Pvt. Ltd."

    mysql.connection.commit()
    cursor.close()

    # ✅ Email thread now runs inside `app.app_context()`
    email_thread = threading.Thread(target=send_email_async, args=(email_subject, email_body, email))
    email_thread.start()

    return jsonify({"message": response_message, "status": status, "user_id": user_id}), 200


#====================api/monthly_attendance=====================================

@app.route('/api/monthly_attendance', methods=['GET'])
def get_monthly_attendance():
    """Fetch attendance details based on month and specific date filters."""
    try:
        month_year = request.args.get('month_year', datetime.now().strftime('%Y-%m'))
        employee_name = request.args.get('employee_name', '').strip()
        specific_date = request.args.get('specific_date', None)

        cursor = mysql.connection.cursor()

        if employee_name:
            # ✅ Fetch detailed records for a single employee
            query = """
                SELECT user_id, name, DATE(`IN`) AS date, TIME(`IN`), TIME(`OUT`)
                FROM employee_attendance
                WHERE DATE_FORMAT(`IN`, '%%Y-%%m') = %s
                AND LOWER(name) LIKE %s
            """
            params = [month_year, f"%{employee_name.lower()}%"]

            if specific_date:
                query += " AND DATE(`IN`) = %s"
                params.append(specific_date)

            query += " ORDER BY DATE(`IN`), TIME(`IN`);"

        else:
            # ✅ Fetch summary for all employees
            query = """
                SELECT user_id, name, COUNT(DISTINCT DATE(`IN`)) AS total_days_present,
                       SEC_TO_TIME(SUM(IF(`OUT` IS NOT NULL, TIME_TO_SEC(TIMEDIFF(`OUT`, `IN`)), 0))) AS total_hours
                FROM employee_attendance
                WHERE DATE_FORMAT(`IN`, '%%Y-%%m') = %s
            """
            params = [month_year]

            if specific_date:
                query += " AND DATE(`IN`) = %s"
                params.append(specific_date)

            query += " GROUP BY user_id, name;"

        cursor.execute(query, tuple(params))
        records = cursor.fetchall()
        cursor.close()

        if not records:
            return jsonify({"data": []}), 200  # No data found

        if employee_name:
            # ✅ Process data for a single employee
            employee_data = {
                "name": employee_name,
                "total_days": set(),
                "total_hours": 0,
                "daily_records": {}
            }

            total_seconds = 0

            for row in records:
                user_id, name, date, in_time, out_time = row
                date_str = date.strftime('%Y-%m-%d')

                employee_data["total_days"].add(date_str)

                if date_str not in employee_data["daily_records"]:
                    employee_data["daily_records"][date_str] = {
                        "in_times": [],
                        "out_times": [],
                        "total_day_hours": "0 hr 0 min",
                        "time_durations": []
                    }

                if in_time:
                    employee_data["daily_records"][date_str]["in_times"].append(str(in_time))
                if out_time:
                    employee_data["daily_records"][date_str]["out_times"].append(str(out_time))

                # ✅ Calculate total working hours dynamically
                if in_time and out_time:
                    in_time_dt = datetime.strptime(str(in_time), "%H:%M:%S")
                    out_time_dt = datetime.strptime(str(out_time), "%H:%M:%S")
                    time_diff_seconds = (out_time_dt - in_time_dt).total_seconds()
                    employee_data["daily_records"][date_str]["time_durations"].append(time_diff_seconds)
                    total_seconds += time_diff_seconds

            # ✅ Finalize total hours and daily records
            employee_data["total_days"] = len(employee_data["total_days"])
            employee_data["total_hours"] = format_time(total_seconds)

            for date_str, record in employee_data["daily_records"].items():
                total_daily_seconds = sum(record["time_durations"])
                record["total_day_hours"] = format_time(total_daily_seconds) if total_daily_seconds > 0 else "0 hr 0 min"

            return jsonify({"data": [employee_data]}), 200
        else:
            # ✅ Process summary for all employees
            attendance_summary = []
            for row in records:
                user_id, name, total_days, total_hours = row
                attendance_summary.append({
                    "user_id": user_id,
                    "name": name,
                    "total_days": total_days,
                    "total_hours": format_time(total_hours.total_seconds()) if total_hours else "0 hr 0 min"
                })

            return jsonify({"data": attendance_summary}), 200

    except Exception as e:
        print("🚨 ERROR:", traceback.format_exc())  
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500




def format_time(total_seconds):
    """Convert seconds to 'H hr M min' format."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours} hr {minutes} min"


# ✅ Helper Functions for Time Formatting
def sum_total_hours(daily_records):
    """Sum up total working hours across all days."""
    total_seconds = 0
    for record in daily_records.values():
        for duration in record["time_durations"]:
            total_seconds += parse_time(duration)
    return format_time(total_seconds)


def parse_time(time_str):
    """Convert 'H hr M min' format into seconds."""
    parts = time_str.split(" ")
    hours = int(parts[0]) if "hr" in parts else 0
    minutes = int(parts[2]) if "min" in parts else 0
    return (hours * 3600) + (minutes * 60)


# ✅ Convert MySQL Time Format to Total Seconds
def convert_timedelta_to_seconds(mysql_time):
    if mysql_time is None:
        return 0
    h, m, s = map(int, str(mysql_time).split(':'))
    return h * 3600 + m * 60 + s




# =================== ✅ FIXED EMPLOYEE FETCHING ✅ =================== #

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Fetch employee data with search query."""
    query = request.args.get('query', '')

    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT user_id, first_name, last_name, image_path 
        FROM employees
        WHERE first_name LIKE %s OR last_name LIKE %s OR user_id LIKE %s
    """, (f"%{query}%", f"%{query}%", f"%{query}%"))

    data = cursor.fetchall()
    cursor.close()

    return jsonify({"data": [{"user_id": row[0], "name": f"{row[1]} {row[2]}", "image_path": row[3]} for row in data]})


# =================== Fetch Daily Attendance =================== #

@app.route('/api/daily_attendance', methods=['GET'])
def get_daily_attendance():
    """Fetch daily attendance details dynamically."""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT user_id, name, `IN`, `OUT`, TIMEDIFF(`OUT`, `IN`)
        FROM employee_attendance
        WHERE DATE(`IN`) = %s
    """, (date,))

    records = cursor.fetchall()
    cursor.close()

    if not records:
        return jsonify({"data": [], "message": "No Data Available"})

    attendance_summary = []

    for row in records:
        user_id, name, in_time, out_time, time_diff = row
        if out_time:
            total_seconds = time_diff.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            total_hours = f"{hours} hr {minutes} min"
        else:
            total_hours = "Still IN"

        attendance_summary.append({
            "user_id": user_id,
            "name": name,
            "date": date,
            "total_hours": total_hours
        })

    return jsonify({"data": attendance_summary})


#==========================================register================================================#

@app.route('/register', methods=['POST'])
def register():
    """Register a new employee or student using either a captured or uploaded photo."""
    data = request.form  
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    user_type = data.get('user_type')  # Employee or Student
    email = data.get('email')
    user_id = f"{user_type}_{first_name.lower()}_{last_name.lower()}"

    # Validate input
    if not first_name or not last_name or not user_type or not email:
        return jsonify({"error": "All fields are required"}), 400

    if user_type not in ["employee", "student"]:
        return jsonify({"error": "Invalid user type"}), 400

    # Check if an image file was uploaded
    if 'photo' in request.files and request.files['photo'].filename != '':
        file = request.files['photo']
        file_path = os.path.join(EMPLOYEE_FOLDER, f"{user_id}.jpg")
        file.save(file_path)
    else:
        initialize_camera()
        success, frame = camera.read()
        if not success:
            return jsonify({"error": "Failed to capture image."}), 500
        file_path = os.path.join(EMPLOYEE_FOLDER, f"{user_id}.jpg")
        cv2.imwrite(file_path, frame)

    # Convert image to RGB and check face encoding
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodes = face_recognition.face_encodings(img)

    if not encodes:
        os.remove(file_path)
        return jsonify({"error": "No face detected. Try again with better lighting."}), 400

    # Store face encoding
    encode_list, id_list = (employeeEncodeListKnown, employeeIDs) if user_type == "employee" else (studentEncodeListKnown, studentIDs)
    encode_list.append(encodes[0])
    id_list.append(user_id)
    save_encodings(EMPLOYEE_ENCODE_FILE if user_type == "employee" else STUDENT_ENCODE_FILE, encode_list, id_list)

    try:
        cursor = mysql.connection.cursor()
        if user_type == "employee":
            cursor.execute(
                "INSERT INTO employees (first_name, last_name, user_id, email, image_path) VALUES (%s, %s, %s, %s, %s)",
                (first_name, last_name, user_id, email, file_path)
            )
        else:
            cursor.execute(
                "INSERT INTO students (first_name, last_name, user_id, email, image_path) VALUES (%s, %s, %s, %s, %s)",
                (first_name, last_name, user_id, email, file_path)
            )
        mysql.connection.commit()
        cursor.close()
    except Exception as e:
        return jsonify({"error": "Failed to save registration data"}), 500

    return jsonify({"message": f"Registration successful for {user_id}"}), 200




#===================================captuer photo ============================================#

@app.route('/capture_photo', methods=['GET'])
def capture_photo():
    """Capture a photo using the camera."""
    try:
        initialize_camera()
        time.sleep(2)  # ✅ Allow the camera to adjust

        for attempt in range(5):  # ✅ Try capturing up to 5 times
            success, frame = camera.read()
            if success and frame is not None:
                break
            print(f"⚠️ Warning: Failed to capture photo (Attempt {attempt + 1}/5)")
            time.sleep(1)

        if not success or frame is None:
            return jsonify({"error": "Failed to capture photo. Try again."}), 500

        # Save the captured photo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"static/images/captured_{timestamp}.jpg"
        cv2.imwrite(file_path, frame)

        # Convert to RGB and check face encoding
        img = cv2.imread(file_path)
        if img is None:
            os.remove(file_path)
            return jsonify({"error": "Failed to read the image. Try again."}), 500

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if not encodes:  # ✅ If no face detected, delete the image
            os.remove(file_path)
            return jsonify({"error": "No face detected. Please try again with better lighting."}), 400

        return jsonify({"image_path": file_path}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




#============================================================Admin Integration ========================================================================#

# Ensure reports directory exists
REPORTS_DIR = os.path.join(os.getcwd(), "static", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# =================== ✅ ROUTES TO RENDER PAGES ✅ =================== #
@app.route('/')
def index():
    """Render the home page."""    
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    """Render the admin panel."""    
    return render_template('admin.html')


@app.route('/registration')
def registration():
    """Render the registration page."""
    return render_template('registration.html')


# =================== ✅ FIXED ADMIN LOGIN ✅ =================== #
@app.route('/admin_login', methods=['POST'])
def admin_login():
    """Admin Login."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username == 'admin' and password == 'admin123':
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/logout', methods=['GET'])
def logout():
    """Logout and redirect to home page."""
    return jsonify({"message": "Logged out successfully", "redirect": "/"})

# ======================= METRICS =======================
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Fetch total employees & students count."""
    cursor = mysql.connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM employees")
    total_employees = cursor.fetchone()[0]

    return jsonify({"totalEmployees": total_employees}), 200


# ======================= EXPORT DAILY ATTENDANCE TO CSV =======================

@app.route('/export_daily_csv', methods=['GET'])
def export_daily_csv():
    """Download daily attendance as a CSV file."""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT e.user_id, e.first_name, e.last_name, a.date, a.total_hours
        FROM attendance_summary a
        JOIN employees e ON e.user_id = a.user_id
        WHERE a.date = %s
    """, (date,))

    data = cursor.fetchall()
    cursor.close()

    file_path = f"static/reports/daily_attendance_{date}.csv"
    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "First Name", "Last Name", "Date", "Total Hours"])
        writer.writerows(data)

    return send_file(file_path, as_attachment=True)


# =========================== 📌 FIXED ATTENDANCE EXPORT API 📌 ===========================

@app.route('/export_attendance', methods=['GET'])
def export_attendance():
    """Export attendance based on filters (Month, Employee, Date)"""
    
    month_year = request.args.get('month_year', None)
    employee_name = request.args.get('employee_name', None)
    selected_date = request.args.get('selected_date', None)

    cursor = mysql.connection.cursor()

    if employee_name:
        # ✅ Fetch **Daily Records** for a Specific Employee
        query = """
            SELECT e.user_id, e.first_name, e.last_name, DATE(a.`IN`) AS date, 
                   SEC_TO_TIME(SUM(TIME_TO_SEC(TIMEDIFF(a.`OUT`, a.`IN`)))) AS total_hours
            FROM employee_attendance a
            JOIN employees e ON e.user_id = a.user_id
            WHERE CONCAT(e.first_name, ' ', e.last_name) LIKE %s
        """
        params = [f"%{employee_name}%"]

        if month_year:
            query += " AND DATE_FORMAT(a.`IN`, '%%Y-%%m') = %s"
            params.append(month_year)

        if selected_date:
            query += " AND DATE(a.`IN`) = %s"
            params.append(selected_date)

        query += " GROUP BY e.user_id, DATE(a.`IN`) ORDER BY e.user_id, date"

        cursor.execute(query, params)
        records = cursor.fetchall()
        cursor.close()

        if not records:
            return jsonify({"error": "No attendance data found"}), 404

        # ✅ Convert to DataFrame for export
        df = pd.DataFrame(records, columns=["User ID", "First Name", "Last Name", "Date", "Total Hours"])
    else:
        # ✅ Fetch **Summary** for All Employees
        query = """
            SELECT e.user_id, e.first_name, e.last_name,
                   COUNT(DISTINCT DATE(a.`IN`)) AS total_days_present,
                   SEC_TO_TIME(SUM(TIME_TO_SEC(TIMEDIFF(a.`OUT`, a.`IN`)))) AS total_hours
            FROM employee_attendance a
            JOIN employees e ON e.user_id = a.user_id
            WHERE 1=1
        """
        params = []
        
        if month_year:
            query += " AND DATE_FORMAT(a.`IN`, '%%Y-%%m') = %s"
            params.append(month_year)

        if selected_date:
            query += " AND DATE(a.`IN`) = %s"
            params.append(selected_date)

        query += " GROUP BY e.user_id ORDER BY e.user_id"

        cursor.execute(query, params)
        records = cursor.fetchall()
        cursor.close()

        if not records:
            return jsonify({"error": "No attendance data found"}), 404

        # ✅ Convert to DataFrame for export
        df = pd.DataFrame(records, columns=["User ID", "First Name", "Last Name", "Total Days Present", "Total Hours"])

    # ✅ Convert timedelta format to 'H hr M min'
    # df["Total Hours"] = df["Total Hours"].astype(str).apply(lambda x: f"{int(x.split(':')[0])} hr {int(x.split(':')[1])} min" if x != "None" else "0 hr 0 min")
    df["Total Hours"] = df["Total Hours"].astype(str).apply(lambda x: 
    f"{int(x.split(' ')[-1].split(':')[0])} hr {int(x.split(' ')[-1].split(':')[1])} min" 
    if x != "None" else "0 hr 0 min")


    # ✅ Generate Filename Dynamically
    file_name = "attendance_report.xlsx"
    if month_year and not employee_name and not selected_date:
        file_name = f"attendance_{month_year}.xlsx"
    elif employee_name and not selected_date:
        file_name = f"attendance_{employee_name.replace(' ', '_')}.xlsx"
    elif selected_date:
        file_name = f"attendance_{selected_date}.xlsx"

    file_path = os.path.join(REPORTS_DIR, file_name)
    df.to_excel(file_path, index=False)

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
