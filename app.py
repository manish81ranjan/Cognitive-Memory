from flask import Flask, render_template, request, redirect, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "demnet_secret_key"

# ---------- MySQL Config ----------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Man@6jan'
app.config['MYSQL_DB'] = 'demnet_db'

mysql = MySQL(app)

# ---------- Pages ----------
@app.route("/")
def index():
    if 'user_id' in session:
        return render_template("index.html")
    return redirect("/signup")

@app.route("/signup")
def signup_page():
    return render_template("profile.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/signup")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["POST"])
def signup():
    name = request.form["name"]
    email = request.form["email"]
    password = generate_password_hash(request.form["password"])

    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
        (name, email, password)
    )
    mysql.connection.commit()
    cur.close()

    return redirect("/signup")

# ---------- LOGIN ----------
from werkzeug.security import check_password_hash

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if user and check_password_hash(user[3], password):
        session["user_id"] = user[0]
        session["user_name"] = user[1]
        return redirect("/")
    else:
        flash("Invalid email or password", "error")
        return redirect("/signup")


if __name__ == "__main__":
    app.run(debug=True)
