from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from SVR import PIPELINE

Pipe = PIPELINE()

# ================= FastAPI App =================
app = FastAPI(title="Voiture d'Occasion API", docs_url="/api/docs")

# Session middleware
app.add_middleware(SessionMiddleware, secret_key="super_secret_key_change_this")

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ================= Auth helpers =================
def login_required(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/login")
    return True

# ================= Routes HTML =================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    if user and check_password_hash(user[1], password):
        request.session["user_id"] = user[0]
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Login invalide"})

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

@app.get("/Sign_Up", response_class=HTMLResponse)
def signup_get(request: Request):
    return templates.TemplateResponse("Sign_Up.html", {"request": request})

@app.post("/Sign_Up", response_class=HTMLResponse)
def signup_post(request: Request,
                username: str = Form(...),
                email: str = Form(...),
                password: str = Form(...)):
    hashed_password = generate_password_hash(password)
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, hashed_password))
        conn.commit()
    except:
        conn.close()
        return templates.TemplateResponse("Sign_Up.html", {"request": request, "error": "Utilisateur déjà existant"})
    conn.close()
    return RedirectResponse(url="/login", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if not login_required(request):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("DATAVIZ.html", {"request": request})

# ================= Estimate =================
@app.get("/estimate", response_class=HTMLResponse)
def estimate_get(request: Request):
    return templates.TemplateResponse("estimate.html", {"request": request})

@app.post("/estimate", response_class=HTMLResponse)
def estimate_post(request: Request,
                  Name: str = Form(...),
                  Location: str = Form(...),
                  Year: int = Form(...),
                  Kilometers_Driven: int = Form(...),
                  Fuel_Type: str = Form(...),
                  Transmission: str = Form(...),
                  Owner_Type: str = Form(...),
                  Mileage: str = Form(...),
                  Engine: str = Form(...),
                  Power: str = Form(...),
                  Seats: int = Form(...)):
    data = {
        "Name": Name,
        "Location": Location,
        "Year": Year,
        "Kilometers_Driven": Kilometers_Driven,
        "Fuel_Type": Fuel_Type,
        "Transmission": Transmission,
        "Owner_Type": Owner_Type,
        "Mileage": Mileage,
        "Engine": Engine,
        "Power": Power,
        "Seats": Seats
    }
    df_input = pd.DataFrame([data])
    prediction = Pipe.predict(df_input)
    price = int(prediction[0])
    return templates.TemplateResponse("estimate.html", {"request": request, "prediction": price})

# ================= API JSON =================
@app.get("/api/hello")
def hello():
    return {"message": "Bonjour"}

# Swagger UI → accessible sur /api/docs automatiquement

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)