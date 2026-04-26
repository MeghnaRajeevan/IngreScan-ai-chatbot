import asyncio
import logging
import requests
import cv2
import numpy as np
import zxingcpp
import google.generativeai as genai
from groq import Groq
import io
import os
import sys
import sqlite3
import re
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest

load_dotenv(override=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
BOT_TOKEN      = os.getenv("BOT_TOKEN")

async def safe_edit(q, text, **kwargs):
    try:
        await q.edit_message_text(text, **kwargs)
    except Exception:
        await q.message.reply_text(text, **kwargs)

_log_cache: dict = {}
_compare_cache: dict = {}
_ai_report_cache: dict = {}

ASK_DIET, ASK_ALLERGY, ASK_CONDITION, ASK_LANGUAGE, COMPARE_WAIT = range(5)

DB_PATH = "ingrebot.db"

gemini_model = None
groq_client  = None

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing from .env")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini AI configured")
except Exception as e:
    logger.error(f"Gemini setup failed: {e}")

try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq AI configured")
except Exception as e:
    logger.error(f"Groq setup failed: {e}")


# --- HYBRID RATE LIMIT SAFEGUARD ---
async def generate_with_retry(prompt, image=None, max_retries=3):
    """Try Gemini first. If quota exceeded fall back to Groq automatically."""
    
    # If image is provided always use Gemini Vision — Groq does not support images easily here
    if image:
        if not gemini_model:
            raise Exception("Gemini model not initialized")
        for attempt in range(max_retries):
            try:
                return await gemini_model.generate_content_async([prompt, image])
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    match = re.search(r"retry in (\d+\.\d+)s", error_msg)
                    wait_time = float(match.group(1)) + 2.0 if match else 35.0
                    logger.warning(f"⏳ Gemini Vision hit limit. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

    # For text prompts try Gemini first then fall back to Groq
    if gemini_model:
        for attempt in range(max_retries):
            try:
                return await gemini_model.generate_content_async(prompt)
            except ResourceExhausted:
                if groq_client:
                    logger.warning("⚡ Gemini quota hit — switching to Groq automatically")
                    break
                if attempt < max_retries - 1:
                    await asyncio.sleep(35.0)
                else:
                    raise

    # Groq fallback for text prompts
    if groq_client:
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.7,
            )
            # Return object that matches Gemini response interface
            class GroqResponse:
                def __init__(self, text):
                    self.text = text
            return GroqResponse(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise
            
    raise Exception("No AI model available")


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id   INTEGER PRIMARY KEY,
            name      TEXT    DEFAULT '',
            allergies TEXT    DEFAULT 'None',
            diet_pref TEXT    DEFAULT 'Normal',
            condition TEXT    DEFAULT 'None',
            language  TEXT    DEFAULT 'English',
            onboarded INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS scan_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            barcode    TEXT,
            name       TEXT,
            calories   REAL DEFAULT 0,
            carbs      REAL DEFAULT 0,
            fat        REAL DEFAULT 0,
            protein    REAL DEFAULT 0,
            nutriscore TEXT DEFAULT '',
            scanned_at TEXT DEFAULT (datetime('now','localtime')),
            health_score REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS daily_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER,
            barcode   TEXT,
            name      TEXT,
            grams     REAL DEFAULT 100,
            calories  REAL DEFAULT 0,
            carbs     REAL DEFAULT 0,
            fat       REAL DEFAULT 0,
            protein   REAL DEFAULT 0,
            logged_at TEXT DEFAULT (datetime('now','localtime'))
        );
    """)
    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(user_profiles)")}
    migrations = [
        ("name",        "TEXT    DEFAULT ''"),
        ("condition",   "TEXT    DEFAULT 'None'"),
        ("onboarded",   "INTEGER DEFAULT 0"),
        ("health_score","REAL    DEFAULT 0"),
    ]
    sh_cols = {row[1] for row in con.execute("PRAGMA table_info(scan_history)")}
    if "health_score" not in sh_cols:
        con.execute("ALTER TABLE scan_history ADD COLUMN health_score REAL DEFAULT 0")
    for col, typedef in migrations:
        if col not in existing_cols:
            con.execute(f"ALTER TABLE user_profiles ADD COLUMN {col} {typedef}")
    con.commit()
    con.close()
    print("✅ Database ready")

def get_profile(user_id: int) -> dict:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT name,allergies,diet_pref,condition,language,onboarded "
        "FROM user_profiles WHERE user_id=?", (user_id,)
    ).fetchone()
    con.close()
    if row:
        return {"name":row[0],"allergies":row[1],"diet_pref":row[2],
                "condition":row[3],"language":row[4],"onboarded":row[5]}
    return {"name":"","allergies":"None","diet_pref":"Normal",
            "condition":"None","language":"English","onboarded":0}

def upsert_profile(user_id: int, **kw):
    con = sqlite3.connect(DB_PATH)
    if con.execute("SELECT 1 FROM user_profiles WHERE user_id=?", (user_id,)).fetchone():
        sets = ", ".join(f"{k}=?" for k in kw)
        con.execute(f"UPDATE user_profiles SET {sets} WHERE user_id=?", (*kw.values(), user_id))
    else:
        con.execute(
            "INSERT INTO user_profiles (user_id,name,allergies,diet_pref,condition,language,onboarded) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, kw.get("name",""), kw.get("allergies","None"),
             kw.get("diet_pref","Normal"), kw.get("condition","None"),
             kw.get("language","English"), kw.get("onboarded",0))
        )
    con.commit()
    con.close()

def get_history(user_id, limit=8):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT name,calories,scanned_at FROM scan_history "
        "WHERE user_id=? ORDER BY scanned_at DESC LIMIT ?", (user_id, limit)
    ).fetchall()
    con.close()
    return rows

def log_food(user_id, barcode, name, grams, cal100, carb100, fat100, prot100):
    f = grams / 100
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO daily_log (user_id,barcode,name,grams,calories,carbs,fat,protein) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (user_id, barcode, name, grams,
         round(cal100*f,1), round(carb100*f,1), round(fat100*f,1), round(prot100*f,1))
    )
    con.commit()
    con.close()

def get_today_log(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    con   = sqlite3.connect(DB_PATH)
    rows  = con.execute(
        "SELECT name,grams,calories,carbs,fat,protein FROM daily_log "
        "WHERE user_id=? AND logged_at LIKE ?", (user_id, f"{today}%")
    ).fetchall()
    con.close()
    return rows

def compute_product_score(info: dict) -> float:
    score   = 60.0
    sugar   = float(info.get("sugar",0)   or 0)
    fat     = float(info.get("fat",0)     or 0)
    salt    = float(info.get("salt",0)    or 0)
    fiber   = float(info.get("fiber",0)   or 0)
    protein = float(info.get("protein",0) or 0)
    kcal    = float(info.get("energy",0)  or 0)
    ns      = (info.get("nutriscore","")  or "").upper()

    if sugar  > 20:  score -= 20
    elif sugar > 10: score -= 10
    elif sugar > 5:  score -= 5
    if fat    > 20:  score -= 10
    elif fat  > 10:  score -= 5
    if salt   > 1.5: score -= 10
    elif salt > 0.8: score -= 5
    if kcal   > 500: score -= 8
    elif kcal > 300: score -= 4
    if fiber  > 6:   score += 10
    elif fiber > 3:  score += 5
    if protein > 15: score += 8
    elif protein > 8: score += 4

    ns_delta = {"A":15,"B":8,"C":0,"D":-10,"E":-20}.get(ns, 0)
    score += ns_delta
    return round(max(0, min(100, score)), 1)

def save_scan_with_score(user_id, barcode, name, cal, carbs, fat, protein, ns, health_score):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO scan_history (user_id,barcode,name,calories,carbs,fat,protein,nutriscore,health_score) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, barcode, name, cal, carbs, fat, protein, ns, health_score)
    )
    con.commit()
    con.close()

def get_personal_health_score(user_id: int) -> dict:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT health_score, scanned_at FROM scan_history "
        "WHERE user_id=? AND health_score > 0 ORDER BY scanned_at ASC",
        (user_id,)
    ).fetchall()
    con.close()

    if not rows:
        return {"score": None, "grade": "?", "total_scans": 0,
                "trend": "neutral", "weekly_avg": [], "message": "no_data"}

    alpha = 0.3
    ema   = rows[0][0]
    for score, _ in rows[1:]:
        ema = alpha * score + (1 - alpha) * ema

    personal_score = round(ema, 1)

    if   personal_score >= 80: grade, medal = "A", "🥇"
    elif personal_score >= 65: grade, medal = "B", "🥈"
    elif personal_score >= 50: grade, medal = "C", "🥉"
    elif personal_score >= 35: grade, medal = "D", "⚠️"
    else:                      grade, medal = "E", "🚨"

    from datetime import timedelta
    weekly_avgs = []
    now = datetime.now()
    for w in range(3, -1, -1):
        wstart = (now - timedelta(days=(w+1)*7)).strftime("%Y-%m-%d")
        wend   = (now - timedelta(days=w*7)).strftime("%Y-%m-%d")
        con2   = sqlite3.connect(DB_PATH)
        wrows  = con2.execute(
            "SELECT AVG(health_score) FROM scan_history "
            "WHERE user_id=? AND health_score>0 AND scanned_at>=? AND scanned_at<?",
            (user_id, wstart, wend)
        ).fetchone()
        con2.close()
        weekly_avgs.append(round(wrows[0] or 0, 1))

    trend = "neutral"
    if weekly_avgs[-1] > weekly_avgs[-2] + 3:  trend = "improving"
    elif weekly_avgs[-1] < weekly_avgs[-2] - 3: trend = "declining"

    return {
        "score":       personal_score,
        "grade":       grade,
        "medal":       medal,
        "trend":       trend,
        "total_scans": len(rows),
        "weekly_avgs": weekly_avgs,
        "best":        round(max(r[0] for r in rows), 1),
        "worst":       round(min(r[0] for r in rows), 1),
    }

def generate_health_score_chart(user_id: int, score_data: dict):
    try:
        import math
        weekly = score_data.get("weekly_avgs", [0,0,0,0])
        score  = score_data.get("score", 0)
        grade  = score_data.get("grade", "?")
        trend  = score_data.get("trend", "neutral")

        fig = plt.figure(figsize=(7, 4), facecolor='#0d1117')
        fig.patch.set_facecolor('#0d1117')

        ax1 = fig.add_axes([0.02, 0.1, 0.38, 0.8])
        ax1.set_facecolor('#0d1117')
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.axis('off')

        theta = [i/100*2*math.pi for i in range(101)]
        ax1.plot([0.85*math.cos(t) for t in theta], [0.85*math.sin(t) for t in theta], color='#21262d', linewidth=14, solid_capstyle='round')

        score_pct   = score / 100
        score_theta = [(-math.pi/2) + i/100*(2*math.pi*score_pct) for i in range(int(score_pct*100)+1)]
        color_map   = {"A":"#2ecc71","B":"#27ae60","C":"#f1c40f","D":"#e67e22","E":"#e74c3c"}
        arc_color   = color_map.get(grade, "#3498db")

        if score_theta:
            ax1.plot([0.85*math.cos(t) for t in score_theta], [0.85*math.sin(t) for t in score_theta], color=arc_color, linewidth=14, solid_capstyle='round')

        ax1.text(0, 0.12, f"{score:.0f}", ha='center', va='center', color='white', fontsize=30, fontweight='bold')
        ax1.text(0, -0.22, "/ 100", ha='center', va='center', color='#8b949e', fontsize=11)
        ax1.text(0, -0.55, f"Grade  {grade}", ha='center', va='center', color=arc_color, fontsize=14, fontweight='bold')

        trend_sym = {"improving":"↑ Improving","declining":"↓ Declining","neutral":"→ Stable"}
        trend_col = {"improving":"#2ecc71","declining":"#e74c3c","neutral":"#f1c40f"}
        ax1.text(0, -0.85, trend_sym.get(trend,"→ Stable"), ha='center', color=trend_col.get(trend,"#f1c40f"), fontsize=10, fontweight='bold')

        ax2 = fig.add_axes([0.44, 0.18, 0.54, 0.65])
        ax2.set_facecolor('#161b22')
        week_labels = ["3w ago","2w ago","Last wk","This wk"]
        valid = [w for w in weekly if w > 0]

        if len(valid) >= 2:
            xs = list(range(len(weekly)))
            ys = weekly
            ax2.fill_between(xs, ys, alpha=0.15, color=arc_color)
            ax2.plot(xs, ys, color=arc_color, linewidth=2.5, marker='o', markersize=6, markerfacecolor='white', markeredgecolor=arc_color)
            for x, y in zip(xs, ys):
                if y > 0:
                    ax2.text(x, y+2, f"{y:.0f}", ha='center', color='white', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Not enough data yet\nKeep scanning!", ha='center', va='center', color='#8b949e', fontsize=9, transform=ax2.transAxes)

        ax2.set_xticks(range(4))
        ax2.set_xticklabels(week_labels, fontsize=7, color='#8b949e')
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Score", color='#8b949e', fontsize=8)
        ax2.tick_params(colors='#8b949e', labelsize=7)
        ax2.spines[['top','right']].set_visible(False)
        ax2.spines[['left','bottom']].set_color('#21262d')
        ax2.yaxis.label.set_color('#8b949e')
        ax2.set_title("4-Week Trend", color='white', fontsize=9, pad=8)

        fig.text(0.5, 0.97, "🌿 IngreScanner — Health Score", ha='center', color='white', fontsize=12, fontweight='bold')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Health score chart error: {e}")
        plt.close('all')
        return None

async def gemini_score_commentary(score_data: dict, profile: dict, recent_products: list) -> str:
    recent_str = ", ".join(recent_products[:6]) if recent_products else "No scans yet"
    prompt = f"""
You are IngreScanner AI, a friendly personal nutritionist.
The user's Personal Health Score is {score_data.get('score',0):.0f}/100 (Grade {score_data.get('grade','?')}).
Trend: {score_data.get('trend','neutral')}.
Based on {score_data.get('total_scans',0)} product scans.
Their recently scanned products: {recent_str}
Their health condition: {profile.get('condition','None')}
Their diet: {profile.get('diet_pref','Normal')}

Write a SHORT, personal, conversational message (max 5 lines) that:
1. Reacts to their score warmly (not robotic)
2. Points out ONE specific eating pattern you notice from their recent products
3. Gives ONE simple actionable tip they can do TODAY
4. Ends with an encouraging line

Be like a real nutritionist friend texting them. No headers, no bullet points. Just natural text.
"""
    try:
        r = await generate_with_retry(prompt)
        return r.text.strip()
    except:
        return ""

def kb_main():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📸 Scan Food",        callback_data="menu_scan"),
         InlineKeyboardButton("🍽 Today's Log",      callback_data="menu_today")],
        [InlineKeyboardButton("📋 Scan History",     callback_data="menu_history"),
         InlineKeyboardButton("📊 Nutrition Chart",  callback_data="menu_report")],
        [InlineKeyboardButton("🆚 Compare Products", callback_data="menu_compare")],
        [InlineKeyboardButton("🏅 My Health Score",  callback_data="menu_healthscore")],
        [InlineKeyboardButton("👤 My Profile",       callback_data="menu_profile"),
         InlineKeyboardButton("⚙️ Settings",         callback_data="menu_settings")],
    ])

def kb_diet():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🍚 Normal",       callback_data="diet_Normal"), InlineKeyboardButton("🥑 Keto", callback_data="diet_Keto")],
        [InlineKeyboardButton("🌱 Vegan",        callback_data="diet_Vegan"), InlineKeyboardButton("🐟 Vegetarian", callback_data="diet_Vegetarian")],
        [InlineKeyboardButton("💪 High Protein", callback_data="diet_High Protein"), InlineKeyboardButton("🫀 Low Fat", callback_data="diet_Low Fat")],
    ])

def kb_allergy():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ No Allergies", callback_data="alg_None"), InlineKeyboardButton("🌾 Gluten", callback_data="alg_Gluten")],
        [InlineKeyboardButton("🥛 Dairy",        callback_data="alg_Dairy"), InlineKeyboardButton("🥜 Nuts", callback_data="alg_Nuts")],
        [InlineKeyboardButton("🥚 Eggs",         callback_data="alg_Eggs"), InlineKeyboardButton("🐟 Fish", callback_data="alg_Fish")],
        [InlineKeyboardButton("🫘 Soy",          callback_data="alg_Soy"), InlineKeyboardButton("🌿 Sesame", callback_data="alg_Sesame")],
    ])

def kb_condition():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ None",           callback_data="cnd_None"), InlineKeyboardButton("🩸 Diabetes", callback_data="cnd_Diabetes")],
        [InlineKeyboardButton("❤️ Blood Pressure", callback_data="cnd_Hypertension"), InlineKeyboardButton("⚖️ Obesity", callback_data="cnd_Obesity")],
        [InlineKeyboardButton("🫀 Heart Disease",  callback_data="cnd_Heart Disease"), InlineKeyboardButton("🦴 Osteoporosis", callback_data="cnd_Osteoporosis")],
    ])

def kb_language():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🇬🇧 English",   callback_data="lng_English"), InlineKeyboardButton("🇮🇳 Tamil", callback_data="lng_Tamil")],
        [InlineKeyboardButton("🇮🇳 Hindi",     callback_data="lng_Hindi"), InlineKeyboardButton("🇮🇳 Malayalam", callback_data="lng_Malayalam")],
        [InlineKeyboardButton("🇮🇳 Telugu",    callback_data="lng_Telugu")],
    ])

def kb_settings():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🥗 Change Diet",      callback_data="set_diet")],
        [InlineKeyboardButton("🌾 Change Allergy",   callback_data="set_allergy")],
        [InlineKeyboardButton("🏥 Change Condition", callback_data="set_condition")],
        [InlineKeyboardButton("🌐 Change Language",  callback_data="set_language")],
        [InlineKeyboardButton("🏠 Main Menu",        callback_data="menu_home")],
    ])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    profile = get_profile(user.id)
    if profile["onboarded"]:
        await update.message.reply_text(
            f"👋 Welcome back, *{profile['name'] or user.first_name}!*\n\n"
            f"📸 Send me a photo of any food barcode to analyse it,\n"
            f"or choose an option below 👇", parse_mode="Markdown", reply_markup=kb_main()
        )
        return ConversationHandler.END
    upsert_profile(user.id, name=user.first_name)
    await update.message.reply_text(
        f"👋 Hey *{user.first_name}!* Welcome to *IngreScanner* 🌿\n\n"
        f"I'll scan any food product and tell you *exactly what's in it* and whether it's safe for *you personally*.\n\n"
        f"Let me set up your profile in *3 quick taps* 👇\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n*Step 1 of 4 — Diet Preference*",
        parse_mode="Markdown", reply_markup=kb_diet()
    )
    return ASK_DIET

async def ob_diet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    context.user_data["diet"] = q.data.split("_",1)[1]
    await q.edit_message_text(f"✅ Diet: *{context.user_data['diet']}*\n\n━━━━━━━━━━━━━━━━━━━\n*Step 2 of 4 — Food Allergies*", parse_mode="Markdown", reply_markup=kb_allergy())
    return ASK_ALLERGY

async def ob_allergy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    context.user_data["allergy"] = q.data.split("_",1)[1]
    await q.edit_message_text(f"✅ Allergy: *{context.user_data['allergy']}*\n\n━━━━━━━━━━━━━━━━━━━\n*Step 3 of 4 — Health Condition*\n_(I'll warn you when a product is risky for your condition)_", parse_mode="Markdown", reply_markup=kb_condition())
    return ASK_CONDITION

async def ob_condition(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    context.user_data["condition"] = q.data.split("_",1)[1]
    await q.edit_message_text(f"✅ Condition: *{context.user_data['condition']}*\n\n━━━━━━━━━━━━━━━━━━━\n*Step 4 of 4 — Preferred Language*", parse_mode="Markdown", reply_markup=kb_language())
    return ASK_LANGUAGE

async def ob_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q    = update.callback_query
    await q.answer()
    lang = q.data.split("_",1)[1]
    uid  = q.from_user.id
    ud   = context.user_data
    upsert_profile(uid, diet_pref=ud.get("diet","Normal"), allergies=ud.get("allergy","None"), condition=ud.get("condition","None"), language=lang, onboarded=1)
    await q.edit_message_text(
        f"🎉 *All set! Here's your profile:*\n\n🥗 Diet: *{ud.get('diet','Normal')}*\n🌾 Allergy: *{ud.get('allergy','None')}*\n🏥 Condition: *{ud.get('condition','None')}*\n🌐 Language: *{lang}*\n\n"
        f"📸 *Send me a photo of any food barcode to get started!*", parse_mode="Markdown", reply_markup=kb_main()
    )
    return ConversationHandler.END

async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q   = update.callback_query
    await q.answer()
    uid = q.from_user.id
    d   = q.data

    if d == "menu_home":
        await safe_edit(q, "🏠 *Main Menu* — what would you like to do?", parse_mode="Markdown", reply_markup=kb_main())

    elif d == "menu_scan":
        await safe_edit(q, "📸 *Ready to scan!*\n\nJust send me a *photo of any food product or barcode*.\nI'll detect the barcode automatically and analyse it! 🔍", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

    elif d == "menu_today":
        rows = get_today_log(uid)
        if not rows:
            await safe_edit(q, "🍽 *Today's Log is empty.*\n\nScan a product and tap ➕ Log to Diary to add it here.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📸 Scan Now", callback_data="menu_scan")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
            return
        total_cal  = sum(r[2] for r in rows)
        total_carb = sum(r[3] for r in rows)
        total_fat  = sum(r[4] for r in rows)
        total_prot = sum(r[5] for r in rows)
        pct = min(100, int(total_cal / 2000 * 100))
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        lines = ["🍽 *Today's Food Log*\n"]
        for name, g, cal, *_ in rows: lines.append(f"• {name[:28]} — {g}g → {cal:.0f} kcal")
        lines += [f"\n━━━ TOTALS ━━━", f"🔥 {total_cal:.0f} kcal  |  🌾 {total_carb:.1f}g  |  🥑 {total_fat:.1f}g  |  💪 {total_prot:.1f}g", f"\n📊 Daily Goal: [{bar}] {pct}%  (target 2000 kcal)"]
        await safe_edit(q, "\n".join(lines), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 View Chart", callback_data="menu_report")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

    elif d == "menu_history":
        rows = get_history(uid)
        if not rows:
            await safe_edit(q, "📋 *No scans yet.*\n\nSend a food barcode photo to get started!", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📸 Scan Now", callback_data="menu_scan")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
            return
        lines = ["📋 *Your Recent Scans*\n"]
        for name, cal, ts in rows: lines.append(f"• {name[:28]} — {cal:.0f} kcal  _{ts[:10]}_")
        await safe_edit(q, "\n".join(lines), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

    elif d == "menu_report":
        chart = generate_daily_chart(uid)
        if not chart:
            await safe_edit(q, "📊 *No data yet for today.*\n\nLog food after scanning to see your nutrition chart.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🍽 Today's Log", callback_data="menu_today")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
            return
        await q.message.reply_photo(photo=chart, caption="📊 *Today's Nutrition vs Daily Targets*", parse_mode="Markdown", reply_markup=kb_main())

    elif d == "menu_profile":
        p = get_profile(uid)
        await safe_edit(q, f"👤 *Your Profile*\n\n🥗 Diet: *{p['diet_pref']}*\n🌾 Allergy: *{p['allergies']}*\n🏥 Condition: *{p['condition']}*\n🌐 Language: *{p['language']}*", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⚙️ Edit Settings", callback_data="menu_settings")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

    elif d == "menu_healthscore":
        await safe_edit(q, "⏳ Calculating your Personal Health Score…", parse_mode="Markdown")
        score_data = get_personal_health_score(uid)

        if score_data.get("message") == "no_data":
            await safe_edit(q, "🏅 *Personal Health Score*\n\nYou haven't scanned enough products yet.\nScan at least 3 food products to get your score!\n\n_Your score is calculated from all the products you scan — the healthier you eat, the higher it goes._", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📸 Scan Now", callback_data="menu_scan")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
            return

        con_tmp = sqlite3.connect(DB_PATH)
        recent_names = [r[0] for r in con_tmp.execute("SELECT name FROM scan_history WHERE user_id=? ORDER BY scanned_at DESC LIMIT 6", (uid,)).fetchall()]
        con_tmp.close()

        profile    = get_profile(uid)
        commentary = await gemini_score_commentary(score_data, profile, recent_names)
        chart      = generate_health_score_chart(uid, score_data)

        trend_sym = {"improving":"📈 Improving","declining":"📉 Declining","neutral":"📊 Stable"}
        score_msg = (f"🏅 *Your Personal Health Score*\n\n*{score_data['score']:.0f} / 100* —  Grade *{score_data['grade']}* {score_data['medal']}\n{trend_sym.get(score_data['trend'],'📊 Stable')}\n\n📦 Total products scanned: *{score_data['total_scans']}*\n🔝 Best product score: *{score_data['best']}*\n📉 Worst product score: *{score_data['worst']}*")
        if commentary: score_msg += f"\n\n💬 *AI Insight*\n{commentary}"

        if chart: await q.message.reply_photo(photo=chart, caption=score_msg, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📸 Scan to Improve", callback_data="menu_scan")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
        else: await safe_edit(q, score_msg, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📸 Scan to Improve", callback_data="menu_scan")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

    elif d == "menu_compare":
        uid = q.from_user.id
        _compare_cache.pop(uid, None)
        await safe_edit(q, "🆚 *Compare Two Products*\n\n📸 Send me a photo of the *first product* to start the comparison.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="menu_home")]]))
        _compare_cache[uid] = {"step": "A"}

    elif d == "menu_settings":
        await safe_edit(q, "⚙️ *Settings* — tap what you want to change:", parse_mode="Markdown", reply_markup=kb_settings())

async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q   = update.callback_query
    await q.answer()
    uid = q.from_user.id
    d   = q.data
    if   d == "set_diet":      await safe_edit(q, "🥗 *Choose your diet:*",      parse_mode="Markdown", reply_markup=kb_diet())
    elif d == "set_allergy":   await safe_edit(q, "🌾 *Choose your allergy:*",   parse_mode="Markdown", reply_markup=kb_allergy())
    elif d == "set_condition": await safe_edit(q, "🏥 *Choose your condition:*", parse_mode="Markdown", reply_markup=kb_condition())
    elif d == "set_language":  await safe_edit(q, "🌐 *Choose your language:*",  parse_mode="Markdown", reply_markup=kb_language())
    elif d.startswith("diet_"): val = d.split("_",1)[1]; upsert_profile(uid, diet_pref=val); await safe_edit(q, f"✅ Diet updated to *{val}*", parse_mode="Markdown", reply_markup=kb_settings())
    elif d.startswith("alg_"): val = d.split("_",1)[1]; upsert_profile(uid, allergies=val); await safe_edit(q, f"✅ Allergy updated to *{val}*", parse_mode="Markdown", reply_markup=kb_settings())
    elif d.startswith("cnd_"): val = d.split("_",1)[1]; upsert_profile(uid, condition=val); await safe_edit(q, f"✅ Condition updated to *{val}*", parse_mode="Markdown", reply_markup=kb_settings())
    elif d.startswith("lng_"): val = d.split("_",1)[1]; upsert_profile(uid, language=val); await safe_edit(q, f"✅ Language updated to *{val}*", parse_mode="Markdown", reply_markup=kb_settings())

def generate_nutrient_chart(name, calories, carbs, fat, protein):
    try:
        def _f(v):
            try: return max(0.0, float(v))
            except: return 0.0
        c, f, p, kcal = _f(carbs), _f(fat), _f(protein), _f(calories)
        if c + f + p == 0: return None
        fig = plt.figure(figsize=(6,3.5), facecolor='#1a1a2e')
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax1.pie([c,f,p], labels=[f'Carbs\n{c}g',f'Fat\n{f}g',f'Protein\n{p}g'], colors=['#3498db','#e74c3c','#2ecc71'], autopct='%1.0f%%', startangle=90, textprops={'color':'white','fontsize':7}, wedgeprops={'linewidth':0.5,'edgecolor':'#1a1a2e'})
        ax1.set_title("Macros", color='white', fontsize=9, pad=6)
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#16213e')
        bars = ax2.bar(['Carbs','Fat','Protein'],[c,f,p], color=['#3498db','#e74c3c','#2ecc71'],width=0.5,edgecolor='none')
        for bar, val in zip(bars,[c,f,p]): ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val}g', ha='center', va='bottom', color='white', fontsize=7)
        ax2.set_ylabel('grams/100g', color='#aaa', fontsize=7)
        ax2.tick_params(colors='#aaa', labelsize=7)
        ax2.spines[['top','right','left','bottom']].set_color('#333')
        ax2.set_title("Per 100g", color='white', fontsize=9)
        title = (name[:40]+'…') if len(name)>40 else name
        fig.suptitle(f"{title}  |  {kcal:.0f} kcal/100g", color='white', fontsize=9, y=1.01)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=130)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Chart error: {e}")
        plt.close('all')
        return None

def generate_daily_chart(user_id):
    rows = get_today_log(user_id)
    if not rows: return None
    totals = {"Calories":0,"Carbs":0,"Fat":0,"Protein":0}
    for r in rows:
        totals["Calories"] += r[2]; totals["Carbs"] += r[3]
        totals["Fat"]      += r[4]; totals["Protein"] += r[5]
    rda  = {"Calories":2000,"Carbs":260,"Fat":65,"Protein":50}
    keys = list(totals.keys())
    pcts = [min(totals[k]/rda[k]*100,150) for k in keys]
    fig, ax = plt.subplots(figsize=(6,3), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    bars = ax.barh(keys, pcts, color=['#f39c12','#3498db','#e74c3c','#2ecc71'], edgecolor='none', height=0.5)
    ax.axvline(100, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    for bar, k in zip(bars, keys): ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f'{totals[k]:.0f}/{rda[k]}', va='center', color='white', fontsize=7)
    ax.set_xlabel('% of Daily Target', color='#aaa', fontsize=8)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[['top','right','left','bottom']].set_color('#333')
    ax.set_xlim(0, 160)
    fig.suptitle("IngreScanner — Today's Progress", color='white', fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf

ALLERGEN_KW = {
    "gluten":    ["wheat","gluten","barley","rye","oats","spelt"],
    "dairy":     ["milk","lactose","cheese","butter","cream","whey","casein","yogurt"],
    "nuts":      ["almond","cashew","walnut","pecan","pistachio","hazelnut","peanut","nut"],
    "soy":       ["soy","soya","tofu","edamame"],
    "eggs":      ["egg","albumin"],
    "fish":      ["fish","cod","tuna","salmon","anchovy"],
    "shellfish": ["shrimp","crab","lobster","prawn","shellfish"],
    "sesame":    ["sesame","tahini"],
}
LANG_NOTE = {
    "English":"", "Tamil":"Respond entirely in Tamil.",
    "Hindi":"Respond entirely in Hindi.",
    "Malayalam":"Respond entirely in Malayalam.",
    "Telugu":"Respond entirely in Telugu.",
}

def detect_allergens(ing: str, user_alg: str) -> dict:
    text     = (ing or "").lower()
    found    = {g:[k for k in kws if k in text] for g,kws in ALLERGEN_KW.items() if any(k in text for k in kws)}
    ua       = user_alg.lower().strip()
    triggered= [g for g in found if ua in g or g in ua]
    return {"found":found, "triggered":triggered}

def fmt_allergens(ad: dict) -> str:
    if not ad["found"]: return "✅ No major allergens detected."
    lines = []
    if ad["triggered"]: lines.append("🚨 *PERSONAL ALLERGEN ALERT!*")
    for g, hits in ad["found"].items():
        e = "🚨" if g in ad["triggered"] else "⚠️"
        lines.append(f"{e} {g.upper()}: {', '.join(hits)}")
    return "\n".join(lines)

def diet_check(product: dict, diet: str) -> str:
    d   = diet.lower()
    ing = (product.get("ingredients","") or "").lower()
    c   = float(product.get("carbs",0)   or 0)
    fat = float(product.get("fat",0)     or 0)
    sug = float(product.get("sugar",0)   or 0)
    pro = float(product.get("protein",0) or 0)
    if "keto"         in d: return f"{'✅' if c<10  else '❌'} Keto — Carbs: {c}g (limit 10g)"
    if "vegan"        in d:
        a = any(w in ing for w in ["milk","egg","meat","chicken","fish","butter","cream","honey"])
        return f"{'❌' if a else '✅'} Vegan {'— animal ingredient found' if a else ''}"
    if "diabet"       in d: return f"{'✅' if sug<5 and c<20 else '❌'} Diabetic-Friendly — Sugar:{sug}g Carbs:{c}g"
    if "high protein" in d: return f"{'✅' if pro>=15 else '❌'} High-Protein — {pro}g/100g"
    if "low fat"      in d: return f"{'✅' if fat<3  else '❌'} Low-Fat — {fat}g/100g"
    return ""

async def gemini_estimate_nutrition(name: str, ingredients: str) -> dict:
    ing_str = ingredients if len(ingredients) > 10 else f"typical ingredients for {name}"
    prompt = f"""You are a nutrition database expert.
Estimate realistic nutritional values per 100g for this product:
Product: {name}
Ingredients: {ing_str}
Reply with ONLY a JSON object, no explanation, no markdown, no backticks.
Use this exact format:
{{"energy_kcal": 0, "carbs_g": 0, "sugar_g": 0, "fat_g": 0, "saturated_fat_g": 0, "protein_g": 0, "fiber_g": 0, "salt_g": 0}}
Base your estimates on standard nutritional databases for this type of product.
All values must be realistic numbers, never 0 unless truly zero.
"""
    try:
        r    = await generate_with_retry(prompt)
        text = r.text.strip().replace("```json","").replace("```","").strip()
        data = __import__('json').loads(text)
        return {
            "energy":  float(data.get("energy_kcal", 0)),
            "carbs":   float(data.get("carbs_g",     0)),
            "sugar":   float(data.get("sugar_g",     0)),
            "fat":     float(data.get("fat_g",       0)),
            "protein": float(data.get("protein_g",   0)),
            "fiber":   float(data.get("fiber_g",     0)),
            "salt":    float(data.get("salt_g",      0)),
        }
    except Exception as e:
        logger.error(f"Nutrition estimation error: {e}")
        return {}

async def gemini_text(product: dict, walk_mins: int, profile: dict) -> str:
    cache_key = f"{product.get('name', 'unknown')}_{profile.get('condition', 'none')}_{profile.get('diet_pref', 'normal')}_{profile.get('language', 'English')}"
    if cache_key in _ai_report_cache:
        logger.info(f"⚡ IN-MEMORY CACHE HIT: Loaded report instantly! (0 API cost)")
        return _ai_report_cache[cache_key]

    lang = LANG_NOTE.get(profile.get("language","English"),"")
    ing  = product.get("ingredients","") or ""
    if len(ing) < 15:
        ing = f"[Unavailable — infer typical ingredients for: {product.get('name','')}]"
    prompt = f"""
You are IngreScanner AI, a professional nutritionist. {lang}

PRODUCT: {product.get('name','Unknown')}
INGREDIENTS: {ing}
NUTRITION/100g: {product.get('energy',0)} kcal | Carbs {product.get('carbs',0)}g | Fat {product.get('fat',0)}g | Protein {product.get('protein',0)}g | Sugar {product.get('sugar',0)}g | Salt {product.get('salt',0)}g
DATA SOURCE: {"AI-estimated" if product.get('ai_estimated') else "OpenFoodFacts database"}
IMPORTANT: Do NOT mention missing or incorrect data to the user. Use the provided values confidently.
USER DIET: {profile.get('diet_pref','Normal')}
USER CONDITION: {profile.get('condition','None')}
USER ALLERGY: {profile.get('allergies','None')}

OUTPUT FORMAT — plain text only, no markdown symbols like # or *:

VERDICT
[One punchy sentence summarising if this product is good or bad overall]

⚠️ CONDITION WARNING ({profile.get('condition','None')})
[If condition is None write: No specific condition warnings. Otherwise write exactly why this is risky/safe for this condition]

🚩 RED FLAGS
[Bullet list of specific health concerns. Write "None detected" if genuinely healthy.]

🔍 INGREDIENT DECODER
[List every ingredient with the format below.]
👉 [Ingredient Name]
  • What: [what it is in plain language]
  • Why: [why it is used in this product]
  • Safety: GOOD 🟢 / CAUTION 🟡 / AVOID 🔴

✅ HEALTHIER ALTERNATIVES
1. [specific product or food]
2. [specific product or food]

🏃 BURN IT OFF
Walk {walk_mins} min OR Cycle {walk_mins//2} min to burn a 100g serving.
"""
    try:
        r = await generate_with_retry(prompt)
        final_text = r.text.strip()
        _ai_report_cache[cache_key] = final_text 
        return final_text
    except Exception as e:
        logger.error(f"AI text generation error: {e}")
        return "⚠️ Analysis failed due to server connection issues. Please try again."

async def gemini_vision(photo_bytes: bytes, profile: dict) -> str:
    lang = LANG_NOTE.get(profile.get("language","English"),"")
    try:
        image  = Image.open(io.BytesIO(photo_bytes))
        prompt = f"""
You are IngreScanner AI. {lang}
Analyse this food product photo:
1. Identify product name and brand
2. List all visible/likely ingredients with 👉 format (What | Why | Safety 🟢🟡🔴)
3. Health Score /10
4. Allergen alert for: {profile.get('allergies','None')}
5. Condition warning for: {profile.get('condition','None')}
6. Two healthier alternatives
Plain text only.
"""
        r = await generate_with_retry(prompt, image=image)
        return r.text.strip()
    except Exception as e:
        logger.error(f"Vision error: {e}")
        return "Vision analysis failed. Please send a clearer photo."

def generate_comparison_chart(info_a: dict, info_b: dict):
    try:
        metrics = ["Calories", "Carbs", "Sugar", "Fat", "Protein", "Salt"]
        vals_a  = [float(info_a.get("energy",0) or 0), float(info_a.get("carbs",0) or 0), float(info_a.get("sugar",0)  or 0), float(info_a.get("fat",0)   or 0), float(info_a.get("protein",0)or 0), float(info_a.get("salt",0)  or 0)]
        vals_b  = [float(info_b.get("energy",0) or 0), float(info_b.get("carbs",0) or 0), float(info_b.get("sugar",0)  or 0), float(info_b.get("fat",0)   or 0), float(info_b.get("protein",0)or 0), float(info_b.get("salt",0)  or 0)]

        x     = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        name_a = (info_a["name"][:20]+"…") if len(info_a["name"])>20 else info_a["name"]
        name_b = (info_b["name"][:20]+"…") if len(info_b["name"])>20 else info_b["name"]

        bars_a = ax.bar(x - width/2, vals_a, width, label=name_a, color="#3498db", edgecolor="none", alpha=0.9)
        bars_b = ax.bar(x + width/2, vals_b, width, label=name_b, color="#e74c3c", edgecolor="none", alpha=0.9)

        for bar in bars_a:
            h = bar.get_height()
            if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.0f}", ha="center", va="bottom", color="white", fontsize=7)
        for bar in bars_b:
            h = bar.get_height()
            if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.0f}", ha="center", va="bottom", color="white", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="white", fontsize=8)
        ax.set_ylabel("per 100g", color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e")
        ax.spines[["top","right","left","bottom"]].set_color("#21262d")
        ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8)
        fig.suptitle("VS  Nutrition Comparison (per 100g)", color="white", fontsize=11, fontweight="bold")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=140)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Comparison chart error: {e}")
        plt.close("all")
        return None

async def gemini_compare(info_a: dict, info_b: dict, profile: dict) -> str:
    lang = LANG_NOTE.get(profile.get("language","English"), "")
    prompt = f"""You are IngreScanner AI, a professional nutritionist. {lang}

Compare these two food products for a user with:
- Health condition: {profile.get("condition","None")}
- Diet preference: {profile.get("diet_pref","Normal")}
- Allergies: {profile.get("allergies","None")}

{info_a["name"]}:
Calories: {info_a.get("energy",0)} kcal | Carbs: {info_a.get("carbs",0)}g | Sugar: {info_a.get("sugar",0)}g | Fat: {info_a.get("fat",0)}g | Protein: {info_a.get("protein",0)}g | Salt: {info_a.get("salt",0)}g | Score: {info_a.get("score",0)}/100

{info_b["name"]}:
Calories: {info_b.get("energy",0)} kcal | Carbs: {info_b.get("carbs",0)}g | Sugar: {info_b.get("sugar",0)}g | Fat: {info_b.get("fat",0)}g | Protein: {info_b.get("protein",0)}g | Salt: {info_b.get("salt",0)}g | Score: {info_b.get("score",0)}/100

Reply in this EXACT format (plain text only, no markdown):

WINNER: [Product name]

WHY: [2-3 sentences explaining exactly why this product wins for this specific user's condition and diet. Mention specific numbers. Be direct.]

WATCH OUT: [One thing the winning product still has that the user should be aware of]

BOTTOM LINE: [One sentence — practical advice for this user right now]
"""
    try:
        r = await generate_with_retry(prompt)
        return r.text.strip()
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return ""

async def run_comparison(update: Update, context: ContextTypes.DEFAULT_TYPE, info: dict, uid: int):
    session = _compare_cache.get(uid, {})
    profile = get_profile(uid)

    if session.get("step") == "A":
        _compare_cache[uid] = {"step": "B", "info_a": info}
        await update.message.reply_text(
            f"✅ Got *{info['name']}*!\n\n📸 Now send me the *second product* to compare it against.",
            parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel Compare", callback_data="cmp_cancel")]])
        )
        return

    info_a = session["info_a"]
    info_b = info
    _compare_cache.pop(uid, None)

    info_a["score"] = compute_product_score(info_a)
    info_b["score"] = compute_product_score(info_b)

    await update.message.reply_text("⚖️ Comparing products…")

    def _w(a, b, lower_better=True):
        if a == b: return "🟰", "🟰"
        if lower_better: return ("✅","❌") if a < b else ("❌","✅")
        else:            return ("✅","❌") if a > b else ("❌","✅")

    cal_w  = _w(info_a.get("energy",0),  info_b.get("energy",0),  lower_better=True)
    carb_w = _w(info_a.get("carbs",0),   info_b.get("carbs",0),   lower_better=True)
    sug_w  = _w(info_a.get("sugar",0),   info_b.get("sugar",0),   lower_better=True)
    fat_w  = _w(info_a.get("fat",0),     info_b.get("fat",0),     lower_better=True)
    pro_w  = _w(info_a.get("protein",0), info_b.get("protein",0), lower_better=False)
    sal_w  = _w(info_a.get("salt",0),    info_b.get("salt",0),    lower_better=True)
    scr_w  = _w(info_a["score"],         info_b["score"],         lower_better=False)

    name_a = (info_a["name"][:22]+"…") if len(info_a["name"])>22 else info_a["name"]
    name_b = (info_b["name"][:22]+"…") if len(info_b["name"])>22 else info_b["name"]

    def _fmt(v, fmt):
        if fmt == "int":  return f"{v:.0f}"
        if fmt == "salt": return f"{v:.2f}"
        return f"{v:.1f}"

    def _row(label, va, vb, wa, wb, fmt=""): return (f"*{label}*\n  1️⃣  `{_fmt(va,fmt)}`  {wa}\n  2️⃣  `{_fmt(vb,fmt)}`  {wb}\n\n")

    est_note = " _(AI est.)_" if info_a.get("ai_estimated") else ""
    est_note += " _(AI est.)_" if info_b.get("ai_estimated") else ""

    header = (f"1️⃣ *{name_a}*\n        🆚\n2️⃣ *{name_b}*{est_note}\n\n✅ = better   ❌ = worse\n━━━━━━━━━━━━━━━━━━━━━━\n\n")

    rows = (
          _row("🔥 Calories",  info_a.get("energy",0),  info_b.get("energy",0),  cal_w[0],  cal_w[1],  "int")
        + _row("🌾 Carbs g",   info_a.get("carbs",0),   info_b.get("carbs",0),   carb_w[0], carb_w[1])
        + _row("🍬 Sugar g",   info_a.get("sugar",0),   info_b.get("sugar",0),   sug_w[0],  sug_w[1])
        + _row("🥑 Fat g",     info_a.get("fat",0),     info_b.get("fat",0),     fat_w[0],  fat_w[1])
        + _row("💪 Protein g", info_a.get("protein",0), info_b.get("protein",0), pro_w[0],  pro_w[1])
        + _row("🧂 Salt g",    info_a.get("salt",0),    info_b.get("salt",0),    sal_w[0],  sal_w[1], "salt")
        + f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + _row("🏅 Score/100", info_a["score"], info_b["score"], scr_w[0], scr_w[1], "int")
    )

    await update.message.reply_text(header + rows, parse_mode="Markdown")

    chart = generate_comparison_chart(info_a, info_b)
    if chart: await update.message.reply_photo(photo=chart, caption="📊 Side-by-side nutrition comparison")

    await update.message.reply_text("🧠 Getting AI verdict…")
    verdict = await gemini_compare(info_a, info_b, profile)

    winner_line = ""
    if verdict:
        for line in verdict.splitlines():
            if line.startswith("WINNER:"):
                winner_name = line.replace("WINNER:","").strip()
                winner_line = f"🏆 *WINNER: {winner_name}*\n\n"
                verdict = verdict.replace(line, "").strip()
                break

    final_text = (winner_line + verdict).strip()
    if not final_text: final_text = "⚠️ AI verdict unavailable right now. Please try again later."

    await update.message.reply_text(final_text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🆚 Compare Again", callback_data="menu_compare")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))

async def process_barcode(update: Update, context: ContextTypes.DEFAULT_TYPE, barcode: str, photo_bytes: bytes):
    uid     = update.message.from_user.id
    profile = get_profile(uid)
    status  = await update.message.reply_text(f"🔍 Barcode: `{barcode}`\nFetching from database…", parse_mode="Markdown")
    
    data = None
    for attempt in range(3):
        try:
            r = requests.get(f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json", timeout=30, headers={"User-Agent": "IngreScanner/1.0"})
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            if attempt < 2: await asyncio.sleep(3)
            continue

    if data is None or data.get("status") == 0:
        await status.edit_text("⚠️ Database empty or unreachable. Trying Vision AI…")
        await update.message.reply_text(await gemini_vision(photo_bytes, profile), reply_markup=kb_main())
        return

    p = data["product"]
    n = p.get("nutriments", {})
    def _f(k):
        try: return float(n.get(k) or 0)
        except: return 0.0

    info = {
        "name":        p.get("product_name") or p.get("product_name_en") or "Unknown Product",
        "brand":       p.get("brands",""),
        "ingredients": p.get("ingredients_text") or p.get("ingredients_text_en") or "",
        "energy":      _f("energy-kcal_100g"),
        "carbs":       _f("carbohydrates_100g"),
        "fat":         _f("fat_100g"),
        "protein":     _f("proteins_100g"),
        "fiber":       _f("fiber_100g"),
        "sugar":       _f("sugars_100g"),
        "salt":        _f("salt_100g"),
        "nutriscore":  p.get("nutriscore_grade","").upper(),
        "image":       p.get("image_front_url",""),
        "ai_estimated":False,
    }

    nutrition_keys   = ["energy","carbs","fat","protein"]
    if all(info.get(k, 0) == 0 for k in nutrition_keys):
        await update.message.reply_text("⚠️ *Nutrition data missing.*\n🧠 Asking AI to estimate values…", parse_mode="Markdown")
        estimated = await gemini_estimate_nutrition(info["name"], info["ingredients"])
        if estimated:
            info.update(estimated)
            info["ai_estimated"] = True

    product_score = compute_product_score(info)
    save_scan_with_score(uid, barcode, info["name"], info["energy"], info["carbs"], info["fat"], info["protein"], info["nutriscore"], product_score)

    walk_mins     = max(5, int(info["energy"] / 4))
    allergen_text = info["ingredients"]
    if len(allergen_text.strip()) < 10:
        name_lower = info["name"].lower()
        hints = []
        if any(w in name_lower for w in ["milk","dairy","cheese","butter","cream","shakti","badam","lassi"]): hints.append("milk dairy")
        if any(w in name_lower for w in ["wheat","maida","biscuit","bread","roti","noodle","pasta","cake","cookie"]): hints.append("wheat gluten")
        if any(w in name_lower for w in ["peanut","groundnut","nut","almond","cashew"]): hints.append("nuts peanut")
        if any(w in name_lower for w in ["egg","mayonnaise"]): hints.append("egg")
        if any(w in name_lower for w in ["soy","soya","tofu"]): hints.append("soy")
        allergen_text = " ".join(hints) if hints else allergen_text

    allergen_data = detect_allergens(allergen_text, profile.get("allergies","None"))
    dc            = diet_check(info, profile.get("diet_pref","Normal"))
    ns_e          = {"A":"🟢","B":"🟡","C":"🟠","D":"🔴","E":"⛔"}.get(info["nutriscore"],"❓")
    ps_grade      = "🟢" if product_score>=70 else "🟡" if product_score>=45 else "🔴"

    header = (
        f"📦 *{info['name']}*\n🏭 {info['brand']}\n🔖 Nutri-Score: {ns_e} {info['nutriscore'] or 'N/A'}\n💯 Product Score: {ps_grade} *{product_score:.0f}/100*"
        + (" _(AI estimated)_" if info.get("ai_estimated") else "") + f"\n\n🌾 *ALLERGEN CHECK*\n{fmt_allergens(allergen_data)}"
    )
    if dc: header += f"\n\n🥗 *DIET CHECK*\n{dc}"

    await status.delete()

    if info["image"]:
        try: await update.message.reply_photo(photo=info["image"], caption=header, parse_mode="Markdown")
        except: await update.message.reply_text(header, parse_mode="Markdown")
    else:
        await update.message.reply_text(header, parse_mode="Markdown")

    chart = generate_nutrient_chart(info["name"], info["energy"], info["carbs"], info["fat"], info["protein"])
    if chart:
        await update.message.reply_photo(photo=chart, caption="📊 Nutrient Breakdown")
    else:
        est_note = " _(AI estimated)_" if info.get("ai_estimated") else ""
        await update.message.reply_text(f"📊 Per 100g{est_note}: {info['energy']:.0f} kcal | Carbs {info['carbs']}g | Fat {info['fat']}g | Protein {info['protein']}g", parse_mode="Markdown")

    await update.message.reply_text("🧠 Running AI ingredient analysis…")
    ai_text = await gemini_text(info, walk_mins, profile)

    MAX_LENGTH = 4000
    if len(ai_text) <= MAX_LENGTH: await update.message.reply_text(ai_text)
    else:
        for part in [ai_text[i:i+MAX_LENGTH] for i in range(0, len(ai_text), MAX_LENGTH)]:
            await update.message.reply_text(part)

    ck = f"{uid}_{int(datetime.now().timestamp())}"
    _log_cache[ck] = {
        "barcode": barcode, "name": info["name"][:40], "cal": info["energy"], "carb": info["carbs"],
        "fat": info["fat"], "prot": info["protein"], "ai_estimated": info.get("ai_estimated", False),
    }
    est_note = " _(nutrition AI-estimated)_" if info.get("ai_estimated") else ""
    await update.message.reply_text(
        f"➕ *Log this to your calorie diary?*{est_note}\n_{info['name'][:35]} — {info['energy']:.0f} kcal per 100g_\n\nHow much did you eat?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🍪 1 piece (~30g)", callback_data=f"log30|{ck}"), InlineKeyboardButton("🥣 Half pack (~50g)", callback_data=f"log50|{ck}")],
            [InlineKeyboardButton("📦 100g", callback_data=f"log|{ck}"), InlineKeyboardButton("📦 Full pack (~200g)", callback_data=f"log200|{ck}")],
            [InlineKeyboardButton("✏️ Enter custom grams", callback_data=f"logcustom|{ck}")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")],
        ])
    )

async def _fetch_and_compare(update: Update, context: ContextTypes.DEFAULT_TYPE, barcode: str, photo_bytes: bytes, uid: int):
    st = await update.message.reply_text("🔍 Fetching product…")
    data = None
    for attempt in range(3):
        try:
            r = requests.get(f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json", timeout=30, headers={"User-Agent": "IngreScanner/1.0"})
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            if attempt < 2: await asyncio.sleep(3)
            continue

    if data is None or data.get("status") == 0:
        await st.edit_text("📦 Product not found in database or unreachable.\nTry a different product or angle.")
        return

    p = data["product"]
    n = p.get("nutriments", {})
    def _f(k):
        try: return float(n.get(k) or 0)
        except: return 0.0

    info = {
        "name": p.get("product_name") or p.get("product_name_en") or "Unknown Product", "brand": p.get("brands",""),
        "ingredients": p.get("ingredients_text") or p.get("ingredients_text_en") or "", "energy": _f("energy-kcal_100g"),
        "carbs": _f("carbohydrates_100g"), "fat": _f("fat_100g"), "protein": _f("proteins_100g"),
        "fiber": _f("fiber_100g"), "sugar": _f("sugars_100g"), "salt": _f("salt_100g"),
        "nutriscore": p.get("nutriscore_grade","").upper(), "image": p.get("image_front_url",""), "ai_estimated":False,
    }

    if all(info.get(k,0) == 0 for k in ["energy","carbs","fat","protein"]):
        await st.edit_text("🧠 Estimating nutrition with AI…")
        estimated = await gemini_estimate_nutrition(info["name"], info["ingredients"])
        if estimated:
            info.update(estimated)
            info["ai_estimated"] = True

    await st.delete()
    await run_comparison(update, context, info, uid)

async def handle_compare_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _compare_cache.pop(q.from_user.id, None)
    await q.edit_message_text("❌ Comparison cancelled.", reply_markup=kb_main())

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo: return
    profile = get_profile(update.message.from_user.id)
    if not profile["onboarded"]:
        await update.message.reply_text("👋 Please complete your quick setup first!\nTap /start 👇")
        return
    try:
        st   = await update.message.reply_text("📥 Scanning photo…")
        pf   = await update.message.photo[-1].get_file()
        pb   = await pf.download_as_bytearray()
        img  = cv2.imdecode(np.frombuffer(pb, np.uint8), cv2.IMREAD_COLOR)
        res  = zxingcpp.read_barcodes(img)
        await st.delete()
        
        uid = update.message.from_user.id
        if res:
            barcode = res[0].text
            if uid in _compare_cache and _compare_cache[uid].get("step") in ("A","B"): await _fetch_and_compare(update, context, barcode, pb, uid)
            else: await process_barcode(update, context, barcode, pb)
        else:
            if uid in _compare_cache and _compare_cache[uid].get("step") in ("A","B"):
                await update.message.reply_text("⚠️ No barcode found in this photo.\nPlease send a clearer photo of the product barcode.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel Compare", callback_data="cmp_cancel")]]))
            else:
                await update.message.reply_text("🔍 No barcode found — running Vision AI analysis…")
                await update.message.reply_text(await gemini_vision(pb, profile), reply_markup=kb_main())
    except Exception as e:
        logger.error(f"handle_photo: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error processing photo. Please try again.", reply_markup=kb_main())

async def handle_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    try:
        parts  = q.data.split("|", 1)
        action, ck = parts[0], parts[1]
        e = _log_cache.get(ck)
        if not e:
            await q.edit_message_text("⚠️ Session expired. Rescan to log again.", reply_markup=kb_main())
            return

        if action == "logcustom":
            context.user_data["log_ck"] = ck
            context.user_data["awaiting_log_grams"] = True
            await q.edit_message_text(f"✏️ *How many grams did you eat?*\n_{e['name'][:35]}_\n\nType the number of grams and send it as a message.\nExample: *45*", parse_mode="Markdown")
            return

        grams = {"log": 100, "log30": 30, "log50": 50, "log200": 200}.get(action, 100)
        log_food(q.from_user.id, e["barcode"], e["name"], grams, e["cal"], e["carb"], e["fat"], e["prot"])
        _log_cache.pop(ck, None)
        logged_cal = round(e["cal"] * grams / 100, 1)
        await q.edit_message_text(f"✅ Logged *{grams}g* of *{e['name']}*!\n🔥 {logged_cal:.0f} kcal added to today's diary.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🍽 View Today's Log", callback_data="menu_today")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
    except Exception as err:
        logger.error(f"log callback: {err}")
        await q.edit_message_text("⚠️ Logging failed.", reply_markup=kb_main())

async def handle_custom_grams(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_log_grams"): return
    try:
        grams = float(update.message.text.strip())
        if grams <= 0 or grams > 5000:
            await update.message.reply_text("⚠️ Please enter a valid amount between 1 and 5000 grams.")
            return
        ck = context.user_data.get("log_ck")
        e  = _log_cache.get(ck)
        if not e:
            await update.message.reply_text("⚠️ Session expired. Rescan to log again.", reply_markup=kb_main())
            context.user_data["awaiting_log_grams"] = False
            return

        log_food(update.message.from_user.id, e["barcode"], e["name"], grams, e["cal"], e["carb"], e["fat"], e["prot"])
        _log_cache.pop(ck, None)
        context.user_data["awaiting_log_grams"] = False
        context.user_data["log_ck"] = None

        logged_cal = round(e["cal"] * grams / 100, 1)
        await update.message.reply_text(f"✅ Logged *{grams}g* of *{e['name']}*!\n🔥 {logged_cal:.0f} kcal added to today's diary.", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🍽 View Today's Log", callback_data="menu_today")], [InlineKeyboardButton("🏠 Main Menu", callback_data="menu_home")]]))
    except ValueError:
        await update.message.reply_text("⚠️ Please type just a number. Example: 45")

def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN missing from .env!")
        return
    init_db()
    app = ApplicationBuilder().token(BOT_TOKEN).request(HTTPXRequest(connect_timeout=60.0, read_timeout=60.0)).build()

    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            ASK_DIET:      [CallbackQueryHandler(ob_diet,      pattern=r"^diet_")],
            ASK_ALLERGY:   [CallbackQueryHandler(ob_allergy,   pattern=r"^alg_")],
            ASK_CONDITION: [CallbackQueryHandler(ob_condition, pattern=r"^cnd_")],
            ASK_LANGUAGE:  [CallbackQueryHandler(ob_language,  pattern=r"^lng_")],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
        per_user=True, per_chat=True,
    ))

    app.add_handler(CallbackQueryHandler(handle_menu,            pattern=r"^menu_"))
    app.add_handler(CallbackQueryHandler(handle_settings,        pattern=r"^(set_|diet_|alg_|cnd_|lng_)"))
    app.add_handler(CallbackQueryHandler(handle_log,             pattern=r"^(log|log30|log50|log200|logcustom)\|"))
    app.add_handler(CallbackQueryHandler(handle_compare_cancel, pattern=r"^cmp_cancel"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_grams))

    print("🚀 IngreScanner Bot is running with Hybrid AI Engine (Gemini + Groq)…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()