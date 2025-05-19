from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
import validators
import trafilatura
from newspaper import Article
from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import logging
import time
from urllib.parse import urlparse
from datetime import datetime
import json
import os
from langdetect import detect
from pdf_handler import PDFProcessor
from image_handler import ImageProcessor
from feedback_handler import FeedbackHandler
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Use a secure random key in production

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for Flask-Login and SQLAlchemy
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    
    def __init__(self, email, name):
        self.email = email
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Initialize models
try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    logging.info("Loaded DistilBART summarizer")
except Exception:
    logging.exception("Failed to load summarizer")
    summarizer = None

try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
    logging.info("Loaded emotion classifier")
except Exception:
    logging.exception("Failed to load emotion classifier")
    emotion_classifier = None

# Language codes mapping
LANGUAGE_CODES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'fa': 'Persian',
    'id': 'Indonesian',
    'ms': 'Malay',
    'ta': 'Tamil',
    'te': 'Telugu',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'si': 'Sinhala',
    'ne': 'Nepali',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'bo': 'Tibetan',
    'ug': 'Uyghur',
    'mn': 'Mongolian',
    'ka': 'Georgian',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'uz': 'Uzbek',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'tg': 'Tajik',
    'ps': 'Pashto',
    'ku': 'Kurdish',
    'he': 'Hebrew',
    'yi': 'Yiddish',
    'am': 'Amharic',
    'ti': 'Tigrinya',
    'om': 'Oromo',
    'so': 'Somali',
    'sw': 'Swahili',
    'rw': 'Kinyarwanda',
    'ny': 'Chichewa',
    'sn': 'Shona',
    'st': 'Sesotho',
    'tn': 'Setswana',
    'ts': 'Tsonga',
    'ss': 'Swati',
    've': 'Venda',
    'nr': 'Southern Ndebele',
    'xh': 'Xhosa',
    'zu': 'Zulu',
    'af': 'Afrikaans',
    'lb': 'Luxembourgish',
    'fy': 'Western Frisian',
    'gd': 'Scottish Gaelic',
    'cy': 'Welsh',
    'br': 'Breton',
    'is': 'Icelandic',
    'fo': 'Faroese',
    'gl': 'Galician',
    'eu': 'Basque',
    'ca': 'Catalan',
    'ast': 'Asturian',
    'an': 'Aragonese',
    'oc': 'Occitan',
    'co': 'Corsican',
    'sc': 'Sardinian',
    'wa': 'Walloon',
    'fur': 'Friulian',
    'lij': 'Ligurian',
    'lmo': 'Lombard',
    'nap': 'Neapolitan',
    'scn': 'Sicilian',
    'vec': 'Venetian',
    'roa-tara': 'Tarantino',
    'pms': 'Piedmontese',
    'rm': 'Romansh',
    'lad': 'Ladino',
    'frp': 'Franco-Provençal',
    'gsw': 'Swiss German',
    'bar': 'Bavarian',
    'cim': 'Cimbrian',
    'pfl': 'Palatinate German',
    'ksh': 'Colognian',
    'nds': 'Low German',
    'stq': 'Saterland Frisian',
    'vls': 'West Flemish',
    'zea': 'Zeelandic',
    'li': 'Limburgish',
    'sli': 'Lower Silesian',
    'frr': 'North Frisian',
    'dsb': 'Lower Sorbian',
    'hsb': 'Upper Sorbian',
    'sma': 'Southern Sami',
    'se': 'Northern Sami',
    'smj': 'Lule Sami',
    'sme': 'Inari Sami',
    'smn': 'Skolt Sami',
    'sms': 'Skolt Sami',
    'sje': 'Pite Sami',
    'sju': 'Ume Sami',
    'sma': 'Southern Sami',
    'smj': 'Lule Sami',
    'sme': 'Inari Sami',
    'smn': 'Skolt Sami',
    'sms': 'Skolt Sami'
}

def detect_language(text):
    """Detect the language of the text using multiple methods."""
    try:
        # First try with Google Translate
        translator = GoogleTranslator(source='auto', target='en')
        detected = translator.detect(text)
        lang_code = detected['lang']
        
        # If confidence is low or language is unknown, try langdetect as backup
        if detected.get('confidence', 0) < 0.5 or lang_code == 'und':
            try:
                lang_code = detect(text)
            except:
                lang_code = 'en'
        
        # Normalize language code
        lang_code = lang_code.lower()
        if lang_code not in LANGUAGE_CODES:
            lang_code = 'en'
            
        return lang_code
    except Exception as e:
        logging.error(f"Language detection error: {str(e)}")
        try:
            # Fallback to langdetect
            return detect(text)
        except:
            return 'en'  # final fallback to English

def translate_text(text, target_lang, source_lang='auto'):
    """Translate text to target language with improved error handling and chunking."""
    try:
        # Normalize language codes
        target_lang = target_lang.lower()
        source_lang = source_lang.lower()
        
        # Validate language codes
        if target_lang not in LANGUAGE_CODES:
            logging.warning(f"Invalid target language: {target_lang}, defaulting to English")
            target_lang = 'en'
        if source_lang != 'auto' and source_lang not in LANGUAGE_CODES:
            logging.warning(f"Invalid source language: {source_lang}, using auto detection")
            source_lang = 'auto'

        # If source and target are the same, return original text
        if source_lang != 'auto' and source_lang == target_lang:
            return text

        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        # Split text into smaller chunks for better translation
        # Google Translate has a limit of 5000 characters
        max_chunk_size = 4000  # Slightly lower to be safe
        chunks = []
        
        # Split by sentences first
        sentences = text.split('. ')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        # Translate chunks with delay to avoid rate limiting
        translated_chunks = []
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
                time.sleep(0.5)  # Add delay between chunks
            except Exception as e:
                logging.error(f"Error translating chunk: {str(e)}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        return ' '.join(translated_chunks)
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

# Data storage (in production, use a proper database)
SEARCH_HISTORY_FILE = 'search_history.json'
FEEDBACK_FILE = 'feedback.json'

def load_search_history():
    if os.path.exists(SEARCH_HISTORY_FILE):
        with open(SEARCH_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_search_history(history):
    with open(SEARCH_HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback(feedback):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f)

def get_domain(url: str) -> str:
    parsed = urlparse(url)
    h = parsed.netloc.lower()
    return h[4:] if h.startswith("www.") else h

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Check if email ends with common email domains
        valid_domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'hotmail.com']
        is_valid_email = any(email.lower().endswith(domain) for domain in valid_domains)
        
        # Allow admin access with specific email patterns and fixed password
        if is_valid_email and password == "admin":
            user = User.query.filter_by(email=email).first()
            if not user:
                # Create new user if doesn't exist
                user = User(email=email, name=email.split('@')[0])
                db.session.add(user)
                db.session.commit()
            
            login_user(user)
            return redirect(url_for("dashboard"))
        
        flash("Invalid email or password. Please try again.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
@login_required
def dashboard():
    recent_searches = load_search_history()[-5:]  # Get last 5 searches
    return render_template("dashboard.html", recent_searches=recent_searches)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_article():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        chosen_lang = request.form.get("language", "en").lower()
        try:
            word_count = int(request.form.get("word_count", 100))
        except (TypeError, ValueError):
            word_count = 100

        if not validators.url(url):
            flash("Please enter a valid URL.", "error")
            return redirect(url_for("upload_article"))
        if word_count < 50:
            flash("Word count must be at least 50.", "error")
            return redirect(url_for("upload_article"))
        if word_count > 500:
            flash("Word count cannot exceed 500.", "error")
            return redirect(url_for("upload_article"))

        try:
            start = time.time()
            logging.info(f"Processing {url} → {word_count} words, lang={chosen_lang}")

            # Show processing splash screen
            return render_template("processing.html", url=url, word_count=word_count, language=chosen_lang)

        except Exception as e:
            flash(f"Error processing article: {str(e)}", "error")
            return redirect(url_for("upload_article"))

    return render_template("upload.html", languages=LANGUAGE_CODES)

@app.route("/process-article", methods=["POST"])
@login_required
def process_article():
    """Background processing endpoint for article summarization."""
    try:
        url = request.form.get("url")
        word_count = int(request.form.get("word_count", 100))
        chosen_lang = request.form.get("language", "en").lower()

        # Fetch and extract article content
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return jsonify({"error": "Failed to download article"}), 400

        # Extract text
        try:
            extracted = trafilatura.extract(html, favor_recall=True)
        except Exception:
            extracted = None

        # Fallback to Newspaper3k
        article = Article(url)
        try:
            article.download(input_html=html)
            article.parse()
        except Exception:
            pass

        full_text = (extracted or article.text or "").strip()
        if not full_text:
            return jsonify({"error": "Could not extract any text from the article"}), 400
        if len(full_text.split()) < 100:
            return jsonify({"error": "Article too short to summarize (min 100 words)"}), 400

        # Detect article language
        detected_lang = detect_language(full_text)
        article_language = LANGUAGE_CODES.get(detected_lang, detected_lang.upper())
        logging.info(f"Detected article language: {article_language}")

        # Handle translation for summarization
        text_for_summary = full_text
        if detected_lang != 'en':
            try:
                text_for_summary = translate_text(full_text, 'en', source_lang=detected_lang)
                logging.info(f"Translated article from {article_language} to English for summarization")
            except Exception as e:
                logging.error(f"Translation to English failed: {str(e)}")
                return jsonify({"error": "Failed to translate article to English for summarization"}), 400

        # Generate summary in English
        summary_txt = ""
        if summarizer:
            try:
                out = summarizer(
                    text_for_summary,
                    max_length=word_count * 3,
                    min_length=word_count,
                    do_sample=False
                )
                summary_txt = out[0]["summary_text"]
                logging.info("Generated summary in English")
            except Exception as e:
                logging.error(f"Summarization error: {str(e)}")
                pass

        # Fallback to first N words if summarization fails
        words = summary_txt.split() if summary_txt else []
        if len(words) < word_count:
            words = text_for_summary.split()[:word_count]
        summary_txt = " ".join(words[:word_count]).rstrip(".,") + "."

        # Handle summary translation
        display_summary = summary_txt
        summary_language = 'English'

        if chosen_lang != 'en':
            try:
                display_summary = translate_text(summary_txt, chosen_lang, source_lang='en')
                summary_language = LANGUAGE_CODES.get(chosen_lang, chosen_lang.upper())
                logging.info(f"Translated summary to {summary_language}")
            except Exception as e:
                logging.error(f"Translation to {chosen_lang} failed: {str(e)}")
                # Keep English summary if translation fails
                pass

        # Emotion detection
        emotion_label = "neutral"
        if emotion_classifier:
            try:
                snippet = text_for_summary[:512]
                res = emotion_classifier(snippet)[0]
                emotion_label = res["label"].lower()
            except Exception as e:
                logging.error(f"Emotion classification error: {str(e)}")
                pass

        emoji_map = {
            "anger": "😠", "disgust": "🤢", "fear": "😨",
            "joy": "😊", "neutral": "😐", "sadness": "😢",
            "surprise": "😮",
        }
        emotion = f"{emotion_label.capitalize()} {emoji_map.get(emotion_label,'')}"

        # Save to history
        history = load_search_history()
        history.append({
            "id": len(history) + 1,
            "title": article.title or get_domain(url),
            "url": url,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment": emotion,
            "summary": display_summary,
            "top_image": article.top_image,
            "word_count": word_count,
            "article_language": article_language,
            "summary_language": summary_language
        })
        save_search_history(history)

        return jsonify({
            "success": True,
            "redirect": url_for("view_article", id=len(history))
        })

    except Exception as e:
        logging.error(f"Error processing article: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/article/<int:id>")
@login_required
def view_article(id):
    history = load_search_history()
    if 0 <= id - 1 < len(history):
        article_data = history[id - 1]
        return render_template("view_article.html", article=article_data)
    flash("Article not found", "error")
    return redirect(url_for("dashboard"))

# Initialize feedback handler
feedback_handler = FeedbackHandler()

@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    if request.method == "POST":
        try:
            email = request.form.get("email")
            rating = request.form.get("rating")
            feedback_type = request.form.get("feedback_type")
            message = request.form.get("message")

            if not all([email, rating, feedback_type, message]):
                flash("Please fill in all fields", "error")
                return redirect(url_for("feedback"))

            # Validate email
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                flash("Please enter a valid email address", "error")
                return redirect(url_for("feedback"))

            # Save feedback
            if feedback_handler.save_feedback(email, rating, feedback_type, message):
                flash("Thank you for your feedback!", "success")
            else:
                flash("Failed to save feedback. Please try again.", "error")
            
            return redirect(url_for("dashboard"))
        except Exception as e:
            logging.error(f"Error processing feedback: {str(e)}")
            flash("An error occurred while processing your feedback", "error")
            return redirect(url_for("feedback"))

    # Get feedback statistics for display
    feedback_stats = feedback_handler.get_feedback_stats()
    return render_template("feedback.html", stats=feedback_stats)

@app.route("/recent-searches")
@login_required
def recent_searches():
    history = load_search_history()
    return render_template("recent_searches.html", searches=history)

# Initialize PDF processor
pdf_processor = PDFProcessor()

@app.route("/upload-pdf", methods=["GET", "POST"])
@login_required
def upload_pdf():
    if request.method == "POST":
        if 'pdf_file' not in request.files:
            flash("No PDF file uploaded", "error")
            return redirect(request.url)
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            flash("No PDF file selected", "error")
            return redirect(request.url)
        
        if not pdf_file.filename.endswith('.pdf'):
            flash("Please upload a PDF file", "error")
            return redirect(request.url)

        try:
            word_count = int(request.form.get("word_count", 100))
        except (TypeError, ValueError):
            word_count = 100

        if word_count < 50:
            flash("Word count must be at least 50.", "error")
            return redirect(request.url)
        if word_count > 500:
            flash("Word count cannot exceed 500.", "error")
            return redirect(request.url)

        chosen_lang = request.form.get("language", "en")

        try:
            # Process the PDF
            result = pdf_processor.process_pdf(pdf_file, word_count, chosen_lang)
            
            # Save to history
            history = load_search_history()
            history.append({
                "id": len(history) + 1,
                "title": pdf_file.filename,
                "url": "PDF Upload",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": "Neutral 😐",  # PDFs don't get sentiment analysis
                "summary": result['summary'],
                "top_image": None,  # PDFs don't have images
                "word_count": word_count,
                "article_language": result['article_language'],
                "summary_language": result['summary_language']
            })
            save_search_history(history)

            return redirect(url_for("view_article", id=len(history)))
        except Exception as e:
            flash(f"Error processing PDF: {str(e)}", "error")
            return redirect(request.url)

    return render_template("upload_pdf.html", languages=LANGUAGE_CODES)

@app.route("/input-selection")
@login_required
def input_selection():
    logging.info("Accessing input selection page")
    return render_template("input_selection.html")

# Initialize image processor
image_processor = ImageProcessor()

@app.route("/upload-image", methods=["GET", "POST"])
@login_required
def upload_image():
    logging.info("Accessing image upload page")
    if request.method == "POST":
        if 'image_file' not in request.files:
            flash("No image file uploaded", "error")
            return redirect(request.url)
        
        image_file = request.files['image_file']
        if image_file.filename == '':
            flash("No image file selected", "error")
            return redirect(request.url)
        
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            flash("Please upload a valid image file", "error")
            return redirect(request.url)

        try:
            word_count = int(request.form.get("word_count", 100))
        except (TypeError, ValueError):
            word_count = 100

        if word_count < 50:
            flash("Word count must be at least 50.", "error")
            return redirect(request.url)
        if word_count > 500:
            flash("Word count cannot exceed 500.", "error")
            return redirect(request.url)

        chosen_lang = request.form.get("language", "en")

        try:
            # Process the image
            result = image_processor.process_image(image_file, word_count, chosen_lang)
            
            # Save to history
            history = load_search_history()
            history.append({
                "id": len(history) + 1,
                "title": f"Image: {image_file.filename}",
                "url": result['article_url'] or "Image Upload",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": "Neutral 😐",  # Images don't get sentiment analysis
                "summary": result['summary'],
                "top_image": None,  # Original image not stored
                "word_count": word_count,
                "article_language": result['article_language'],
                "summary_language": result['summary_language']
            })
            save_search_history(history)

            return redirect(url_for("view_article", id=len(history)))
        except Exception as e:
            flash(f"Error processing image: {str(e)}", "error")
            return redirect(request.url)

    return render_template("upload_image.html", languages=LANGUAGE_CODES)

# Add this after all routes are defined
logging.info("Registered routes: %s", [str(rule) for rule in app.url_map.iter_rules()])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
