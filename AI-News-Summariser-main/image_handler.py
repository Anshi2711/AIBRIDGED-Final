import pytesseract
from PIL import Image
import io
import logging
import requests
from transformers import pipeline
from deep_translator import GoogleTranslator
import time
from langdetect import detect
import re
from newspaper import Article
import validators

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class ImageProcessor:
    def __init__(self):
        try:
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6"
            )
            logging.info("Loaded DistilBART summarizer")
        except Exception as e:
            logging.exception("Failed to load summarizer")
            self.summarizer = None

    def extract_text_from_image(self, image_file):
        """Extract text from image using OCR."""
        try:
            # Convert image to PIL Image if it's a file object
            if isinstance(image_file, bytes):
                image = Image.open(io.BytesIO(image_file))
            else:
                image = Image.open(image_file)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            raise

    def find_matching_article(self, extracted_text):
        """Find a matching article based on extracted text."""
        try:
            # Extract potential keywords from the text
            words = re.findall(r'\b\w{4,}\b', extracted_text.lower())
            keywords = ' '.join(set(words[:5]))  # Use top 5 unique words as keywords
            
            # Use Google News search to find matching articles
            search_url = f"https://news.google.com/search?q={keywords}&hl=en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                # Extract the first article URL
                article_url = None
                for line in response.text.split('\n'):
                    if 'href="./articles/' in line:
                        article_url = line.split('href="./articles/')[1].split('"')[0]
                        article_url = f"https://news.google.com/articles/{article_url}"
                        break
                
                if article_url:
                    # Get article content
                    article = Article(article_url)
                    article.download()
                    article.parse()
                    return article.text, article_url
            
            return None, None
        except Exception as e:
            logging.error(f"Error finding matching article: {str(e)}")
            return None, None

    def detect_language(self, text):
        """Detect the language of the text."""
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
            
            return lang_code
        except Exception as e:
            logging.error(f"Language detection error: {str(e)}")
            try:
                return detect(text)
            except:
                return 'en'

    def translate_text(self, text, target_lang, source_lang='auto'):
        """Translate text to target language."""
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            # Split text into chunks if it's too long
            max_chunk_size = 5000
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
                time.sleep(0.5)  # Add delay between chunks
            
            return ' '.join(translated_chunks)
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text

    def process_image(self, image_file, word_count=100, target_lang='en'):
        """Process image and generate summary in target language."""
        try:
            # Extract text from image
            extracted_text = self.extract_text_from_image(image_file)
            if not extracted_text:
                raise ValueError("No text could be extracted from the image")

            # Find matching article
            article_text, article_url = self.find_matching_article(extracted_text)
            if not article_text:
                raise ValueError("Could not find a matching article")

            # Detect language
            detected_lang = self.detect_language(article_text)
            article_language = LANGUAGE_CODES.get(detected_lang, detected_lang.upper())
            logging.info(f"Detected article language: {article_language}")

            # Translate to English if needed for summarization
            text_for_summary = article_text
            if detected_lang != 'en':
                text_for_summary = self.translate_text(article_text, 'en', source_lang=detected_lang)
                logging.info(f"Translated article from {article_language} to English")

            # Generate summary
            summary_txt = ""
            if self.summarizer:
                try:
                    out = self.summarizer(
                        text_for_summary,
                        max_length=word_count * 3,
                        min_length=word_count,
                        do_sample=False
                    )
                    summary_txt = out[0]["summary_text"]
                    logging.info("Generated summary in English")
                except Exception as e:
                    logging.error(f"Summarization error: {str(e)}")
                    # Fallback to first N words
                    words = text_for_summary.split()[:word_count]
                    summary_txt = " ".join(words).rstrip(".,") + "."
            else:
                # Fallback to first N words if summarizer not available
                words = text_for_summary.split()[:word_count]
                summary_txt = " ".join(words).rstrip(".,") + "."

            # Handle translation to target language
            display_summary = summary_txt
            summary_language = 'English'

            if target_lang.lower() != 'en':
                try:
                    display_summary = self.translate_text(summary_txt, target_lang, source_lang='en')
                    summary_language = LANGUAGE_CODES.get(target_lang, target_lang.upper())
                    logging.info(f"Translated summary to {summary_language}")
                except Exception as e:
                    logging.error(f"Translation error: {str(e)}")
                    logging.info("Falling back to English summary")

            return {
                'article_language': article_language,
                'summary_language': summary_language,
                'summary': display_summary,
                'word_count': word_count,
                'article_url': article_url,
                'extracted_text': extracted_text
            }

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise

# Language codes mapping (same as in app.py)
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
    'zu': 'Zulu'
} 