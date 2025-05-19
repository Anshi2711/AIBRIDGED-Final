import PyPDF2
import io
import logging
from transformers import pipeline
from deep_translator import GoogleTranslator
import time
from langdetect import detect

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

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
    'zu': 'Zulu'
}

class PDFProcessor:
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

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            raise

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

    def process_pdf(self, pdf_file, word_count=100, target_lang='en'):
        """Process PDF file and generate summary in target language."""
        try:
            # Extract text from PDF
            full_text = self.extract_text_from_pdf(pdf_file)
            if not full_text:
                raise ValueError("No text could be extracted from the PDF")

            if len(full_text.split()) < 100:
                raise ValueError("PDF content too short (minimum 100 words required)")

            # Detect language
            detected_lang = self.detect_language(full_text)
            article_language = LANGUAGE_CODES.get(detected_lang, detected_lang.upper())
            logging.info(f"Detected PDF language: {article_language}")

            # Translate to English if needed for summarization
            text_for_summary = full_text
            if detected_lang != 'en':
                text_for_summary = self.translate_text(full_text, 'en', source_lang=detected_lang)
                logging.info(f"Translated PDF content from {article_language} to English")

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
                'word_count': word_count
            }

        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            raise 