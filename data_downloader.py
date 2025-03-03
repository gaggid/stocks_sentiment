import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import datetime
import time
import random
import logging
import re
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import mysql.connector
from dateutil import parser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_sentiment_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'stock_sentiment'
}

# Default stocks for fallback/testing
DEFAULT_NIFTY_STOCKS = {
    'RELIANCE': ['Reliance', 'Reliance Industries', 'RIL', 'Mukesh Ambani'],
    'TCS': ['Tata Consultancy Services', 'TCS', 'Tata Consultancy'],
    'HDFCBANK': ['HDFC Bank', 'HDFC'],
    'ICICIBANK': ['ICICI Bank', 'ICICI'],
    'HINDUNILVR': ['Hindustan Unilever', 'HUL', 'Unilever'],
    'INFY': ['Infosys', 'Infy'],
    'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
    'ITC': ['ITC'],
    'KOTAKBANK': ['Kotak Mahindra Bank', 'Kotak Bank', 'Kotak'],
    'LT': ['Larsen & Toubro', 'L&T']
}

class StockDictionaryBuilder:
    """Dynamically builds a dictionary for stock recognition from basic inputs"""
    
    def __init__(self):
        self.name_variants_cache = {}  # Cache for processed companies
        
    def build_dictionary(self, stock_list):
        """
        Build a comprehensive stock recognition dictionary from a list of basic stock info
        
        Parameters:
        stock_list: List of dicts, each with at least 'symbol' and 'name' keys
        
        Returns:
        Dictionary mapping stock symbols to lists of name variants
        """
        stock_dict = {}
        
        for stock in stock_list:
            symbol = stock['symbol']
            name = stock['name']
            
            # Generate name variants
            variants = self._generate_name_variants(symbol, name)
            
            # Add to dictionary
            stock_dict[symbol] = variants
            
        return stock_dict
    
    def _generate_name_variants(self, symbol, name):
        """Generate variants of company names for better matching"""
        # Start with the most basic variants
        variants = [symbol, name]
        
        # Check if we already processed this company
        cache_key = f"{symbol}:{name}"
        if cache_key in self.name_variants_cache:
            return self.name_variants_cache[cache_key]
        
        # Add the symbol itself
        if symbol not in variants:
            variants.append(symbol)
        
        # Add common variations of the name
        name_parts = name.split()
        
        # If multi-word name, add the first word (often the main company name)
        if len(name_parts) > 1 and len(name_parts[0]) > 2:  # Avoid short prefixes like "The"
            variants.append(name_parts[0])
        
        # Add without common suffixes
        for suffix in [" Limited", " Ltd", " Corp", " Corporation", " Inc", " Industries"]:
            if name.endswith(suffix):
                variants.append(name[:-len(suffix)].strip())
        
        # Create acronym from capital letters or first letters of words
        acronym = ''.join(part[0] for part in name_parts if part[0].isupper())
        if len(acronym) > 1:
            variants.append(acronym)
        
        # For banks, add common variations
        if "Bank" in name:
            bank_name = name.replace(" Bank", "").strip()
            variants.append(bank_name)
        
        # For Tata companies, add "Tata" variant
        if name.startswith("Tata "):
            variants.append("Tata")
        
        # Add founder names for well-known companies
        founder_mapping = {
            'RELIANCE': ['Mukesh Ambani', 'Ambani'],
            'ADANIPORTS': ['Gautam Adani', 'Adani'],
            'BAJAJFINSV': ['Rahul Bajaj', 'Bajaj'],
            'BAJFINANCE': ['Rahul Bajaj', 'Bajaj'],
            'WIPRO': ['Azim Premji', 'Premji']
        }
        
        if symbol in founder_mapping:
            variants.extend(founder_mapping[symbol])
        
        # Remove duplicates while preserving order
        unique_variants = []
        for v in variants:
            if v and v not in unique_variants:
                unique_variants.append(v)
        
        # Cache the result
        self.name_variants_cache[cache_key] = unique_variants
        
        return unique_variants
    
    def enhance_with_nlp(self, stock_dict, text_corpus=None):
        """
        Use NLP to discover additional name variants from a corpus of financial text
        
        Parameters:
        stock_dict: Existing stock dictionary to enhance
        text_corpus: List of text samples or path to a corpus file
        
        Returns:
        Enhanced stock dictionary
        """
        if not text_corpus:
            return stock_dict
            
        enhanced_dict = stock_dict.copy()
        
        try:
            # Load spaCy for NLP processing
            import spacy
            nlp = spacy.load("en_core_web_sm")
            
            # Process the corpus
            texts = []
            if isinstance(text_corpus, str):
                # Assume it's a file path
                with open(text_corpus, 'r', encoding='utf-8') as f:
                    texts = f.read().split('\n\n')  # Split by paragraphs
            else:
                # Assume it's a list of texts
                texts = text_corpus
            
            # For each stock symbol, look for potential aliases
            for symbol, variants in stock_dict.items():
                # Get the main company name (usually the second item after the symbol)
                main_name = variants[1] if len(variants) > 1 else variants[0]
                
                # Find potential aliases in the corpus
                potential_aliases = set()
                
                for text in texts:
                    # Check if this text mentions the company
                    if any(variant.lower() in text.lower() for variant in variants):
                        # Process with spaCy
                        doc = nlp(text)
                        
                        # Look for organizational entities that might be aliases
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                # Check if this might be an alias (not already known)
                                entity_text = ent.text.strip()
                                
                                # Simple heuristic: If it shares at least 1 word with known variants
                                words_in_entity = set(entity_text.lower().split())
                                
                                for variant in variants:
                                    words_in_variant = set(variant.lower().split())
                                    
                                    # If there's word overlap but it's not an exact known variant
                                    if (words_in_variant & words_in_entity) and entity_text not in variants:
                                        potential_aliases.add(entity_text)
                
                # Add potential aliases that seem valid
                for alias in potential_aliases:
                    if len(alias) > 3 and alias not in variants:
                        enhanced_dict[symbol].append(alias)
            
            return enhanced_dict
            
        except ImportError:
            logger.warning("spaCy not available for NLP-based dictionary enhancement")
            return stock_dict
        except Exception as e:
            logger.error(f"Error in NLP-based dictionary enhancement: {e}")
            return stock_dict
class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple dimensions"""
    
    def __init__(self, model_name="kdave/FineTuned_Finbert"):
        logger.info(f"Initializing enhanced sentiment analyzer with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            
            # Additional fine-grained models if available
            try:
                from transformers import pipeline as zs_pipeline
                self.confidence_analyzer = zs_pipeline("zero-shot-classification", 
                                                model="facebook/bart-large-mnli",
                                                candidate_labels=["certain", "likely", "speculative", "uncertain"])
                self.impact_analyzer = zs_pipeline("zero-shot-classification", 
                                                model="facebook/bart-large-mnli",
                                                candidate_labels=["major impact", "moderate impact", "minor impact"])
                logger.info("Zero-shot classifiers loaded successfully")
            except Exception as e:
                logger.warning(f"Zero-shot classifiers not loaded: {e}")
                self.confidence_analyzer = None
                self.impact_analyzer = None
                
            logger.info("Enhanced sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing enhanced sentiment analyzer: {e}")
            self.nlp = None
    
    def analyze(self, text, truncate=True):
        """Analyze multiple dimensions of sentiment"""
        if not self.nlp or not text:
            return {
                "basic_sentiment": {"label": "neutral", "score": 0.5},
                "confidence": {"label": "uncertain", "score": 0.25},
                "impact": {"label": "minor impact", "score": 0.3},
                "sentiment_score": 0  # Normalized score for database
            }
        
        try:
            # Truncate long text if needed
            if truncate and len(text) > 512:
                text = text[:512]
            
            # Basic sentiment
            basic_result = self.nlp(text)[0]
            
            # Map sentiment label to numeric score (-1 to 1)
            sentiment_mapping = {
                "positive": 1.0,
                "negative": -1.0,
                "neutral": 0.0
            }
            normalized_score = sentiment_mapping.get(basic_result["label"], 0) * basic_result["score"]
            
            # Additional dimensions (when available)
            confidence_result = {"label": "uncertain", "score": 0.5}
            impact_result = {"label": "minor impact", "score": 0.3}
            
            if self.confidence_analyzer:
                try:
                    confidence_output = self.confidence_analyzer(text)
                    confidence_result = {
                        "label": confidence_output["labels"][0],
                        "score": confidence_output["scores"][0]
                    }
                except Exception as e:
                    logger.error(f"Error in confidence analysis: {e}")
                    
            if self.impact_analyzer:
                try:
                    impact_output = self.impact_analyzer(text)
                    impact_result = {
                        "label": impact_output["labels"][0],
                        "score": impact_output["scores"][0]
                    }
                except Exception as e:
                    logger.error(f"Error in impact analysis: {e}")
            
            return {
                "basic_sentiment": basic_result,
                "confidence": confidence_result,
                "impact": impact_result,
                "sentiment_score": normalized_score  # Normalized score for database
            }
        except Exception as e:
            logger.error(f"Error analyzing enhanced sentiment: {e}")
            return {
                "basic_sentiment": {"label": "neutral", "score": 0.5},
                "confidence": {"label": "uncertain", "score": 0.25},
                "impact": {"label": "minor impact", "score": 0.3},
                "sentiment_score": 0
            }
class DatabaseManager:
    """Class to handle database operations"""
    
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Connect to the database"""
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self.conn = None
            self.cursor = None
    
    def initialize_tables(self):
        """Create database tables if they don't exist"""
        if not self.cursor:
            logger.error("No database connection available")
            return False
            
        try:
            # Articles table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                url VARCHAR(255) UNIQUE NOT NULL,
                content TEXT,
                published_date DATETIME,
                scraped_at DATETIME,
                source VARCHAR(100),
                category VARCHAR(100),
                sentiment_score FLOAT,
                sentiment_label VARCHAR(20),
                confidence_label VARCHAR(20),
                impact_label VARCHAR(20)
            ) ENGINE=InnoDB
            """)
            
            # Stock mentions table (many-to-many relationship)
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_mentions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                article_id INT,
                symbol VARCHAR(20),
                mention_count INT DEFAULT 1,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                UNIQUE KEY unique_article_stock (article_id, symbol)
            ) ENGINE=InnoDB
            """)
            
            # Temporal sentiment table (aggregated daily)
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS temporal_sentiment (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                date DATE,
                daily_sentiment_avg FLOAT,
                sentiment_volume INT,
                rolling_7day_avg FLOAT,
                rolling_30day_avg FLOAT,
                sentiment_momentum FLOAT,
                sentiment_volatility FLOAT,
                UNIQUE KEY unique_symbol_date (symbol, date)
            ) ENGINE=InnoDB
            """)
            
            # Stock dictionary table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_dictionary (
                symbol VARCHAR(20) PRIMARY KEY,
                name_variants TEXT,
                last_updated DATETIME
            ) ENGINE=InnoDB
            """)
            
            self.conn.commit()
            logger.info("Database tables created or verified")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {e}")
            return False
    
    def store_article(self, article_data, target_stocks):
        """Store article and its associated data in the database"""
        if not self.cursor or not self.conn:
            logger.error("No database connection available")
            return False
            
        try:
            # Check if URL already exists
            self.cursor.execute("SELECT id FROM articles WHERE url = %s", (article_data['url'],))
            existing = self.cursor.fetchone()
            if existing:
                logger.info(f"Article already exists: {article_data['url']}")
                return False  # Article already exists
            
            # Parse date
            try:
                published_date = parser.parse(article_data['published_date'])
            except:
                published_date = datetime.datetime.now()
            
            # Insert article
            insert_article_sql = """
            INSERT INTO articles (
                title, url, content, published_date, scraped_at, 
                source, category, sentiment_score, sentiment_label,
                confidence_label, impact_label
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_article_sql, (
                article_data['title'],
                article_data['url'],
                article_data['content'],
                published_date,
                datetime.datetime.now(),
                article_data['source'],
                article_data['category'],
                article_data.get('sentiment_score', 0),
                article_data.get('sentiment', 'neutral'),
                article_data.get('confidence_label', 'uncertain'),
                article_data.get('impact_label', 'minor impact')
            ))
            
            article_id = self.cursor.lastrowid
            
            # Insert stock mentions
            for symbol in article_data['symbols']:
                insert_mention_sql = """
                INSERT INTO stock_mentions (article_id, symbol, mention_count)
                VALUES (%s, %s, %s)
                """
                
                # Count occurrences of this stock in the text
                mention_count = 0
                keywords = target_stocks.get(symbol, [symbol])
                text = article_data['content'].lower()
                
                for keyword in keywords:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    mention_count += len(re.findall(pattern, text, re.IGNORECASE))
                
                # Ensure at least 1 mention
                mention_count = max(1, mention_count)
                
                self.cursor.execute(insert_mention_sql, (article_id, symbol, mention_count))
            
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing article in database: {e}")
            return False
    
    def update_temporal_sentiment(self):
        """Update temporal sentiment aggregations"""
        if not self.cursor or not self.conn:
            logger.error("No database connection available")
            return False
            
        try:
            # Get all stock symbols from mentions
            self.cursor.execute("SELECT DISTINCT symbol FROM stock_mentions")
            symbols = [row[0] for row in self.cursor.fetchall()]
            
            # Current date for calculations
            today = datetime.date.today()
            
            for symbol in symbols:
                logger.info(f"Updating temporal sentiment for {symbol}")
                
                # Get dates and average sentiment for this symbol
                query = """
                SELECT DATE(a.published_date) as article_date, 
                       AVG(a.sentiment_score) as avg_sentiment,
                       COUNT(*) as article_count
                FROM articles a
                JOIN stock_mentions sm ON a.id = sm.article_id
                WHERE sm.symbol = %s
                GROUP BY DATE(a.published_date)
                ORDER BY article_date
                """
                
                self.cursor.execute(query, (symbol,))
                results = self.cursor.fetchall()
                
                if not results:
                    continue
                
                # Process daily sentiment
                for article_date, avg_sentiment, article_count in results:
                    # Check if we already have this date in our temporal table
                    check_query = "SELECT id FROM temporal_sentiment WHERE symbol = %s AND date = %s"
                    self.cursor.execute(check_query, (symbol, article_date))
                    existing = self.cursor.fetchone()
                    
                    if existing:
                        # Update existing record
                        update_query = """
                        UPDATE temporal_sentiment 
                        SET daily_sentiment_avg = %s, sentiment_volume = %s
                        WHERE symbol = %s AND date = %s
                        """
                        self.cursor.execute(update_query, (avg_sentiment, article_count, symbol, article_date))
                    else:
                        # Insert new record
                        insert_query = """
                        INSERT INTO temporal_sentiment 
                        (symbol, date, daily_sentiment_avg, sentiment_volume)
                        VALUES (%s, %s, %s, %s)
                        """
                        self.cursor.execute(insert_query, (symbol, article_date, avg_sentiment, article_count))
                
                # Calculate rolling averages and other metrics
                # This is a simplified version using MySQL's window functions
                rolling_query = """
                UPDATE temporal_sentiment t1
                JOIN (
                    SELECT 
                        id,
                        AVG(daily_sentiment_avg) OVER (
                            PARTITION BY symbol 
                            ORDER BY date 
                            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                        ) as rolling_7day,
                        AVG(daily_sentiment_avg) OVER (
                            PARTITION BY symbol 
                            ORDER BY date 
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                        ) as rolling_30day,
                        STDDEV(daily_sentiment_avg) OVER (
                            PARTITION BY symbol 
                            ORDER BY date 
                            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                        ) as volatility
                    FROM temporal_sentiment
                    WHERE symbol = %s
                ) t2 ON t1.id = t2.id
                SET 
                    t1.rolling_7day_avg = t2.rolling_7day,
                    t1.rolling_30day_avg = t2.rolling_30day,
                    t1.sentiment_volatility = COALESCE(t2.volatility, 0)
                """
                
                self.cursor.execute(rolling_query, (symbol,))
                
                # Calculate momentum (difference between current and previous 7-day avg)
                momentum_query = """
                UPDATE temporal_sentiment t1
                JOIN (
                    SELECT 
                        ts.id,
                        ts.rolling_7day_avg - LAG(ts.rolling_7day_avg, 7) OVER (
                            PARTITION BY ts.symbol ORDER BY ts.date
                        ) as momentum
                    FROM temporal_sentiment ts
                    WHERE ts.symbol = %s
                ) t2 ON t1.id = t2.id
                SET t1.sentiment_momentum = COALESCE(t2.momentum, 0)
                """
                
                self.cursor.execute(momentum_query, (symbol,))
            
            self.conn.commit()
            logger.info("Temporal sentiment data updated successfully")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating temporal sentiment: {e}")
            return False
            
    def store_stock_dictionary(self, stock_dict):
        """Store the stock recognition dictionary in the database"""
        if not self.cursor or not self.conn:
            logger.error("No database connection available")
            return False
            
        try:
            # Store each symbol's variants
            for symbol, variants in stock_dict.items():
                # Join variants with delimiter
                variants_str = "|".join(variants)
                
                # Check if symbol already exists
                self.cursor.execute("SELECT symbol FROM stock_dictionary WHERE symbol = %s", (symbol,))
                existing = self.cursor.fetchone()
                
                if existing:
                    # Update existing record
                    self.cursor.execute(
                        "UPDATE stock_dictionary SET name_variants = %s, last_updated = %s WHERE symbol = %s",
                        (variants_str, datetime.datetime.now(), symbol)
                    )
                else:
                    # Insert new record
                    self.cursor.execute(
                        "INSERT INTO stock_dictionary (symbol, name_variants, last_updated) VALUES (%s, %s, %s)",
                        (symbol, variants_str, datetime.datetime.now())
                    )
            
            self.conn.commit()
            logger.info(f"Stored dictionary for {len(stock_dict)} stocks in database")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing stock dictionary: {e}")
            return False

    def load_stock_dictionary(self):
        """Load the stock recognition dictionary from the database"""
        if not self.cursor:
            logger.error("No database connection available")
            return {}
            
        try:
            self.cursor.execute("SELECT symbol, name_variants FROM stock_dictionary")
            results = self.cursor.fetchall()
            
            stock_dict = {}
            for symbol, variants_str in results:
                variants = variants_str.split("|")
                stock_dict[symbol] = variants
            
            logger.info(f"Loaded dictionary for {len(stock_dict)} stocks from database")
            return stock_dict
        except Exception as e:
            logger.error(f"Error loading stock dictionary: {e}")
            return {}
    
    def close(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connections closed")
class NewsScraperBase:
    """Base class for news scrapers"""
    
    def __init__(self, source_name, base_url):
        self.source_name = source_name
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    
    def _make_request(self, url):
        """Make HTTP request with error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = random.uniform(1, 3) * (attempt + 1)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    return None
    
    def identify_stock_symbols(self, text, target_stocks):
        """Identify which target stocks are mentioned in the text"""
        mentioned_symbols = []
        
        # Create a combined text from title and content for searching
        text = text.lower()
        
        for symbol, keywords in target_stocks.items():
            for keyword in keywords:
                # Match whole words only to avoid partial matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text):
                    mentioned_symbols.append(symbol)
                    break  # Found a match for this symbol, move to next
        
        return mentioned_symbols
    
    def get_latest_news_urls(self):
        """Get URLs of latest news articles (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement get_latest_news_urls()")
    
    def parse_article(self, url):
        """Parse article content (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement parse_article()")

class EconomicTimesScraper(NewsScraperBase):
    """Scraper for Economic Times"""
    
    def __init__(self):
        super().__init__(
            source_name="Economic Times",
            base_url="https://economictimes.indiatimes.com"
        )
    
    def get_latest_news_urls(self):
        """Get latest news article URLs from Economic Times"""
        urls = []
        
        # Pages to scrape
        sections = [
            "/markets/stocks/news",
            "/markets/stocks/recos",
            "/markets/stocks/earnings",
            "/markets/stocks/announcements"
        ]
        
        try:
            for section in sections:
                response = self._make_request(f"{self.base_url}{section}")
                if not response:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                news_items = soup.select('.eachStory, .statsContent')
                
                for item in news_items:
                    link = item.select_one('a')
                    if link and link.has_attr('href'):
                        url = link['href']
                        if not url.startswith('http'):
                            url = self.base_url + url
                        urls.append(url)
                
                # Add delay between section requests
                time.sleep(random.uniform(1, 2))
            
            return list(set(urls))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error getting latest news from {self.source_name}: {e}")
            return []
    
    def parse_article(self, url):
        """Parse Economic Times article content with improved content extraction"""
        try:
            response = self._make_request(url)
            if not response:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.select_one('h1.artTitle')
            if not title:
                title = soup.select_one('.article_title, .title')
            title = title.text.strip() if title else "No title found"
            
            # Extract date
            date_elem = soup.select_one('.publish_on, .date-format')
            published_date = date_elem.text.strip() if date_elem else "Unknown date"
            
            # Improved content extraction - focus on article body
            # First try the main article content
            content_elems = soup.select('.artText p, .Normal')
            
            # If that fails, try alternative selectors
            if not content_elems:
                content_elems = soup.select('.artCont p, .article-content p, .article_content p')
            
            # Filter out ads and irrelevant content
            cleaned_paragraphs = []
            for p in content_elems:
                text = p.text.strip()
                # Skip promotional content
                if "Stock Trading" in text or "By -" in text:
                    continue
                # Skip very short paragraphs that might be ads
                if len(text) < 20:
                    continue
                cleaned_paragraphs.append(text)
            
            content = "\n\n".join(cleaned_paragraphs) if cleaned_paragraphs else "No content found"
            
            # Check if we have meaningful content
            if content == "No content found" or len(content) < 100:
                # Try one more selector
                article_div = soup.select_one('.artText, .article')
                if article_div:
                    # Get all text directly
                    raw_text = article_div.get_text()
                    # Split by newlines and filter
                    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                    # Filter out short lines and promotional content
                    filtered_lines = [line for line in lines if len(line) > 30 and "Stock Trading" not in line and "By -" not in line]
                    if filtered_lines:
                        content = "\n\n".join(filtered_lines)
            
            # Extract category
            # Extract category
            category_elem = soup.select_one('.breadcrumb a:nth-child(2)')
            category = category_elem.text.strip() if category_elem else "Markets"
            
            return {
                'title': title,
                'url': url,
                'content': content,
                'published_date': published_date,
                'source': self.source_name,
                'category': category,
                'symbols': []  # Will be filled in by the calling method
            }
        except Exception as e:
            logger.error(f"Error parsing article {url}: {e}")
            return None


class MoneycontrolScraper(NewsScraperBase):
    """Scraper for Moneycontrol"""
    
    def __init__(self):
        super().__init__(
            source_name="Moneycontrol",
            base_url="https://www.moneycontrol.com"
        )
    
    def get_latest_news_urls(self):
        """Get latest news article URLs from Moneycontrol"""
        urls = []
        
        # Pages to scrape
        sections = [
            "/news/business",
            "/news/market",
            "/stocksmarketsnews-243.html",
            "/news/economy",
            "/news/trends/market-trends"
        ]
        
        try:
            for section in sections:
                full_url = self.base_url + section
                response = self._make_request(full_url)
                if not response:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Different selectors for different page layouts
                news_items = soup.select('.clearfix li h2 a, .common-article a.article_title, .mid-contener-row a.arial11')
                
                for item in news_items:
                    if item and item.has_attr('href'):
                        url = item['href']
                        if not url.startswith('http'):
                            url = self.base_url + url
                        urls.append(url)
                
                # Add delay between section requests
                time.sleep(random.uniform(1, 2))
            
            return list(set(urls))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error getting latest news from {self.source_name}: {e}")
            return []
    
    def parse_article(self, url):
        """Parse Moneycontrol article content with improved content extraction"""
        try:
            response = self._make_request(url)
            if not response:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.select_one('h1.article_title, .artTitle')
            title = title.text.strip() if title else "No title found"
            
            # Extract date
            date_elem = soup.select_one('.article_schedule, .article_schedule span')
            published_date = date_elem.text.strip() if date_elem else "Unknown date"
            
            # Improved content extraction - focus on article body
            content_elems = soup.select('.content_wrapper p, .arti-flow p, .article-content p')
            
            # Filter out ads and irrelevant content
            cleaned_paragraphs = []
            for p in content_elems:
                text = p.text.strip()
                # Skip promotional content
                if "Stock Trading" in text or "By -" in text:
                    continue
                # Skip very short paragraphs that might be ads
                if len(text) < 20:
                    continue
                cleaned_paragraphs.append(text)
            
            content = "\n\n".join(cleaned_paragraphs) if cleaned_paragraphs else "No content found"
            
            # Check if we have meaningful content
            if content == "No content found" or len(content) < 100:
                # Try one more selector
                article_div = soup.select_one('.content_wrapper, .article-content')
                if article_div:
                    # Get all text directly
                    raw_text = article_div.get_text()
                    # Split by newlines and filter
                    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                    # Filter out short lines and promotional content
                    filtered_lines = [line for line in lines if len(line) > 30 and "Stock Trading" not in line and "By -" not in line]
                    if filtered_lines:
                        content = "\n\n".join(filtered_lines)
            
            # Extract category
            category = "Business"  # Default category
            breadcrumb = soup.select('.breadcrumb a')
            if breadcrumb and len(breadcrumb) > 1:
                category = breadcrumb[1].text.strip()
            
            return {
                'title': title,
                'url': url,
                'content': content,
                'published_date': published_date,
                'source': self.source_name,
                'category': category,
                'symbols': []  # Will be filled in by the calling method
            }
        except Exception as e:
            logger.error(f"Error parsing article {url}: {e}")
            return None


class StockNewsScraper:
    """Main class to coordinate the scraping process"""
    
    def __init__(self, db_config=None, stock_list=None):
        # Initialize database manager
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.initialize_tables()
        
        # Set up stock dictionary
        self.target_stocks = self._initialize_stock_dictionary(stock_list)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
        # List of scrapers
        self.scrapers = []
    
    def _initialize_stock_dictionary(self, stock_list):
        """Initialize the stock dictionary from database or build it from stock_list"""
        # First try to load from database
        stock_dict = self.db_manager.load_stock_dictionary()
        
        # If no dictionary in database and stock_list provided, build new one
        if not stock_dict and stock_list:
            logger.info(f"Building new stock dictionary from {len(stock_list)} stocks")
            dictionary_builder = StockDictionaryBuilder()
            stock_dict = dictionary_builder.build_dictionary(stock_list)
            
            # Store in database for future use
            self.db_manager.store_stock_dictionary(stock_dict)
        
        # If still no dictionary, use default
        if not stock_dict:
            logger.info("Using default stock dictionary")
            stock_dict = DEFAULT_NIFTY_STOCKS
        
        return stock_dict
    
    def update_stock_dictionary(self, new_mappings):
        """
        Update the stock dictionary with new discovered mappings
        
        Parameters:
        new_mappings: Dict where keys are stock symbols and values are lists of new name variants
        """
        updated = False
        
        for symbol, new_variants in new_mappings.items():
            if symbol in self.target_stocks:
                # Add only new variants
                for variant in new_variants:
                    if variant not in self.target_stocks[symbol]:
                        self.target_stocks[symbol].append(variant)
                        updated = True
                        logger.info(f"Added new variant '{variant}' for symbol {symbol}")
        
        # If updated, store back to database
        if updated:
            self.db_manager.store_stock_dictionary(self.target_stocks)
    
    def add_stock(self, symbol, name):
        """
        Add a new stock to the dictionary
        
        Parameters:
        symbol: Stock symbol (e.g., 'RELIANCE')
        name: Company name (e.g., 'Reliance Industries Limited')
        """
        if symbol in self.target_stocks:
            logger.info(f"Stock {symbol} already exists in dictionary")
            return False
        
        # Generate variants for this stock
        dictionary_builder = StockDictionaryBuilder()
        variants = dictionary_builder._generate_name_variants(symbol, name)
        
        # Add to dictionary
        self.target_stocks[symbol] = variants
        
        # Store updated dictionary
        self.db_manager.store_stock_dictionary(self.target_stocks)
        logger.info(f"Added new stock {symbol} ({name}) to dictionary")
        
        return True
    
    def add_scraper(self, scraper):
        """Add a scraper to the manager"""
        self.scrapers.append(scraper)
    
    def run(self, limit_per_source=50, min_symbols=1):
        """Run all registered scrapers and process the articles"""
        total_processed = 0
        total_saved = 0
        
        logger.info(f"Starting scraping run with {len(self.scrapers)} scrapers")
        start_time = time.time()
        
        try:
            for scraper in self.scrapers:
                processed, saved = self._run_scraper(scraper, limit_per_source, min_symbols)
                total_processed += processed
                total_saved += saved
            
            # Update temporal sentiment data
            if total_saved > 0:
                self.db_manager.update_temporal_sentiment()
                
            elapsed_time = time.time() - start_time
            logger.info(f"Completed scraping run in {elapsed_time:.2f} seconds. Processed: {total_processed}, Saved: {total_saved}")
            
            return total_processed, total_saved
        except Exception as e:
            logger.error(f"Error in scraping run: {e}")
            return total_processed, total_saved
    
    def _run_scraper(self, scraper, limit, min_symbols):
        """Run a single scraper and process its articles"""
        processed = 0
        saved = 0
        
        logger.info(f"Starting {scraper.source_name} scraper")
        
        try:
            # Get URLs of latest news articles
            urls = scraper.get_latest_news_urls()
            
            if not urls:
                logger.warning(f"No URLs found for {scraper.source_name}")
                return processed, saved
            
            # Process each URL
            for url in urls[:limit]:  # Limit the number of articles to process
                processed += 1
                logger.info(f"Processing article {processed}/{min(limit, len(urls))}: {url}")
                
                # Parse the article
                article_data = scraper.parse_article(url)
                
                if not article_data:
                    logger.warning(f"Failed to parse article: {url}")
                    continue
                
                # Identify mentioned stock symbols
                mentioned_symbols = scraper.identify_stock_symbols(
                    article_data['title'] + " " + article_data['content'],
                    self.target_stocks
                )
                article_data['symbols'] = mentioned_symbols
                
                # Skip articles that don't mention enough target stocks
                if len(mentioned_symbols) < min_symbols:
                    logger.info(f"Skipping article with insufficient stock mentions: {article_data['title']}")
                    continue
                
                # Analyze sentiment
                if self.sentiment_analyzer:
                    sentiment_results = self.sentiment_analyzer.analyze(
                        article_data['title'] + " " + article_data['content']
                    )
                    
                    article_data['sentiment'] = sentiment_results['basic_sentiment']['label']
                    article_data['sentiment_score'] = sentiment_results['sentiment_score']
                    article_data['confidence_label'] = sentiment_results['confidence']['label']
                    article_data['impact_label'] = sentiment_results['impact']['label']
                
                # Store in database
                success = self.db_manager.store_article(article_data, self.target_stocks)
                if success:
                    saved += 1
                    logger.info(f"Saved article about {mentioned_symbols}: {article_data['title']}")
                
                # Be polite with delays between requests
                time.sleep(random.uniform(1, 3))
            
            return processed, saved
        
        except Exception as e:
            logger.error(f"Error running scraper {scraper.source_name}: {e}")
            return processed, saved
    
    def get_sentiment_summary(self):
        """Get a summary of sentiment data from the database"""
        if not self.db_manager.cursor:
            logger.error("No database connection available")
            return None
        
        try:
            # Query for overall sentiment statistics by stock
            query = """
            SELECT 
                sm.symbol,
                COUNT(DISTINCT a.id) as article_count,
                AVG(a.sentiment_score) as avg_sentiment,
                SUM(CASE WHEN a.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN a.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN a.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM stock_mentions sm
            JOIN articles a ON sm.article_id = a.id
            WHERE a.published_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY sm.symbol
            ORDER BY article_count DESC, avg_sentiment DESC
            """
            
            self.db_manager.cursor.execute(query)
            columns = [column[0] for column in self.db_manager.cursor.description]
            results = [dict(zip(columns, row)) for row in self.db_manager.cursor.fetchall()]
            
            return results
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return None
    
    def get_temporal_sentiment(self, symbol, days=30):
        """Get temporal sentiment data for a specific stock"""
        if not self.db_manager.cursor:
            logger.error("No database connection available")
            return None
        
        try:
            query = """
            SELECT 
                date,
                daily_sentiment_avg,
                rolling_7day_avg,
                sentiment_volume,
                sentiment_momentum
            FROM temporal_sentiment
            WHERE symbol = %s
            AND date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY date
            """
            
            self.db_manager.cursor.execute(query, (symbol, days))
            columns = [column[0] for column in self.db_manager.cursor.description]
            results = [dict(zip(columns, row)) for row in self.db_manager.cursor.fetchall()]
            
            return results
        except Exception as e:
            logger.error(f"Error getting temporal sentiment for {symbol}: {e}")
            return None
    
    def close(self):
        """Close database connections"""
        self.db_manager.close()


# Main execution
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'your_password',
        'database': 'stock_sentiment'
    }
    
    # Example stock list - in production, this could be loaded from a file or API
    # If not provided, system will try to load from database or use defaults
    stocks = [
        {'symbol': 'RELIANCE', 'name': 'Reliance Industries Limited'},
        {'symbol': 'TCS', 'name': 'Tata Consultancy Services Limited'},
        {'symbol': 'HDFCBANK', 'name': 'HDFC Bank Limited'},
        {'symbol': 'ICICIBANK', 'name': 'ICICI Bank Limited'},
        {'symbol': 'HINDUNILVR', 'name': 'Hindustan Unilever Limited'},
        {'symbol': 'INFY', 'name': 'Infosys Limited'},
        {'symbol': 'BHARTIARTL', 'name': 'Bharti Airtel Limited'},
        {'symbol': 'ITC', 'name': 'ITC Limited'},
        {'symbol': 'KOTAKBANK', 'name': 'Kotak Mahindra Bank Limited'},
        {'symbol': 'LT', 'name': 'Larsen & Toubro Limited'}
    ]
    
    # Create scraper with dynamic stock list
    scraper = StockNewsScraper(db_config=db_config, stock_list=stocks)
    
    # Add scrapers for different sources
    scraper.add_scraper(EconomicTimesScraper())
    scraper.add_scraper(MoneycontrolScraper())
    
    try:
        # Run scrapers
        processed, saved = scraper.run(limit_per_source=50, min_symbols=1)
        
        # Display summary of results
        print(f"\nScraping Complete: Processed {processed} articles, saved {saved} relevant articles")
        
        # Get and display sentiment summary
        print("\nSentiment Summary (Last 30 Days):")
        summary = scraper.get_sentiment_summary()
        if summary:
            for stock in summary[:10]:  # Display top 10
                sentiment = stock['avg_sentiment']
                sentiment_str = "POSITIVE" if sentiment > 0.2 else "NEGATIVE" if sentiment < -0.2 else "NEUTRAL"
                print(f"{stock['symbol']}: {sentiment_str} ({sentiment:.2f}) - {stock['article_count']} articles " +
                      f"(+: {stock['positive_count']}, -: {stock['negative_count']}, =: {stock['neutral_count']})")
        
        # Get temporal sentiment for a few top stocks
        top_stocks = [stock['symbol'] for stock in summary[:3]] if summary else ['RELIANCE', 'TCS', 'HDFCBANK']
        
        for symbol in top_stocks:
            print(f"\nTemporal Sentiment for {symbol} (Last 30 Days):")
            data = scraper.get_temporal_sentiment(symbol)
            if data:
                for day in data[-7:]:  # Last 7 days
                    print(f"  {day['date']}: Daily: {day['daily_sentiment_avg']:.2f}, " +
                          f"7-Day Avg: {day['rolling_7day_avg']:.2f}, " +
                          f"Momentum: {day['sentiment_momentum']:.2f}, " +
                          f"Volume: {day['sentiment_volume']}")
    
    finally:
        # Close connections
        scraper.close()