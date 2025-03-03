import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import datetime
import time
import random
import csv
import logging
import re
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define top 20 Nifty 50 stocks (symbols and company names for matching)
TOP_NIFTY_STOCKS = {
    'RELIANCE': ['Reliance', 'Reliance Industries', 'RIL', 'Mukesh Ambani'],
    'TCS': ['Tata Consultancy Services', 'TCS', 'Tata Consultancy'],
    'HDFCBANK': ['HDFC Bank', 'HDFC'],
    'ICICIBANK': ['ICICI Bank', 'ICICI'],
    'HINDUNILVR': ['Hindustan Unilever', 'HUL', 'Unilever'],
    'INFY': ['Infosys', 'Infy'],
    'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
    'ITC': ['ITC'],
    'KOTAKBANK': ['Kotak Mahindra Bank', 'Kotak Bank', 'Kotak'],
    'LT': ['Larsen & Toubro', 'L&T'],
    'SBIN': ['State Bank of India', 'SBI'],
    'BAJFINANCE': ['Bajaj Finance'],
    'ASIANPAINT': ['Asian Paints'],
    'AXISBANK': ['Axis Bank', 'Axis'],
    'MARUTI': ['Maruti Suzuki', 'Maruti'],
    'HCLTECH': ['HCL Technologies', 'HCL Tech', 'HCL'],
    'ULTRACEMCO': ['UltraTech Cement', 'UltraTech'],
    'SUNPHARMA': ['Sun Pharmaceutical', 'Sun Pharma'],
    'TATAMOTORS': ['Tata Motors'],
    'TITAN': ['Titan Company', 'Titan']
}

class SentimentAnalyzer:
    """Class to analyze sentiment of text using transformers"""
    
    def __init__(self, model_name="ProsusAI/finbert"):
        logger.info(f"Initializing sentiment analyzer with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.nlp = None
    
    def analyze(self, text, truncate=True):
        """Analyze sentiment of text"""
        if not self.nlp or not text:
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate long text if needed
            if truncate and len(text) > 512:
                text = text[:512]
                
            result = self.nlp(text)[0]
            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "neutral", "score": 0.5}

class NewsScraperBase:
    """Base class for news scrapers"""
    
    def __init__(self, source_name, base_url, csv_dir="data", target_stocks=None):
        self.source_name = source_name
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        self.csv_dir = csv_dir
        self.target_stocks = target_stocks or TOP_NIFTY_STOCKS
        self._init_storage()
        
    def _init_storage(self):
        """Initialize CSV storage"""
        # Create directory if it doesn't exist
        Path(self.csv_dir).mkdir(parents=True, exist_ok=True)
        
        # Create source-specific CSV file if it doesn't exist
        self.csv_file = os.path.join(self.csv_dir, f"{self.source_name.replace(' ', '_').lower()}_articles.csv")
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['title', 'url', 'content', 'published_date', 'source', 'category', 
                                 'scraped_at', 'symbols', 'sentiment', 'sentiment_score'])
        
        # Keep track of scraped URLs to avoid duplicates
        self.existing_urls = set()
        if os.path.exists(self.csv_file):
            try:
                df = pd.read_csv(self.csv_file)
                if 'url' in df.columns:
                    self.existing_urls = set(df['url'].tolist())
            except Exception as e:
                logger.error(f"Error reading existing URLs: {e}")
        
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
    
    def _save_to_csv(self, article_data):
        """Save article data to CSV file"""
        try:
            # Skip if URL already exists
            if article_data['url'] in self.existing_urls:
                logger.info(f"Article already exists: {article_data['url']}")
                return False
                
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    article_data['title'],
                    article_data['url'],
                    article_data['content'],
                    article_data['published_date'],
                    article_data['source'],
                    article_data['category'],
                    datetime.datetime.now().isoformat(),
                    ','.join(article_data['symbols']),
                    article_data.get('sentiment', 'neutral'),
                    article_data.get('sentiment_score', 0.5)
                ])
            
            # Add to existing URLs set
            self.existing_urls.add(article_data['url'])
            return True
        except Exception as e:
            logger.error(f"CSV error: {e}")
            return False
    
    def identify_stock_symbols(self, text):
        """Identify which target stocks are mentioned in the text"""
        mentioned_symbols = []
        
        # Create a combined text from title and content for searching
        text = text.lower()
        
        for symbol, keywords in self.target_stocks.items():
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
    
    def run(self, limit=10, min_symbols=1, sentiment_analyzer=None):
        """Run the scraper to collect articles"""
        logger.info(f"Starting {self.source_name} scraper, targeting {len(self.target_stocks)} stocks")
        urls = self.get_latest_news_urls()
        
        if not urls:
            logger.warning(f"No URLs found for {self.source_name}")
            return
            
        count = 0
        total_processed = 0
        
        for url in urls:
            if count >= limit:
                break
                
            total_processed += 1
            logger.info(f"Processing article: {url} ({total_processed}/{len(urls)})")
            
            # Skip if URL already exists
            if url in self.existing_urls:
                logger.info(f"Article already exists: {url}")
                continue
                
            article_data = self.parse_article(url)
            
            if article_data:
                # Identify mentioned stock symbols
                mentioned_symbols = self.identify_stock_symbols(
                    article_data['title'] + " " + article_data['content']
                )
                article_data['symbols'] = mentioned_symbols
                
                # Only process if article mentions at least the minimum number of target stocks
                if len(mentioned_symbols) >= min_symbols:
                    # Analyze sentiment if analyzer is provided
                    if sentiment_analyzer:
                        sentiment_result = sentiment_analyzer.analyze(
                            article_data['title'] + " " + article_data['content']
                        )
                        article_data['sentiment'] = sentiment_result['label']
                        article_data['sentiment_score'] = sentiment_result['score']
                        
                    success = self._save_to_csv(article_data)
                    if success:
                        logger.info(f"Saved article about {mentioned_symbols}: {article_data['title']}")
                        count += 1
                else:
                    logger.info(f"Skipping article with insufficient stock mentions: {article_data['title']}")
                    
            # Be polite with delays between requests
            time.sleep(random.uniform(1, 3))
            
        logger.info(f"Completed {self.source_name} scraper. Saved {count} relevant articles out of {total_processed} processed")


class EconomicTimesScraper(NewsScraperBase):
    """Scraper for Economic Times"""
    
    def __init__(self, csv_dir="data", target_stocks=None):
        super().__init__(
            source_name="Economic Times",
            base_url="https://economictimes.indiatimes.com",
            csv_dir=csv_dir,
            target_stocks=target_stocks
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
    
    def __init__(self, csv_dir="data", target_stocks=None):
        super().__init__(
            source_name="Moneycontrol",
            base_url="https://www.moneycontrol.com",
            csv_dir=csv_dir,
            target_stocks=target_stocks
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


class NewsScraperManager:
    """Manager class to run multiple scrapers"""
    
    def __init__(self, csv_dir="data", target_stocks=None, use_sentiment=True):
        self.csv_dir = csv_dir
        self.scrapers = []
        self.target_stocks = target_stocks or TOP_NIFTY_STOCKS
        self.use_sentiment = use_sentiment
        self.sentiment_analyzer = SentimentAnalyzer() if use_sentiment else None
        Path(self.csv_dir).mkdir(parents=True, exist_ok=True)
    
    def add_scraper(self, scraper):
        """Add a scraper to the manager"""
        self.scrapers.append(scraper)
    
    def run_all(self, limit_per_source=20, min_symbols=1):
        """Run all registered scrapers"""
        logger.info(f"Starting scraping run with {len(self.scrapers)} scrapers, targeting {len(self.target_stocks)} stocks")
        start_time = time.time()
        
        for scraper in self.scrapers:
            try:
                scraper.run(
                    limit=limit_per_source, 
                    min_symbols=min_symbols,
                    sentiment_analyzer=self.sentiment_analyzer
                )
            except Exception as e:
                logger.error(f"Error running scraper {scraper.source_name}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed scraping run in {elapsed_time:.2f} seconds")

    def combine_all_data(self, output_file="all_stock_news.csv"):
        """Combine all source-specific CSVs into one master CSV"""
        try:
            all_data = []
            for scraper in self.scrapers:
                csv_file = os.path.join(self.csv_dir, f"{scraper.source_name.replace(' ', '_').lower()}_articles.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                output_path = os.path.join(self.csv_dir, output_file)
                combined_df.to_csv(output_path, index=False)
                logger.info(f"Combined data exported to {output_path} ({len(combined_df)} articles)")
                
                # Generate summary of collected articles by stock symbol
                self._generate_summary(combined_df)
                
                return True
            else:
                logger.warning("No data found to combine")
                return False
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            return False
    
    def _generate_summary(self, df):
        """Generate summary statistics about collected articles"""
        try:
            # Summary by stock symbol
            logger.info("Articles collected by stock symbol:")
            symbol_counts = {}
            
            for _, row in df.iterrows():
                if pd.notna(row['symbols']) and row['symbols']:
                    symbols = str(row['symbols']).split(',')
                    for symbol in symbols:
                        if symbol and symbol.strip():
                            symbol = symbol.strip()
                            if symbol not in symbol_counts:
                                symbol_counts[symbol] = {
                                    'count': 0,
                                    'positive': 0,
                                    'negative': 0,
                                    'neutral': 0
                                }
                            symbol_counts[symbol]['count'] += 1
                            
                            # Count by sentiment
                            sentiment = str(row.get('sentiment', 'neutral')).lower()
                            if sentiment == 'positive':
                                symbol_counts[symbol]['positive'] += 1
                            elif sentiment == 'negative':
                                symbol_counts[symbol]['negative'] += 1
                            else:
                                symbol_counts[symbol]['neutral'] += 1
            
            # Print summary
            for symbol, stats in sorted(symbol_counts.items(), key=lambda x: x[1]['count'], reverse=True):
                logger.info(f"  {symbol}: {stats['count']} articles (Positive: {stats['positive']}, Negative: {stats['negative']}, Neutral: {stats['neutral']})")
            
            # Create summary CSV
            summary_file = os.path.join(self.csv_dir, "stock_news_summary.csv")
            with open(summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['symbol', 'total_articles', 'positive', 'negative', 'neutral', 'sentiment_ratio'])
                
                for symbol, stats in sorted(symbol_counts.items(), key=lambda x: x[1]['count'], reverse=True):
                    # Calculate sentiment ratio (positive - negative) / total
                    total = stats['count']
                    sentiment_ratio = 0
                    if total > 0:
                        sentiment_ratio = (stats['positive'] - stats['negative']) / total
                    
                    writer.writerow([
                        symbol, 
                        stats['count'], 
                        stats['positive'], 
                        stats['negative'], 
                        stats['neutral'],
                        f"{sentiment_ratio:.2f}"
                    ])
            
            logger.info(f"Summary statistics exported to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


# Main execution
if __name__ == "__main__":
    # Create data directory
    data_dir = "stock_news_data"
    
    # Create scraper manager with top Nifty stocks
    manager = NewsScraperManager(csv_dir=data_dir, target_stocks=TOP_NIFTY_STOCKS, use_sentiment=True)
    
    # Add scrapers
    manager.add_scraper(EconomicTimesScraper(csv_dir=data_dir, target_stocks=TOP_NIFTY_STOCKS))
    manager.add_scraper(MoneycontrolScraper(csv_dir=data_dir, target_stocks=TOP_NIFTY_STOCKS))
    
    # Run all scrapers (collect up to 50 articles per source that mention at least 1 target stock)
    manager.run_all(limit_per_source=50, min_symbols=1)
    
    # Combine all data
    manager.combine_all_data()