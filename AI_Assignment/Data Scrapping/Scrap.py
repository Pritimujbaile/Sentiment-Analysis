import pandas as pd
import requests
from bs4 import BeautifulSoup
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util import Retry
import certifi
import time
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
import re
 
# Ask user for the Amazon product URL
product_url = input("Enter the Amazon product URL: ")
 
# Extract ASIN from the URL
asin_match = re.search(r'/dp/([A-Z0-9]{10})', product_url)
if asin_match:
    asin = asin_match.group(1)
else:
    print("Invalid URL. Could not extract ASIN.")
    sys.exit(1)
 
# Construct the reviews URL
url = product_url 

 
# Output directory
out_dir = r"C:\Users\priti.mujbaile\Desktop\AI_Assignment\Data Scrapping"
os.makedirs(out_dir, exist_ok=True)
 
# Path to cookies file (exported from browser)
cookies_file = os.path.join(out_dir, "amazon_cookies.json")
 
# Headers for requests
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36',
    'Accept-Language': 'en-IN,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
    'referer': 'https://www.amazon.in/'
}
 
# Create a custom adapter for legacy TLS
class LegacyTLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        ctx = ssl.create_default_context()
        if hasattr(ssl, "OP_LEGACY_SERVER_CONNECT"):
            ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, ssl_context=ctx)
 
# Create session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(502, 503, 504))
session.mount("https://", LegacyTLSAdapter(max_retries=retries))
session.headers.update(header)
 
print("Session created")
 
# Use Selenium to render the page and get HTML (to handle JS-loaded content)
debug_path = os.path.join(out_dir, "debug_amazon_response_rendered.html")
 
opts = Options()
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-blink-features=AutomationControlled")
opts.add_argument("--disable-extensions")
opts.add_argument("--disable-plugins")
opts.add_argument("--disable-images")  # Speed up loading
#opts.add_argument("--disable-javascript")  # Wait, no, we need JS for reviews
# Remove headless to see if it helps
opts.headless = False
# Add user agent to match browser
opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
 
driver = webdriver.Chrome(options=opts)
 
# Load cookies if file exists
if os.path.exists(cookies_file):
    with open(cookies_file, 'r') as f:
        cookies = json.load(f)
    driver.get("https://www.amazon.in")  # Navigate to domain first
    for cookie in cookies:
        # Filter to Selenium-compatible keys
        selenium_cookie = {
            'name': cookie.get('name'),
            'value': cookie.get('value'),
            'domain': cookie.get('domain'),
            'path': cookie.get('path', '/'),
            'secure': cookie.get('secure', False),
            'httpOnly': cookie.get('httpOnly', False),
        }
        if 'expiry' in cookie:
            selenium_cookie['expiry'] = cookie['expiry']
        try:
            driver.add_cookie(selenium_cookie)
            print(f"Added cookie: {cookie['name']}")
        except Exception as e:
            print(f"Error adding cookie {cookie['name']}: {e}")
    print("Cookies loaded from file")
else:
    print("No cookies file found. Proceeding without cookies.")
 
# Scrape multiple pages
all_reviews = []
page = 1
max_reviews = 70
 
while len(all_reviews) < max_reviews:
    current_url = f"{url}&pageNumber={page}"
    print(f"Fetching page {page}: {current_url}")
   
    driver.get(current_url)
    # Wait for JS to load reviews
    time.sleep(5)
   
    html = driver.page_source
   
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
   
    # Extract reviews
    review_nodes = soup.select('div[data-hook="review"], li[data-hook="review"], div.a-section.review, #cm_cr-review_list div[data-hook="review"]')
    if not review_nodes:
        review_nodes = soup.select('div[data-asin] div[data-hook="review"]')
   
    page_reviews = []
    for node in review_nodes:
        title_node = node.select_one('a[data-hook="review-title"], span[data-hook="review-title"]')
        rating_node = node.select_one('i[data-hook="review-star-rating"] span, span.a-icon-alt')
        author_node = node.select_one('span.a-profile-name')
        date_node = node.select_one('span[data-hook="review-date"]')
        body_node = node.select_one('span[data-hook="review-body"]')
       
        page_reviews.append({
            "title": title_node.get_text(strip=True) if title_node else "",
            "rating": rating_node.get_text(strip=True).split()[0] if rating_node else "",
            "author": author_node.get_text(strip=True) if author_node else "",
            "date": date_node.get_text(strip=True) if date_node else "",
            "body": body_node.get_text(" ", strip=True) if body_node else ""
        })
   
    all_reviews.extend(page_reviews)
    print(f"Found {len(page_reviews)} reviews on page {page}")
   
    # Check for next page
    next_link = soup.select_one('li.a-last a, .a-pagination .a-last a')
    if not next_link or page >= 10:  # Limit to 10 pages to avoid infinite loop
        break
    page += 1
 
# Save cookies after scraping
cookies = driver.get_cookies()
with open(cookies_file, 'w') as f:
    json.dump(cookies, f)
print(f"Cookies saved to {cookies_file}")
 
driver.quit()
 
# Save all reviews
if all_reviews:
    df = pd.DataFrame(all_reviews[:max_reviews])  # Limit to 70
    out_path = os.path.join(out_dir, "raw_reviews.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} reviews to {out_path}")
else:
    print("No reviews found.")