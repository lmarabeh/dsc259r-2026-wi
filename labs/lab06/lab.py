# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(html):
    soup = BeautifulSoup(html, "lxml")
    book_links = []
    
    articles = soup.find_all("article", class_="product_pod")
    
    for article in articles:
        star_div = article.find("p", class_="star-rating")
        rating_classes = star_div.get("class", [])
        
        if "Four" not in rating_classes and "Five" not in rating_classes:
            continue
            
        price_p = article.find("p", class_="price_color")
        if price_p:
            price_str = price_p.get_text().strip()
            price_match = re.search(r"\d+\.\d+", price_str)
            if price_match:
                price = float(price_match.group())
                if price >= 50.0:
                    continue
            else:
                continue
        
        h3 = article.find("h3")
        a_tag = h3.find("a")
        if a_tag:
            book_links.append(a_tag["href"])
            
    return book_links


def get_product_info(html, categories):
    soup = BeautifulSoup(html, "lxml")
    
    # Check Category via Breadcrumbs
    breadcrumb = soup.find("ul", class_="breadcrumb")
    if not breadcrumb:
        return None
        
    links = breadcrumb.find_all("a")
    # Expected structure: Home > Books > Category > Title
    if len(links) < 3:
        return None
        
    category = links[2].get_text().strip()
    
    if category not in categories:
        return None

    # Extract Data
    data = {}
    data["Category"] = category
    
    h1 = soup.find("h1")
    data["Title"] = h1.get_text().strip() if h1 else None
    
    # Product Information Table
    table = soup.find("table", class_="table-striped")
    if table:
        rows = table.find_all("tr")
        for row in rows:
            header = row.find("th").get_text().strip()
            value = row.find("td").get_text().strip()
            data[header] = value
            
    # Rating
    star_p = soup.find("p", class_="star-rating")
    if star_p:
        classes = star_p.get("class", [])
        rating_val = [c for c in classes if c != "star-rating"]
        data["Rating"] = rating_val[0] if rating_val else None
        
    # Description
    desc_header = soup.find("div", id="product_description")
    if desc_header:
        desc_p = desc_header.find_next_sibling("p")
        data["Description"] = desc_p.get_text().strip() if desc_p else None
    else:
        data["Description"] = None

    return data


def scrape_books(k, categories):
    """
    Scrapes first k pages, returning DataFrame of matching books.
    """
    all_books_data = []
    
    # Use a session or headers to mimic a browser (good practice)
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Base URL for catalogue pages
    base_catalogue_url = "http://books.toscrape.com/catalogue/"
    
    for i in range(1, k + 1):
        catalogue_url = f"http://books.toscrape.com/catalogue/page-{i}.html"
        response = requests.get(catalogue_url, headers=headers)
        
        if response.status_code != 200:
            continue
            
        links = extract_book_links(response.text)
        
        for link in links:
            book_url = urljoin(base_catalogue_url, link)
            
            book_response = requests.get(book_url, headers=headers)
            
            if book_response.status_code == 200:
                book_info = get_product_info(book_response.text, categories)
                if book_info:
                    all_books_data.append(book_info)
                    
    # Create DataFrame
    df = pd.DataFrame(all_books_data)
    return df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def get_comments(storyid):
    # Base URL template
    url_template = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
    
    # List to store the processed comment dictionaries
    comments_data = []
    
    # Helper function for Depth-First Search
    def process_ids(ids):
        for item_id in ids:
            response = requests.get(url_template.format(item_id))
            if response.status_code != 200:
                continue
                
            item = response.json()
            
            # Check for "dead" or "deleted" status
            if item.get('dead') or item.get('deleted'):
                continue
            
            # Extract relevant data
            comment_info = {
                'id': item.get('id'),
                'by': item.get('by'),
                'text': item.get('text'),
                'parent': item.get('parent'),
                'time': pd.to_datetime(item.get('time'), unit='s')  # Convert Unix timestamp
            }
            
            comments_data.append(comment_info)
            
            # Recursion: If this comment has kids (replies), process them immediately
            if 'kids' in item:
                process_ids(item['kids'])

    # Main Execution
    story_response = requests.get(url_template.format(storyid))
    story_json = story_response.json()
    
    # Start DFS if there are comments
    process_ids(story_json['kids'])
        
    # Create DataFrame with specific columns
    df = pd.DataFrame(comments_data, columns=['id', 'by', 'text', 'parent', 'time'])
    
    return df
