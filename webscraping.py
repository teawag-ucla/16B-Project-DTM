import fitz
import pandas as pd
from pathlib import Path
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import requests
import pandas as pd

#news functions
def news_webscrapper(keyword, url, article_tag, article_class, headline_tag, headline_class, byline_tag, byline_class):
    """
    Scrapes news articles where the url's are desgined in a way where the only changes per page is the page number and keyword.

    Paramters:
    keyword (str): keyword being explored
    url (str): the link the website
    article_tag (str): the larger part of where the headline and byline are (have to inspect and will see the class and tag)
    article _class (str): the larger part of where the headline and byline are
    headline_tag (str): the part the headline is enclosed in
    headline_class (str): the part the headline is enclosed in
    byline_tag (str): the part the byline is enclosed in
    byline_class (str): the part the byline is enclosed in

    Returns:
    pd.DataFrame: DataFrame with columns "Headline", "Byline"
    """
    headlines = []
    bylines = []

    for page in range(1, 5):  # adjust the range to get more pages / make a checker to see if they have enought pages
        updated_url = url + str(page)

        response = requests.get(updated_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        #to find each individual article
        #https://www.w3schools.com/cssref/sel_attr_contain.php

        articles = soup.select(article_tag + "[class*='" + article_class + "']") #the class* means that it contains that class portion provided )

        for article in articles:
            #to get the headline
            locate_headline = article.select(headline_tag + "[class*='" + headline_class + "']")
            if locate_headline: #make sure that there is a headline
                headline = locate_headline[0].get_text().strip()
            else:
                headline = None
            #print("after headline")

            #to get the byline
            locate_byline = article.select(byline_tag + "[class*='" + byline_class + "']")
            if locate_byline: #make sure that there is a byline
                byline = locate_byline[0].get_text().strip()
            else:
                byline = None

            #appedning the headlines and bylines to their respective lists
            if (headline != None) and (byline != None):
                headlines.append(headline)
                bylines.append(byline)

    #creating a dataframe
    df_to_return = pd.DataFrame({"headline": headlines, "byline": bylines})
    #it was creating duplicates, thus drop them
    df_to_return = df_to_return.drop_duplicates(subset=['headline', 'byline']).reset_index(drop=True)

    return df_to_return

#journal functions
def pdf_scraper(pdf_path: str, keywords: List[str], journal_name: str = None, context_lines: int = 1) -> pd.DataFrame:
    """
    PDF Scraper function that uses fitz to scrape academic journals.

    Paramters:
    pdf_path (str): Path to the pdf file
    journal_name: Name of the journal
    keywords (list): List of keywords to search for
    context_lines (int): Number of lines around the keyword to save as data

    Returns:
    pd.DataFrame: DataFrame with columns "journal_name", "keywords", "line", "page_num",
    """

    #Sets the name of the journal if no name was given
    source = "Journal"
    if journal_name is None:
        journal_name = Path(pdf_path).stem

    line_data = []
    #scrapes the pdf using the fitz method
    def fitz_scrape():
        """
        PDF scraping using PyMuPDF
        """
        journal = fitz.open(pdf_path)

        # Get the page and text from the journal
        for page_num in range(len(journal)):
            page = journal.load_page(page_num)
            text = page.get_text()

            #Split text into lines and clean them
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            #Search for the lines that have the keyword and save them to line_data
            for i, line in enumerate(lines):
                for keyword in keywords:
                    if re.search(re.escape(keyword), line, re.IGNORECASE):
                        #get the words around the keyword
                        beg_index = max(0, i - context_lines)
                        end_index = min(len(lines), i + context_lines + 1)
                        context = ' '.join(lines[beg_index:end_index])

                        line_data.append({
                            "Headline": journal_name,
                            "keyword": keyword,
                            "byline": context,
                            "source": source
                        })
        journal.close()
        return line_data

    line_data = fitz_scrape()
    return pd.DataFrame(line_data)

def journalpdf_concat(*df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """
    Function that concatenates multiple journal DataFrames.
    
    Parameters:
    *df: a number of dataframes
    reset_index: bool with default True, determines whether to reset the index of the combined dataframe.
    
    Returns:
    pd.DataFrame: Concatenated dataframe
    """
    #checks if it is a df, if not returns a empty data frame
    if not df:
        return pd.DataFrame()
    
    #concatenates the passed in dataframes
    concatenated = pd.concat(df, ignore_index = reset_index)
    return concatenated


#magazines code

def click_button_vogue(url, num_of_clicks):
    """
    WARNING: This is very slow. Try not to run too often, as each click adds at least 3s
    waiting for new content to load


    Args:
    url - string of url
    num_of_clicks - int, how many times you want to click the "More Stories" button

    Returns BeautifulSoup object
    """

    # set up browser
    driver = webdriver.Chrome()

    # open the webpage
    driver.get("https://www.vogue.com/search?q=SUSTAINABILITY&sort=score+desc")

    # wait for page to load
    time.sleep(3)

    for num in range(num_of_clicks):
        wait = WebDriverWait(driver, 10)
        button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='More Stories']]")))
        driver.execute_script("arguments[0].scrollIntoView();", button)    # scroll to button
        button.click()
        time.sleep(3) # wait for new content to load

    page_source = driver.page_source #scrape content
    soup = BeautifulSoup(page_source, 'html.parser')
    return soup


def link2soup(link):
    """
    Convert a link to a BeautifulSoup object.
    Credit: this function was copied from the lecture notebook
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    data = requests.get(link, headers=headers).text
    return BeautifulSoup(data)
