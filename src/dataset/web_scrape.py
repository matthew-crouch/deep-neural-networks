"""Web scraping utility functions to extract text and links from a webpage."""

import random
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict


def scrape_text_and_links(url):
    """Scrape text content and links from a webpage."""
    try:
        # Send a GET request to the URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all heading tags
        headings = soup.find_all(["h2", "h3", "h4"])

        data = []
        for heading in headings:
            section_title = heading.get_text(strip=True)
            content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3", "h4"]:
                    break
                # Collect paragraphs and lists
                if sibling.name in ["p", "ul", "ol"]:
                    content.append(sibling.get_text(strip=True))

            section_text = "\n".join(content)
            if section_text:
                data.append({"heading": section_title, "text": section_text})

        df = pd.DataFrame(data)

        links = set()
        for a_tag in soup.find_all("a", href=True):
            full_link = urljoin(url, a_tag["href"])
            links.add(full_link)

        return df, list(links)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, []


def convert_to_dataset_dict(df):
    """Convert a Pandas DataFrame into a Hugging Face DatasetDict."""
    df["id"] = [random.choice(range(1000)) for x in range(len(df))]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({"train": dataset["train"], "test": dataset["test"]})
    return dataset_dict
