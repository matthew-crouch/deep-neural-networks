import random
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict


def scrape_text_and_links(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise an error for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all heading tags
        headings = soup.find_all(["h2", "h3", "h4"])

        data = []
        for heading in headings:
            section_title = heading.get_text(strip=True)  # Extract heading text
            content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3", "h4"]:  # Stop when reaching the next heading
                    break
                if sibling.name in ["p", "ul", "ol"]:  # Collect paragraphs and lists
                    content.append(sibling.get_text(strip=True))

            section_text = "\n".join(content)
            if section_text:
                data.append({"heading": section_title, "text": section_text})

        df = pd.DataFrame(data)

        # Extract all links on the webpage
        links = set()
        for a_tag in soup.find_all("a", href=True):
            full_link = urljoin(url, a_tag["href"])  # Convert relative URLs to absolute
            links.add(full_link)

        return df, list(links)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, []


def convert_to_dataset_dict(sources):
    """Converts a Pandas DataFrame into a Hugging Face DatasetDict
    """
    data = []
    for url in sources:
        text_content, links = scrape_text_and_links(url)
        data.append(text_content)

    df = pd.concat(data)
    df = df[~df["heading"].str.contains("Source|Images", case=False, na=False)]
    df = df.rename(columns={"heading": "summary", "text": "document"}).reset_index(drop=True)
    df["id"] = [random.choice(range(1000)) for x in range(len(df))]
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"train": dataset, "test": dataset})
    return dataset_dict, links


# if __name__ == "__main__":
#     # url = input("Enter the website URL: ")
#     warhammer_sources = [
#         "https://wh40k.lexicanum.com/wiki/Bloodthirster",
#         "https://wh40k.lexicanum.com/wiki/Ka%27Bandha",
#         "https://wh40k.lexicanum.com/wiki/Skulltaker",
#         "https://wh40k.lexicanum.com/wiki/Doombreed"
#     ]
#     data = []
#     for url in warhammer_sources:
#         text_content, links = scrape_text_and_links(url)
#         data.append(text_content)

#     df = pd.concat(data)
#     df = df[~df["heading"].str.contains("Source|Images", case=False, na=False)]
#     df = df.rename(columns={"text": "document"}).reset_index(drop=True)
#     dataset_dict = convert_to_dataset_dict(df)
#     breakpoint()
