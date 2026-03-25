import asyncio
from scraper import scrape_chrono24_full

if __name__ == "__main__":
    MAX_PAGES = 100

    brand_urls = [
        ["/rolex/index.htm", "rolex"],
        ["/omega/index.htm", "omega"],
        ["/patekphilippe/index.htm", "patekphilippe"],
        ["/audemarspiguet/index.htm", "audemarspiguet"],
        ["/breitling/index.htm", "breitling"],
        ["/tudor/index.htm", "tudor"],
        ["/cartier/index.htm", "cartier"],
        ["/panerai/index.htm", "panerai"],
        ["/iwc/index.htm", "iwc"],
        ["/seiko/index.htm", "seiko"],
        ["/jaegerlecoultre/index.htm", "jaegerlecoultre"],
        ["/tagheuer/index.htm", "tagheuer"]
    ]

    tasks = [
        scrape_chrono24_full(link, f"data/{brand_name}_scraped.jsonl", MAX_PAGES, 1) for [link, brand_name] in brand_urls
    ]
    
    asyncio.run(asyncio.gather(*tasks))