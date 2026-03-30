import asyncio
from scraper import scrape_chrono24_full, Context, log_event
import os

def create_brand_scraping_tasks(brand_data_list, max_pages):
    """
    Generates a list of asynchronous scraping tasks for various watch brands.
    """
    return [
        scrape_chrono24_full(
            f"https://www.chrono24.com{relative_url}", 
            Context(
                f"data/{brand_name}_scraped.jsonl",
                "data/images"
            ), 
            max_pages, 
            1
        ) 
        for relative_url, brand_name in brand_data_list
    ]

async def run_brand_scrapers():
    """
    Coordinates the concurrent execution of brand scraping in controlled batches.
    """
    max_concurrent_scrapers = 4
    max_pages_to_scrape = 10
    brand_configurations = get_active_brand_configurations()

    execution_semaphore = asyncio.Semaphore(max_concurrent_scrapers)
    
    scraping_tasks = [
        run_throttled_brand_scraper(config, max_pages_to_scrape, execution_semaphore)
        for config in brand_configurations
    ]

    await asyncio.gather(*scraping_tasks)

async def run_throttled_brand_scraper(config: list, total_pages: int, semaphore: asyncio.Semaphore):
    """
    Executes a single brand scraper once a slot in the batch becomes available.
    """
    path_suffix, brand_name = config
    base_url = f"https://www.chrono24.com{path_suffix}"
    
    async with semaphore:
        log_event(f"Batch Slot Acquired: Starting {brand_name}")
        brand_context = create_brand_context(brand_name)
        
        await scrape_chrono24_full(base_url, brand_context, total_pages, starting_page=1)
        
        log_event(f"Batch Slot Released: Finished {brand_name}")

def get_active_brand_configurations() -> list:
    """
    Returns the list of brand paths and names to be processed.
    """
    return [
        #["/rolex/index.htm", "rolex"],
        #["/omega/index.htm", "omega"],
        #["/patekphilippe/index.htm", "patekphilippe"],
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

def create_brand_context(brand_name: str) -> Context:
    """
    Initializes the storage context for a specific brand.
    """
    output_file = f"data/{brand_name}_listings.jsonl"
    download_path = f"data/images/{brand_name}"
    
    os.makedirs("data", exist_ok=True)
    os.makedirs(download_path, exist_ok=True)
    
    return Context(output_file, download_path)

if __name__ == "__main__":
    asyncio.run(run_brand_scrapers())