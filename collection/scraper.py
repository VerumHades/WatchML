import asyncio
import json
import random
from datetime import datetime
from playwright.async_api import async_playwright, Page, ElementHandle
from playwright_stealth import Stealth
import threading

class Context:
    def __init__(self, output_filename):
        self.output_filename = output_filename

async def scrape_chrono24_full(url: str, output_filename, pages_to_run: int, starting_page: int):
    """
    Orchestrates extraction, ensuring sequential page navigation even when skipping collection.
    """
    stealth_provider = Stealth()
    clicked_elements = set()
    context = Context(output_filename)

    async with async_playwright() as playwright_launcher:
        log_event("Launching browser in headless mode...")
        browser_instance = await playwright_launcher.chromium.launch(headless=False)
        browser_context = await browser_instance.new_context()
        active_page = await browser_context.new_page()
        
        await stealth_provider.apply_stealth_async(active_page)
        await start_proactive_monitors(active_page, clicked_elements)
        
        log_event(f"Starting journey at initial URL: {url}")
        await active_page.goto(url)
        await handle_cookie_consent(active_page)
        await process_pages(context, active_page, pages_to_run, starting_page)
        await browser_instance.close()

def log_event(message: str):
    """
    Prints a formatted timestamped message to the console.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

async def process_pages(context, page: Page, total_pages: int, starting_page: int):
    """
    Navigates through pages sequentially; collects data only after reaching starting_page.
    """
    for current_index in range(1, total_pages + 1):
        if current_index < starting_page:
            log_event(f"Advancing: Page {current_index} (Skipping collection)")
        else:
            log_event(f"Active: Processing Search Page {current_index}")
            await trigger_content_load(page)
            listing_urls = await collect_listing_urls(page)
            await process_listing_urls(context, page, listing_urls)

        if current_index < total_pages:
            await navigate_to_next_page(page, current_index + 1)

async def start_proactive_monitors(page: Page, clicked_elements: set):
    """
    Background listeners for survey and registration modals.
    """
    selectors = [
        ".js-close-modal.btn.btn-link.m-r-3",
        ".btn.btn-secondary.flex-1.w-100-sm.m-r-sm-5.js-close-modal"
    ]
    page.on("requestfinished", lambda _: asyncio.create_task(
        dismiss_visible_popups(page, selectors, clicked_elements)
    ))

async def dismiss_visible_popups(page: Page, selectors: list, clicked: set):
    """
    Proactively dismisses popups in the background.
    """
    for selector in selectors:
        try:
            btn = await page.query_selector(selector)
            if btn and await btn.is_visible() and selector not in clicked:
                log_event(f"Proactive Monitor: Closing popup {selector[:15]}...")
                clicked.add(selector)
                await human_click(page, btn)
                await asyncio.sleep(4)
                clicked.discard(selector)
        except Exception:
            continue

async def process_listing_urls(context, page: Page, listing_urls: list):
    """
    Extracts data from individual watch listings on active pages.
    """
    for index, listing_url in enumerate(listing_urls, 1):
        try:
            log_event(f"  [{index}/{len(listing_urls)}] Extracting: {listing_url[:50]}...")
            await page.goto(listing_url)
            watch_specs = await extract_table_data(page)
            save_scraped_data_to_storage(context, listing_url, watch_specs)
        except Exception as extraction_error:
            log_event(f"  [ERROR] Extraction failed: {extraction_error}")
        finally:
            await return_to_listing_results(page)

async def trigger_content_load(page: Page):
    """
    Skims page to trigger lazy loading.
    """
    for position in [0.4, 0.8, 1.0]:
        scroll_script = f"window.scrollTo(0, document.body.scrollHeight * {position})"
        await page.evaluate(scroll_script)
        await asyncio.sleep(random.uniform(0.3, 0.6))

async def handle_cookie_consent(page: Page):
    """
    Handles the initial cookie wall.
    """
    try:
        cookie_btn = await page.wait_for_selector(".js-cookie-accept-all", timeout=5000)
        if cookie_btn:
            log_event("Handling initial cookie consent.")
            await human_click(page, cookie_btn)
    except Exception:
        pass

async def human_click(page: Page, target_element: ElementHandle):
    """
    Simulates human-like cursor movement and click.
    """
    box = await target_element.bounding_box()
    if box:
        await page.mouse.move(box["x"] + box["width"]/2, box["y"] + box["height"]/2, steps=10)
        await asyncio.sleep(0.3)
        await target_element.click(force=True)

async def extract_table_data(page: Page) -> list:
    """
    Parses table specifications from the listing.
    """
    await page.wait_for_selector("table", timeout=8000)
    return await page.evaluate("""() => {
        const rows = Array.from(document.querySelectorAll('table tr'));
        return rows.map(r => r.innerText.replace(/\\t/g, ': ').trim());
    }""")

def save_scraped_data_to_storage(context, source_url: str, specifications: list):
    """
    Saves data to JSONL format.
    """
    payload = {"timestamp": datetime.now().isoformat(), "url": source_url, "data": specifications}

    with open(context.output_filename, "a", encoding="utf-8") as storage_file:
        storage_file.write(json.dumps(payload) + "\n")

async def collect_listing_urls(page: Page) -> list:
    """
    Finds watch links on the current results page.
    """
    sel = "a.js-listing-item-link"
    await page.wait_for_selector(sel)
    return await page.locator(sel).evaluate_all("links => links.map(a => a.href)")

async def navigate_to_next_page(page: Page, target_page_number: int):
    """
    Clicks the pagination link for the next sequential page.
    """
    pagination_selector = "nav.pagination ul li a"
    await page.wait_for_selector(pagination_selector)
    links = await page.query_selector_all(pagination_selector)
    for link in links:
        if (await link.inner_text()).strip() == str(target_page_number):
            await human_click(page, link)
            await page.wait_for_load_state("networkidle")
            return

async def return_to_listing_results(page: Page):
    """
    Navigates back to search results.
    """
    await page.go_back()
    await page.wait_for_selector(".js-listing-item-link")
