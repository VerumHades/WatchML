import asyncio
import json
import random
from datetime import datetime
from playwright.async_api import async_playwright, Page, ElementHandle
from playwright_stealth import Stealth
import os
import urllib.request

class Context:
    def __init__(self, output_filename, download_directory):
        self.output_filename = output_filename
        self.download_directory = download_directory

async def scrape_chrono24_full(url: str, context, pages_to_run: int, starting_page: int):
    """
    Orchestrates extraction, ensuring sequential page navigation even when skipping collection.
    """
    stealth_provider = Stealth()
    clicked_elements = set()

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

async def capture_listing_images(page: Page, storage_directory: str):
    """
    Saves clean images by removing UI obstructions and clearing hover effects.
    """
    await remove_blocking_banners(page)
    
    images_locator = page.locator("img.listing-image-gallery-image")
    image_count = await images_locator.count()

    for index in range(image_count):
        target_image = images_locator.nth(index)
        
        await prepare_ui_for_screenshot(page, target_image)
        
        file_path = os.path.join(storage_directory, f"image_{index}.jpg")
        await save_clean_element_screenshot(target_image, file_path)

async def remove_blocking_banners(page: Page):
    """
    Permanently deletes the domain hint banner and other UI noise from the DOM.
    """
    banner_selector = ".sticky-change-domain-hint"
    try:
        await page.evaluate(f"""(selector) => {{
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        }}""", banner_selector)
    except Exception:
        pass

async def prepare_ui_for_screenshot(page: Page, target_image):
    """
    Clears mouse position and waits for image rendering to avoid zoom effects.
    """
    await target_image.scroll_into_view_if_needed()
    await page.mouse.move(0, 0)
    
    # Wait for the browser to report the image is fully decoded
    await target_image.evaluate("img => img.complete && img.naturalWidth > 0")
    await asyncio.sleep(0.3)

async def save_clean_element_screenshot(locator, file_path: str):
    """
    Captures the element as a high-quality JPEG directly from the viewport.
    """
    try:
        await locator.screenshot(
            path=file_path, 
            type="jpeg", 
            quality=100,
            scale="device"
        )
    except Exception as error:
        log_event(f"Visual Extraction Error: {error}")
        
async def force_load_gallery_images(page: Page):
    """
    Ensures images are fully rendered before canvas extraction.
    """
    image_selector = "img.listing-image-gallery-image"
    await page.wait_for_selector(image_selector, timeout=5000)
    
    # Scroll images into view to trigger browser rendering
    await page.locator(image_selector).first.scroll_into_view_if_needed()
    await page.wait_for_load_state("networkidle")

async def process_listing_urls(context: Context, page: Page, listing_urls: list):
    """
    Iterates through search results to extract specifications and gallery images.
    """
    for index, url in enumerate(listing_urls, 1):
        log_event(f" [{index}/{len(listing_urls)}] Processing: {url[:40]}...")
        await process_single_listing(context, page, url)
        await return_to_listing_results(page)

async def force_load_gallery_images(page: Page):
    """
    Tricks the browser into eagerly loading all lazy-loaded images.
    """
    image_selector = "img.listing-image-gallery-image"
    await page.wait_for_selector(image_selector, timeout=5000)
    
    await page.evaluate("""(selector) => {
        const images = document.querySelectorAll(selector);
        images.forEach(img => {
            img.setAttribute('loading', 'eager');
            const currentSrc = img.src;
            img.src = ''; 
            img.src = currentSrc;
        });
    }""", image_selector)
    
    # Give the browser a moment to initiate the new requests
    await page.wait_for_load_state("networkidle")
    
async def process_single_listing(context: Context, page: Page, url: str):
    """
    Orchestrates the navigation, data extraction, and image backup for one watch.
    """
    await page.goto(url)
    
    listing_id = url.split('/')[-1].split('.')[0]
    storage_dir = os.path.join(context.download_directory, listing_id)
    os.makedirs(storage_dir, exist_ok=True)

    watch_specs = await extract_table_data(page)
    await capture_listing_images(page, storage_dir)
    
    save_scraped_data_to_storage(context, url, watch_specs, listing_id)

def save_scraped_data_to_storage(context: Context, url: str, specs: list, listing_id: str):
    """
    Records listing metadata and a reference to the local image directory.
    """
    payload = {
        "listing_id": listing_id,
        "url": url,
        "image_directory": f"downloads/{listing_id}",
        "timestamp": datetime.now().isoformat(),
        "data": specs
    }

    with open(context.output_filename, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


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

async def collect_listing_urls(page: Page) -> list:
    """
    Finds watch links while explicitly excluding those in the top model slider.
    """
    slider_class = ".wt-top-model-slider"
    link_selector = f"a.js-listing-item-link:not({slider_class} *)"
    
    await page.wait_for_selector(link_selector)
    
    return await page.locator(link_selector).evaluate_all(
        "elements => elements.map(anchor => anchor.href)"
    )

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
