"""Drive Chrome via CDP with realistic knowledge-worker activity; log ground truth events."""
import asyncio, json, time, random
from playwright.async_api import async_playwright
LOG = open("/home/ubuntu/screenmem/raw/events.jsonl","a")
def ev(kind, **kw):
    kw.update(kind=kind, t=time.time()); LOG.write(json.dumps(kw)+"\n"); LOG.flush()

READS = [
 "https://en.wikipedia.org/wiki/Video_compression_picture_types",
 "https://en.wikipedia.org/wiki/Perceptual_hashing",
 "https://en.wikipedia.org/wiki/AV1",
 "https://news.ycombinator.com/",
 "https://github.com/mediar-ai/screenpipe",
 "https://docs.python.org/3/library/sqlite3.html",
 "https://en.wikipedia.org/wiki/Zstd",
 "https://arxiv.org/abs/1706.03762",
 "https://en.wikipedia.org/wiki/Optical_character_recognition",
 "https://developer.mozilla.org/en-US/docs/Web/API/Screen_Capture_API",
 "https://en.wikipedia.org/wiki/Battle_of_Hastings",
 "https://www.bbc.com/news",
 "https://en.wikipedia.org/wiki/Saffron",
]
TYPING = [
 "Meeting notes: storage budget is 1 GB per day, target 100 days in 100 GB. Action item for Priya: benchmark AV1 vs HEVC at 1 fps by Friday.",
 "TODO: buy tickets to Lisbon for March 14, hotel near Alfama, budget 1200 EUR. Call dentist re invoice #48213.",
 "Design doc draft: capture -> dedup (phash) -> OCR -> FTS5 + embeddings -> tiered cold storage. Keyframes only after 7 days.",
 "Reply to Marcus about the Q3 vendor contract with Northwind Logistics; renewal price went up 18 percent, propose 9.",
]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp("http://localhost:29229")
        ctx = b.contexts[0]
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.bring_to_front()
        # phase 1: reading/browsing with scrolls
        for url in READS:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                ev("goto_fail", url=url, err=str(e)); continue
            ev("goto", url=url, title=await page.title())
            await asyncio.sleep(random.uniform(4,8))
            for _ in range(random.randint(3,7)):
                await page.mouse.wheel(0, random.randint(400,1200))
                ev("scroll", url=url)
                await asyncio.sleep(random.uniform(3,9))
        # phase 2: typing in a textarea (simulates writing)
        await page.goto("data:text/html,<textarea id=t style='width:90vw;height:85vh;font-size:22px;font-family:monospace'></textarea>")
        ev("goto", url="textarea", title="editor")
        await page.click("#t")
        for txt in TYPING:
            ev("type_start", text=txt)
            await page.type("#t", txt+"\n\n", delay=random.uniform(40,90))
            ev("type_end", text=txt)
            await asyncio.sleep(random.uniform(3,6))
        # phase 3: video (worst case for compression)
        try:
            await page.goto("https://www.youtube.com/watch?v=aqz-KE-bpKQ", wait_until="domcontentloaded", timeout=30000)
            ev("goto", url="youtube video", title=await page.title())
            await asyncio.sleep(60)
        except Exception as e:
            ev("goto_fail", url="youtube", err=str(e))
        # phase 4: more browsing
        for url in random.sample(READS, 5):
            try: await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e: ev("goto_fail", url=url, err=str(e)); continue
            ev("goto", url=url, title=await page.title())
            for _ in range(random.randint(2,5)):
                await asyncio.sleep(random.uniform(3,8)); await page.mouse.wheel(0, random.randint(300,1000)); ev("scroll", url=url)
        ev("done")
asyncio.run(main())
