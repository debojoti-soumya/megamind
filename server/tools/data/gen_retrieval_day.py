#!/usr/bin/env python
# ruff: noqa: E501  (hand-written data tables: one moment / question per line)
"""Generate `retrieval_day_v2.json`: one realistic workday (07:00-19:10) of memory tokens + questions with gold ids.

    cd server && python tools/data/gen_retrieval_day.py          # rewrites tools/data/retrieval_day_v2.json
    python tools/data/gen_retrieval_day.py --check               # exit 1 if the committed file is stale

Deterministic (seeded `random.Random`): the same script always writes the same file. The day is built from
- EPISODES: where the wearer is, with filler frame templates sampled every ~`gap` seconds (repeated templates are
  the near-duplicate frames a static scene produces: same base dHash with 1-2 flipped bits);
- MOMENTS: hand-written key frames / speech lines at fixed times (`dups` adds near-duplicate re-captures);
- QUESTIONS: each gold item is a group of equivalent ids (any one counts); a question with several groups needs all
  of them (multi-hop). Gold specs: a moment name (the moment + its re-captures), `range:HH:MM-HH:MM` (every token in
  that window: pure-time / recency questions), `latest:transcript` (the last speech token).
Token ids are opaque (`f0123` frames, `s012` speech) so no backend can read the answer off an id; each question
carries a `note` naming the moments, and `moment` on a stream entry says which hand-written moment it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "retrieval_day_v2.json"
SEED = 20261109
DATE, NOW, TZ = "2026-11-09", "19:10", "America/Los_Angeles"
GAP_SCALE = 1.25  # stretches every episode's filler cadence (~500 tokens for the day)
SPEECH = 'Speech heard nearby: "{}"'

# fmt: off
# (code, start, end, mean gap s, filler templates (caption, ocr, entities))
EPISODES = [
    ("home-am", "07:00", "07:38", 55, [
        ("A kitchen with a kettle on the stove and morning light through the window.", "", ["kitchen", "kettle"]),
        ("A bowl of cereal and a spoon on the kitchen table.", "", ["cereal", "bowl", "table"]),
        ("A fruit bowl with bananas and oranges on the counter.", "", ["fruit bowl", "bananas"]),
        ("Looking down at a phone showing a messaging app.", "", ["phone", "messages"]),
        ("A cat sitting on a chair next to the kitchen table.", "", ["cat", "chair"]),
        ("A hallway with framed photos on the wall.", "", ["hallway", "photos"]),
        ("A bathroom sink with a toothbrush and toothpaste.", "", ["sink", "toothbrush"]),
    ]),
    ("commute-am", "07:39", "08:12", 70, [
        ("A residential street with parked cars and trees.", "", ["street", "cars", "trees"]),
        ("Crosswalk at an intersection with a walk signal.", "", ["crosswalk", "signal"]),
        ("Inside a bus, rows of seats and passengers looking at phones.", "", ["bus", "passengers"]),
        ("View out of a bus window of shops passing by.", "", ["bus window", "shops"]),
        ("A busy downtown sidewalk with office buildings.", "", ["sidewalk", "buildings"]),
    ]),
    ("office-am", "08:13", "09:29", 95, [
        ("Laptop on a desk showing a code editor with a Python file open.", "uploads.py", ["laptop", "code editor"]),
        ("A monitor showing a terminal running tests, most of them passing.", "142 passed", ["monitor", "terminal"]),
        ("Hands typing on a black keyboard.", "", ["keyboard", "hands"]),
        ("Open-plan office with rows of desks and a few coworkers.", "", ["office", "desks", "coworkers"]),
        ("A coffee mug next to the keyboard on the desk.", "", ["mug", "keyboard"]),
        ("A whiteboard with boxes and arrows of a system diagram.", "INGEST -> ENRICH -> STORE", ["whiteboard"]),
    ]),
    ("standup", "09:30", "09:50", 50, [
        ("Coworkers standing in a circle in the open office.", "", ["people", "office"]),
        ("A team board with sticky notes in columns.", "TODO  DOING  DONE", ["team board", "sticky notes"]),
    ]),
    ("office-am2", "09:51", "10:28", 95, [
        ("Laptop on a desk showing a code editor with a Python file open.", "uploads.py", ["laptop", "code editor"]),
        ("Browser showing a dashboard with request graphs.", "requests/s", ["browser", "dashboard"]),
        ("Hands typing on a black keyboard.", "", ["keyboard", "hands"]),
    ]),
    ("design-review", "10:29", "11:20", 70, [
        ("A conference room with people seated around a long table, a projector screen on.", "", ["conference room"]),
        ("A man presenting next to a projected slide titled Architecture.", "Architecture", ["presenter", "slide"]),
        ("Laptops and notebooks on a conference table.", "", ["laptops", "table"]),
    ]),
    ("office-late-am", "11:21", "12:06", 95, [
        ("Laptop on a desk showing a code editor with a Python file open.", "retrieval.py", ["laptop", "code editor"]),
        ("A monitor showing an email inbox.", "Inbox (12)", ["monitor", "email"]),
        ("Hands typing on a black keyboard.", "", ["keyboard", "hands"]),
    ]),
    ("lunch", "12:08", "13:05", 75, [
        ("A busy restaurant dining room with tables and people eating.", "", ["restaurant", "tables"]),
        ("A glass of water and chopsticks on a wooden table.", "", ["glass", "chopsticks"]),
        ("A waiter walking past carrying plates.", "", ["waiter", "plates"]),
    ]),
    ("office-pm", "13:07", "13:59", 95, [
        ("Laptop on a desk showing a code editor with a Python file open.", "hier.py", ["laptop", "code editor"]),
        ("A monitor showing a terminal running tests, most of them passing.", "198 passed", ["monitor", "terminal"]),
        ("Open-plan office with rows of desks and a few coworkers.", "", ["office", "desks"]),
    ]),
    ("video-call", "14:00", "14:36", 75, [
        ("Laptop showing a video call grid with several participants.", "", ["laptop", "video call"]),
        ("Taking notes in a notebook during a call.", "", ["notebook", "pen"]),
    ]),
    ("kitchen-break", "14:37", "14:53", 55, [
        ("Office kitchen counter with a coffee machine and paper cups.", "", ["coffee machine", "cups"]),
        ("Coworkers chatting in the office kitchen.", "", ["coworkers", "kitchen"]),
    ]),
    ("office-pm2", "14:54", "15:29", 95, [
        ("Laptop on a desk showing a code editor with a Python file open.", "hier.py", ["laptop", "code editor"]),
        ("Browser showing a dashboard with request graphs.", "requests/s", ["browser", "dashboard"]),
    ]),
    ("one-on-one", "15:30", "15:56", 70, [
        ("A small meeting room with two chairs and a round table.", "", ["meeting room", "table"]),
        ("A notebook with a few handwritten bullet points.", "", ["notebook"]),
    ]),
    ("walk-dentist", "15:57", "16:11", 80, [
        ("A busy downtown sidewalk with office buildings.", "", ["sidewalk", "buildings"]),
        ("Crosswalk at an intersection with a walk signal.", "", ["crosswalk", "signal"]),
    ]),
    ("dentist", "16:12", "17:00", 85, [
        ("A dental office waiting room with chairs and a reception desk.", "", ["waiting room", "reception"]),
        ("Looking up at a bright dental lamp from the chair.", "", ["dental lamp", "ceiling"]),
        ("A dental tray with instruments.", "", ["dental tray", "instruments"]),
    ]),
    ("grocery", "17:08", "17:42", 55, [
        ("A grocery store aisle with shelves of canned food.", "", ["aisle", "shelves"]),
        ("A shopping basket with a few items in it.", "", ["basket"]),
        ("A produce section with stacks of fruit and vegetables.", "", ["produce"]),
    ]),
    ("walk-home", "17:43", "18:01", 75, [
        ("A residential street with parked cars and trees.", "", ["street", "cars", "trees"]),
        ("Carrying a grocery bag down the sidewalk.", "", ["grocery bag", "sidewalk"]),
    ]),
    ("home-pm", "18:02", "19:08", 65, [
        ("A living room with a grey couch and a TV.", "", ["living room", "couch", "tv"]),
        ("A kitchen with a kettle on the stove.", "", ["kitchen", "kettle"]),
        ("A cat sitting on a chair next to the kitchen table.", "", ["cat", "chair"]),
        ("Looking down at a phone showing a messaging app.", "", ["phone", "messages"]),
        ("A bowl of fruit on the kitchen counter.", "", ["fruit bowl"]),
    ]),
]

# name: (time, kind f|s, caption or speech, ocr, entities, near-duplicate re-captures)
MOMENTS: dict[str, tuple] = {
    "coffee_am": ("07:04:10", "f", "A hand pouring coffee from a French press into a white mug on the kitchen counter.", "", ["french press", "mug", "coffee"], 1),
    "pills": ("07:08:05", "f", "A pill organizer on the kitchen windowsill with the Monday compartment open.", "MON TUE WED THU FRI SAT SUN", ["pill organizer", "windowsill"], 0),
    "fridge": ("07:10:20", "f", "Fridge door with a plumber's magnet and a handwritten grocery list held by a magnet.", "RAPID PLUMBING 415-555-0182\nGROCERY: oat milk, avocados, pasta, basil", ["fridge", "magnet", "grocery list"], 0),
    "news": ("07:12:40", "f", "Phone screen showing a news article while sitting at the kitchen table.", "City council approves new bike lanes on Valencia Street", ["phone", "news article"], 0),
    "sam_dentist": ("07:15:00", "s", "Hey, don't forget your dentist appointment is at four fifteen today. And we're out of oat milk.", "", ["speech"], 0),
    "keys_am": ("07:21:30", "f", "Car keys with a red keychain and a brown wallet on the hallway table by the front door.", "", ["keys", "red keychain", "wallet", "hallway table"], 1),
    "umbrella": ("07:30:15", "f", "A black umbrella leaning against the shoe rack next to the front door.", "", ["umbrella", "shoe rack", "door"], 0),
    "left_house": ("07:36:30", "f", "Closing the front door and locking it from outside; a porch with potted plants.", "", ["front door", "lock", "porch"], 0),
    "bus_stop": ("07:44:00", "f", "A bus stop on a city street with a sign listing the route; a few people waiting.", "MUNI 22 FILLMORE  NEXT BUS 4 MIN", ["bus stop", "sign"], 0),
    "calendar": ("07:52:30", "f", "Inside a bus, a phone screen showing today's calendar.", "9:30 Standup\n10:30 Design review (Redwood)\n14:00 Call w/ Aiko\n15:30 1:1 Elena\n16:15 Dentist", ["phone", "calendar"], 0),
    "coffee_shop": ("08:03:00", "f", "Coffee shop counter with a barista handing over a cup; a menu board behind.", "BLUE BOTTLE COFFEE\nOAT LATTE 5.75\nDRIP 3.50", ["coffee shop", "barista", "cup", "menu board"], 0),
    "coffee_receipt": ("08:05:10", "f", "A paper receipt held in hand outside the coffee shop.", "BLUE BOTTLE  OAT LATTE  $5.75  TAX 0.55  TOTAL $6.30", ["receipt"], 0),
    "lobby": ("08:15:30", "f", "Office building lobby with a directory sign and security turnstiles.", "NORTHWIND LABS  FLOORS 4-6\nVISITORS CHECK IN AT DESK", ["lobby", "directory sign", "turnstiles"], 0),
    "wifi": ("08:22:00", "f", "A yellow sticky note stuck to the monitor bezel.", "GUEST WIFI: nw-guest  PW: tangerine-42", ["sticky note", "monitor"], 1),
    "usb_desk": ("08:26:40", "f", "A silver USB stick next to the keyboard on the desk.", "", ["usb stick", "keyboard", "desk"], 0),
    "budget_email": ("08:40:00", "f", "Laptop showing an email from Marcus about the budget.", "From: Marcus Lee\nSubject: Q4 budget numbers\nWe are at 82% of the cloud budget with six weeks left.", ["laptop", "email"], 0),
    "bottle_am": ("09:05:00", "f", "A green metal water bottle on the desk beside the laptop.", "", ["water bottle", "desk", "laptop"], 0),
    "jira": ("09:15:20", "f", "Monitor showing a Jira board with one ticket highlighted in progress.", "MM-214 Fix upload retries  IN PROGRESS", ["monitor", "jira board"], 0),
    "standup_people": ("09:31:00", "f", "Five people standing in a circle near a team board in the open office.", "", ["people", "team board"], 0),
    "priya_blocked": ("09:33:10", "s", "I'm blocked on the API keys for the staging cluster, I need someone from infra.", "", ["speech"], 0),
    "priya_frame": ("09:33:40", "f", "A woman with short black hair in a green sweater holding a laptop, talking.", "", ["woman", "green sweater", "laptop"], 0),
    "me_ping_infra": ("09:34:30", "s", "Thanks Priya, I can ping infra for you after standup.", "", ["speech"], 0),
    "marcus_release": ("09:36:20", "s", "The release candidate is cut, I'll tag it right after lunch.", "", ["speech"], 0),
    "dana_offsite": ("09:40:00", "s", "Reminder that the team offsite is on December 4th in Half Moon Bay.", "", ["speech"], 0),
    "slack_infra": ("09:55:00", "f", "Slack window with a message posted in the infra channel.", "#infra  can someone give Priya staging API keys? thanks!", ["slack", "laptop"], 0),
    "redwood": ("10:30:00", "f", "Door sign for a conference room.", "REDWOOD  4th floor  capacity 10", ["door sign", "conference room"], 0),
    "slide_latency": ("10:41:00", "f", "Projected slide with a latency chart.", "Recall latency  p50 180 ms  p95 640 ms  target p95 < 500 ms", ["projector", "slide", "chart"], 1),
    "slide_cost": ("10:48:30", "f", "Projected slide with a cost table.", "Cost per 1k queries: $0.42  Storage: $18/month", ["slide", "table"], 0),
    "cookies": ("10:52:00", "f", "A plate of chocolate chip cookies and a coffee carafe on the conference table.", "", ["cookies", "carafe"], 0),
    "whiteboard_ship": ("10:58:00", "f", "Whiteboard with a hand-drawn timeline and dates.", "BETA Nov 20\nGA Dec 15\nOWNER: Dana", ["whiteboard", "timeline"], 2),
    "marcus_spec": ("11:05:00", "f", "A man with glasses in a navy hoodie handing over a printed spec document across the table.", "Upload Service v2 Spec - DRAFT", ["man", "glasses", "navy hoodie", "document"], 0),
    "marcus_spec_said": ("11:05:20", "s", "Here's the printed spec Marcus put together, page four has the retry policy.", "", ["speech"], 0),
    "decision": ("11:12:00", "s", "Okay, decision: we ship the beta on November 20th and cut the image tower for now.", "", ["speech"], 0),
    "paper": ("11:30:00", "f", "Laptop showing a PDF research paper.", "Sparse Retrieval at Scale  Section 3: Learned Term Weights", ["laptop", "pdf", "paper"], 1),
    "notebook": ("11:45:00", "f", "An open paper notebook with a handwritten to-do list.", "TODO: review PR 147, book flight to Austin, call mom", ["notebook", "pen"], 0),
    "usb_bag": ("11:58:00", "f", "Putting the silver USB stick into the front pocket of a grey laptop bag.", "", ["usb stick", "laptop bag"], 0),
    "thai_front": ("12:12:00", "f", "Restaurant storefront with an orange awning.", "SUNNY THAI KITCHEN  OPEN 11-9", ["restaurant", "awning"], 0),
    "menu": ("12:16:00", "f", "A menu on the restaurant table.", "PAD SEE EW 14.50\nGREEN CURRY 15.95\nTHAI ICED TEA 4.50", ["menu"], 0),
    "me_jordan": ("12:20:30", "s", "Good to see you Jordan, how was the trip?", "", ["speech"], 0),
    "jordan": ("12:21:00", "f", "A man with a beard and a red flannel shirt sitting across the table, smiling.", "", ["man", "beard", "red flannel shirt"], 0),
    "jordan_lisbon": ("12:24:00", "s", "I just got back from Lisbon, the pasteis de nata were unreal.", "", ["speech"], 0),
    "food": ("12:30:00", "f", "A plate of pad see ew with broccoli and a glass of Thai iced tea.", "", ["pad see ew", "thai iced tea"], 1),
    "jordan_job": ("12:40:00", "s", "I'm starting a new job at a climbing gym startup in January.", "", ["speech"], 0),
    "lunch_bill": ("12:58:00", "f", "A restaurant bill on a small tray with a credit card.", "SUBTOTAL 34.95  TAX 2.88  TIP 7.00  TOTAL 44.83", ["bill", "credit card"], 0),
    "vendor": ("13:20:00", "f", "Laptop showing a vendor agreement document.", "MASTER SERVICES AGREEMENT  Term: 24 months  Termination: 60 days written notice", ["laptop", "contract"], 1),
    "rc_tag": ("13:40:00", "f", "Slack window with a message from Marcus in the release channel.", "Marcus Lee: tagged v2.3.0-rc1, please smoke test", ["slack", "laptop"], 0),
    "glasses_drawer": ("13:50:00", "f", "Putting tortoiseshell reading glasses into the top desk drawer.", "", ["reading glasses", "desk drawer"], 0),
    "aiko_call": ("14:02:00", "f", "Laptop showing a video call with a woman with long hair in a bright room.", "Aiko Tanaka (Tokyo)", ["laptop", "video call", "woman"], 1),
    "aiko_dataset": ("14:08:00", "s", "We finished labeling the Tokyo dataset, about twelve thousand frames in total.", "", ["speech"], 0),
    "label_chart": ("14:14:00", "f", "Shared screen in a video call showing a bar chart of label counts.", "Labels by class: person 5.2k, vehicle 3.1k, sign 1.9k", ["bar chart", "video call"], 0),
    "aiko_visit": ("14:20:00", "s", "I'll be visiting the San Francisco office the week of December 7th.", "", ["speech"], 0),
    "cake": ("14:44:00", "f", "A birthday cake with candles on the office kitchen island.", "HAPPY BIRTHDAY PRIYANKA", ["cake", "candles"], 0),
    "cake_said": ("14:45:00", "s", "Happy birthday Priyanka! Who brought the cake?", "", ["speech"], 0),
    "bottle_refill": ("14:50:00", "f", "Refilling the green metal water bottle at the office water dispenser.", "", ["water bottle", "water dispenser"], 0),
    "pr_review": ("15:05:00", "f", "GitHub pull request page with review comments.", "PR #147 headset: set MaxFramerate while disabled  2 comments", ["laptop", "github"], 0),
    "fern": ("15:30:00", "f", "Small meeting room door sign.", "FERN  4th floor  capacity 4", ["door sign"], 0),
    "me_elena": ("15:31:00", "s", "Hi Elena, thanks for making time today.", "", ["speech"], 0),
    "elena": ("15:32:00", "f", "A woman with curly grey hair and a black blazer sitting at a small table with a notebook.", "", ["woman", "black blazer", "notebook"], 0),
    "bottle_fern": ("15:34:00", "f", "The green metal water bottle on the small round table in the meeting room.", "", ["water bottle", "table"], 0),
    "promo": ("15:38:00", "s", "Your promotion packet looks strong, let's revisit it in December after calibration.", "", ["speech"], 0),
    "austin": ("15:46:00", "s", "Also, could you present the retrieval work at the Austin summit in January?", "", ["speech"], 0),
    "dentist_sign": ("16:12:00", "f", "Entrance of a dental office with a sign.", "BRIGHT SMILE DENTAL  Dr. Patel  Suite 210", ["sign", "entrance"], 0),
    "magazine": ("16:20:00", "f", "A magazine on a side table in the waiting room.", "NATIONAL GEOGRAPHIC  The Secret Life of Octopuses", ["magazine", "waiting room"], 0),
    "floss": ("16:40:00", "s", "Everything looks good, just floss more on the lower left side.", "", ["speech"], 0),
    "appt_card": ("16:55:00", "f", "An appointment card held in hand.", "BRIGHT SMILE DENTAL  Next visit: Thu Dec 3, 9:00 AM  (415) 555-0147", ["appointment card"], 0),
    "store": ("17:10:00", "f", "Grocery store entrance with shopping carts.", "GREEN LEAF MARKET", ["grocery store", "carts"], 0),
    "oat_milk": ("17:16:00", "f", "A carton of oat milk held in hand in the dairy aisle, a shelf price tag below.", "OATLY OAT MILK 64oz  $4.99", ["oat milk", "price tag"], 1),
    "almond_milk": ("17:17:30", "f", "A shelf of almond milk cartons with a price tag.", "ALMOND BREEZE  $3.49", ["almond milk", "price tag"], 0),
    "avocados": ("17:22:00", "f", "Picking avocados from a produce display.", "HASS AVOCADOS 2 for $3", ["avocados", "produce"], 0),
    "basil": ("17:26:00", "f", "A bunch of fresh basil and a box of spaghetti in a shopping basket.", "", ["basil", "spaghetti", "basket"], 0),
    "checkout": ("17:36:00", "f", "Checkout card reader screen showing the total.", "TOTAL $38.62  INSERT OR TAP CARD", ["card reader"], 0),
    "cashier": ("17:37:00", "s", "Do you want a bag for five cents?", "", ["speech"], 0),
    "bike_lane": ("17:50:00", "f", "A street with a newly painted green bike lane.", "VALENCIA ST", ["bike lane", "street"], 0),
    "keys_bowl": ("18:03:00", "f", "Car keys with a red keychain dropped into a ceramic bowl on the kitchen counter.", "", ["keys", "red keychain", "bowl", "kitchen counter"], 1),
    "unpack": ("18:06:00", "f", "Unpacking groceries on the counter: oat milk, avocados, basil and spaghetti.", "", ["groceries", "oat milk", "avocados"], 0),
    "mom_call": ("18:20:00", "s", "Hi Mom! Yes, I'll be home for Thanksgiving, landing Wednesday evening.", "", ["speech"], 0),
    "mom_recipe": ("18:24:00", "s", "The secret to the stuffing is to toast the bread first, then add the sage.", "", ["speech"], 0),
    "cooking": ("18:35:00", "f", "A pot of spaghetti boiling on the stove with a jar of pesto next to it.", "", ["pot", "stove", "spaghetti", "pesto"], 1),
    "plants": ("18:48:00", "f", "Watering a fiddle leaf fig by the living room window.", "", ["plant", "watering can"], 0),
    "book": ("18:58:00", "f", "Reading a paperback book on the couch.", "THE OVERSTORY  Richard Powers  page 212", ["book", "couch"], 1),
    "charger": ("19:05:00", "f", "Phone plugged into a charger on the nightstand.", "", ["phone", "charger", "nightstand"], 0),
}

# small talk heard during the day: speech distractors
FILLER_SPEECH = [
    ("07:33:00", "Have a good day, see you tonight."),
    ("08:04:00", "Oat latte for here or to go?"),
    ("09:20:00", "Is anyone using the big monitor in the corner?"),
    ("10:35:00", "Can everyone see my screen okay?"),
    ("10:44:00", "Let's take questions at the end."),
    ("12:14:00", "Table for two? Right this way."),
    ("13:15:00", "Did anyone order the team lunch for Thursday?"),
    ("14:03:00", "Can you hear me now? My audio was cutting out."),
    ("14:30:00", "Sounds good, let's sync again next week."),
    ("16:14:00", "Please fill out this form and have a seat."),
    ("17:30:00", "Cleanup on aisle four please."),
    ("18:40:00", "Dinner is almost ready."),
]

# (type, question, gold specs (str = one group; list = alternatives), time_from, time_to, note)
QUESTIONS = [
    # locate_object: last sighting / where it was put
    ("locate", "where are my keys", ["keys_bowl"], None, None, "latest sighting (bowl) beats the morning hallway table"),
    ("locate", "where did I leave my keys this morning", ["keys_am"], None, None, ""),
    ("locate", "where is my water bottle", ["bottle_fern"], None, None, "last seen in the Fern room (after the refill)"),
    ("locate", "where did I put my reading glasses", ["glasses_drawer"], None, None, ""),
    ("locate", "where is my USB stick", ["usb_bag"], None, None, "moved from desk to laptop bag"),
    ("locate", "where was the USB stick this morning", ["usb_desk"], None, None, ""),
    ("locate", "where was the umbrella", ["umbrella"], None, None, ""),
    ("locate", "where did I see the pill organizer", ["pills"], None, None, ""),
    ("locate", "where is my phone charging", ["charger"], None, None, ""),
    ("locate", "where did I refill my water bottle", ["bottle_refill"], None, None, ""),
    # read_text: what did I read / see written
    ("read", "what's the guest wifi password", ["wifi"], None, None, ""),
    ("read", "what was the p95 latency on the slide", ["slide_latency"], None, None, "p50 is the distractor"),
    ("read", "what was the cost per 1k queries in the design review", ["slide_cost"], None, None, ""),
    ("read", "when is the beta according to the whiteboard", ["whiteboard_ship"], None, None, ""),
    ("read", "how much was the oat milk", ["oat_milk"], None, None, "almond milk price is the distractor"),
    ("read", "what was the grocery total", ["checkout"], None, None, ""),
    ("read", "how much did lunch cost", ["lunch_bill"], None, None, ""),
    ("read", "what's the plumber's phone number", ["fridge"], None, None, "dentist number is the distractor"),
    ("read", "what's the dentist's phone number", ["appt_card"], None, None, ""),
    ("read", "when is my next dentist appointment", ["appt_card"], None, None, ""),
    ("read", "what paper was I reading", ["paper"], None, None, ""),
    ("read", "what did the vendor agreement say about termination", ["vendor"], None, None, ""),
    ("read", "which bus route did I take", ["bus_stop"], None, None, ""),
    ("read", "what was on my calendar today", ["calendar"], None, None, ""),
    ("read", "what did the budget email say", ["budget_email"], None, None, ""),
    ("read", "which Jira ticket was in progress", ["jira"], None, None, ""),
    ("read", "what was the news article about this morning", ["news"], None, None, ""),
    ("read", "what book am I reading", ["book"], None, None, ""),
    ("read", "what was on my to-do list", ["notebook"], None, None, ""),
    ("read", "what magazine was in the waiting room", ["magazine"], None, None, ""),
    ("read", "how much was my coffee this morning", [["coffee_receipt", "coffee_shop"]], None, None, ""),
    ("read", "what room was the design review in", [["redwood", "calendar"]], None, None, ""),
    # people
    ("people", "who did I have lunch with", [["jordan", "me_jordan"]], None, None, ""),
    ("people", "who handed me the printed spec", [["marcus_spec", "marcus_spec_said"]], None, None, ""),
    ("people", "who was on the video call", ["aiko_call"], None, None, ""),
    ("people", "who was blocked on API keys at standup", [["priya_blocked", "me_ping_infra"]], None, None, ""),
    ("people", "what was the woman in the green sweater talking about", [["priya_frame", "priya_blocked"]], None, None, ""),
    ("people", "who did I have my one on one with", [["elena", "me_elena", "fern"]], None, None, ""),
    ("people", "whose birthday was it today", [["cake", "cake_said"]], None, None, "Priyanka vs Priya"),
    ("people", "who did I meet in the small meeting room", [["elena", "me_elena"]], None, None, ""),
    ("people", "what was the man with the beard wearing", ["jordan"], None, None, ""),
    # heard
    ("heard", "what did Jordan say about Lisbon", ["jordan_lisbon"], None, None, ""),
    ("heard", "what did Jordan say about his new job", ["jordan_job"], None, None, ""),
    ("heard", "what did the dentist tell me", ["floss"], None, None, ""),
    ("heard", "when is the team offsite", ["dana_offsite"], None, None, ""),
    ("heard", "what was decided in the design review", ["decision"], None, None, ""),
    ("heard", "what did my manager say about my promotion", ["promo"], None, None, ""),
    ("heard", "when is Aiko visiting San Francisco", ["aiko_visit"], None, None, ""),
    ("heard", "how many frames did the Tokyo team label", ["aiko_dataset"], None, None, ""),
    ("heard", "what did I tell my mom about Thanksgiving", ["mom_call"], None, None, ""),
    ("heard", "what's the secret to the stuffing", ["mom_recipe"], None, None, ""),
    ("heard", "what did I hear at home this morning about the dentist", ["sam_dentist"], None, None, ""),
    ("heard", "what page of the spec has the retry policy", ["marcus_spec_said"], None, None, ""),
    ("heard", "what did the cashier ask me", ["cashier"], None, None, ""),
    ("heard", "what was I asked to present in Austin", ["austin"], None, None, ""),
    # temporal anchors (before / after an event)
    ("temporal", "what was I reading before lunch", [["paper", "notebook"]], None, None, ""),
    ("temporal", "what was I working on right after lunch", ["vendor"], None, None, "first thing back at the desk"),
    ("temporal", "what did I see before I left the house this morning", [["umbrella", "keys_am", "left_house"]], None, None, ""),
    ("temporal", "what did I buy after the dentist", [["oat_milk", "avocados", "basil", "checkout", "almond_milk"]], None, None, ""),
    ("temporal", "what did I do after I got home", [["keys_bowl", "unpack"]], None, None, ""),
    ("temporal", "what was I doing before standup", ["range:09:10-09:29"], None, None, ""),
    ("temporal", "what did I eat after the design review", ["food"], None, None, ""),
    ("temporal", "what was on the slides before the cookies came out", [["slide_latency", "slide_cost"]], None, None, ""),
    # time ranges: parsed times or explicit agent-supplied bounds
    ("time_range", "what was I doing at 11", ["range:10:55-11:06"], None, None, ""),
    ("time_range", "what was I looking at around 2:15pm", ["range:14:10-14:20"], None, None, ""),
    ("time_range", "what was I doing this morning at 7:30", ["range:07:25-07:35"], None, None, ""),
    ("time_range", "what was I doing between 4 and 5pm", ["range:16:00-17:00"], None, None, ""),
    ("time_range", "what text did I see", [["coffee_shop", "coffee_receipt", "lobby", "wifi"]], "08:00", "08:30", ""),
    ("time_range", "what slides were shown", [["slide_latency", "slide_cost"]], "10:30", "11:20", ""),
    ("time_range", "what was on my laptop", ["vendor"], "13:10", "13:35", ""),
    ("time_range", "what price tags did I see", [["oat_milk", "almond_milk", "avocados"]], "17:00", "17:45", ""),
    ("time_range", "who was I talking to", [["jordan", "me_jordan", "jordan_lisbon", "jordan_job"]], "12:00", "13:00", ""),
    ("time_range", "what was on the whiteboard", ["whiteboard_ship"], "10:30", "11:20", "office whiteboard is outside the window"),
    # recency
    ("recency", "what did I just see", ["range:19:00-19:10"], None, None, ""),
    ("recency", "what was I doing 10 minutes ago", ["range:18:56-19:04"], None, None, ""),
    ("recency", "what did I just hear", ["latest:transcript"], None, None, ""),
    ("recency", "what was I doing half an hour ago", ["range:18:35-18:45"], None, None, ""),
    # multi-hop: every group must be found
    ("multi_hop", "did I buy everything on the grocery list from the fridge", ["fridge", ["oat_milk", "avocados", "basil", "checkout", "unpack"]], None, None, ""),
    ("multi_hop", "was my morning coffee more expensive than the oat milk", [["coffee_receipt", "coffee_shop"], "oat_milk"], None, None, ""),
    ("multi_hop", "did I walk down the street from the news article I read this morning", ["news", "bike_lane"], None, None, ""),
    ("multi_hop", "Priya was blocked at standup, did I ask infra for help", [["priya_blocked", "me_ping_infra"], "slack_infra"], None, None, ""),
    ("multi_hop", "Marcus said he'd tag the release after lunch, did he", ["marcus_release", "rc_tag"], None, None, ""),
    ("multi_hop", "does the dentist time on my calendar match what I was told at home", ["calendar", "sam_dentist"], None, None, ""),
    ("multi_hop", "does the beta date on the whiteboard match the decision in the meeting", ["whiteboard_ship", "decision"], None, None, ""),
    ("multi_hop", "where did I move the USB stick from and to", ["usb_desk", "usb_bag"], None, None, ""),
    ("multi_hop", "where were my keys this morning and where are they now", ["keys_am", "keys_bowl"], None, None, ""),
    # what did I see / do
    ("see", "what did I have for lunch", [["food", "menu"]], None, None, ""),
    ("see", "what did I cook for dinner", ["cooking"], None, None, ""),
    ("see", "which plant did I water", ["plants"], None, None, ""),
    ("see", "did I see a birthday cake today", ["cake"], None, None, ""),
    ("see", "what was on the plate in the conference room", ["cookies"], None, None, ""),
]

# fmt: on


def _sec(hhmmss: str) -> int:
    p = [int(x) for x in hhmmss.split(":")] + [0]
    return p[0] * 3600 + p[1] * 60 + p[2]


def _hhmmss(s: int) -> str:
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _flip(h: int, rng: random.Random, bits: int) -> int:
    for b in rng.sample(range(64), bits):
        h ^= 1 << b
    return h


def _base_hash(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def _variant(caption: str, i: int) -> str:
    body = caption.rstrip(".")
    return [f"Close-up: {body[0].lower()}{body[1:]}.", f"{body}, seen from a slightly different angle."][i % 2]


def build() -> dict:
    rng = random.Random(SEED)
    rows: list[dict] = []
    moments_by_name: dict[str, list[int]] = {}  # name -> row indices (moment + re-captures)
    ep_of = lambda s: next((code for code, a, b, *_ in EPISODES if _sec(a) <= s <= _sec(b) + 59), "transit")  # noqa: E731
    reserved = []
    for name, (t, kind, text, ocr, ents, dups) in MOMENTS.items():
        s = _sec(t)
        moments_by_name[name] = []
        h = _base_hash(name)
        for d in range(dups + 1):
            ts = s + d * rng.randint(18, 35)
            caption = SPEECH.format(text) if kind == "s" else (text if d == 0 else _variant(text, d - 1))
            rows.append(
                {
                    "_s": ts,
                    "kind": "transcript" if kind == "s" else "frame",
                    "episode": ep_of(s),
                    "caption": caption,
                    "ocr": ocr,
                    "entities": ents,
                    "hash": "" if kind == "s" else f"{(h if d == 0 else _flip(h, rng, 2)):016x}",
                    "moment": name,
                }
            )
            moments_by_name[name].append(len(rows) - 1)
            reserved.append(ts)
    for t, text in FILLER_SPEECH:
        rows.append(
            {
                "_s": _sec(t),
                "kind": "transcript",
                "episode": ep_of(_sec(t)),
                "caption": SPEECH.format(text),
                "ocr": "",
                "entities": ["speech"],
                "hash": "",
            }
        )
    reserved.sort()
    for code, a, b, gap, templates in EPISODES:
        s, end = _sec(a), _sec(b)
        prev = None
        while s <= end:
            if not any(abs(s - r) < 12 for r in reserved):
                # static scenes repeat: 45% of the time keep the previous template (a near-duplicate frame)
                ti = prev if prev is not None and rng.random() < 0.45 else rng.randrange(len(templates))
                cap, ocr, ents = templates[ti]
                h = _flip(_base_hash(f"{code}:{ti}"), rng, rng.randint(1, 3))
                rows.append(
                    {
                        "_s": s,
                        "kind": "frame",
                        "episode": code,
                        "caption": cap,
                        "ocr": ocr,
                        "entities": ents,
                        "hash": f"{h:016x}",
                    }
                )
                prev = ti
            s += int(gap * GAP_SCALE * rng.uniform(0.6, 1.4))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["_s"], rows[i]["kind"], rows[i]["caption"]))
    ids: dict[int, str] = {}
    nf = ns = 0
    for i in order:
        if rows[i]["kind"] == "frame":
            nf += 1
            ids[i] = f"f{nf:04d}"
        else:
            ns += 1
            ids[i] = f"s{ns:03d}"
    stream = []
    for i in order:
        r = rows[i]
        e = {
            "id": ids[i],
            "time": _hhmmss(r["_s"]),
            "episode": r["episode"],
            "caption": r["caption"],
            "ocr": r["ocr"],
            "entities": r["entities"],
        }
        if r["kind"] != "frame":
            e["kind"] = r["kind"]
        if r["hash"]:
            e["hash"] = r["hash"]
        if r.get("moment"):
            e["moment"] = r["moment"]
        stream.append(e)

    def resolve(spec: str) -> list[str]:
        if spec.startswith("range:"):
            lo, hi = (_sec(x) for x in spec[6:].split("-"))
            return [ids[i] for i in order if lo <= rows[i]["_s"] <= hi]
        if spec == "latest:transcript":
            return [ids[max((i for i in order if rows[i]["kind"] == "transcript"), key=lambda i: rows[i]["_s"])]]
        return [ids[i] for i in moments_by_name[spec]]

    questions = []
    for qtype, q, gold, tf, tt, note in QUESTIONS:
        groups = []
        for g in gold:
            alts = g if isinstance(g, list) else [g]
            ids_ = []
            for spec in alts:
                ids_ += [x for x in resolve(spec) if x not in ids_]
            groups.append(ids_[0] if len(ids_) == 1 else ids_)
        names = [x for g in gold for x in (g if isinstance(g, list) else [g])]
        entry = {
            "q": q,
            "type": qtype,
            "gold": groups,
            "time_from": tf,
            "time_to": tt,
            "note": "; ".join(x for x in (", ".join(names), note) if x),
        }
        questions.append(entry)
    return {
        "date": DATE,
        "tz": TZ,
        "now": NOW,
        "device_id": "eval-device",
        "version": 2,
        "generator": f"tools/data/gen_retrieval_day.py (seed {SEED})",
        "stream": stream,
        "questions": questions,
    }


def dumps(data: dict) -> str:
    """One stream entry / question per line: diffable and hand-checkable."""
    head = {k: v for k, v in data.items() if k not in ("stream", "questions")}
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    for key in ("stream", "questions"):
        items = [f"    {json.dumps(x, ensure_ascii=False)}" for x in data[key]]
        lines.append(f'  "{key}": [\n' + ",\n".join(items) + "\n  ]" + ("," if key == "stream" else ""))
    return "\n".join(lines) + "\n}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="fail if --out differs from a fresh generation")
    args = ap.parse_args()
    text = dumps(build())
    if args.check:
        same = args.out.is_file() and args.out.read_text() == text
        print("up to date" if same else f"{args.out} is stale: rerun tools/data/gen_retrieval_day.py")
        return 0 if same else 1
    args.out.write_text(text)
    d = json.loads(text)
    kinds = sum(1 for e in d["stream"] if e.get("kind") == "transcript")
    print(f"wrote {args.out}: {len(d['stream'])} tokens ({kinds} speech), {len(d['questions'])} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
