import os
import sys
from playwright.sync_api import sync_playwright

def test_research_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Get absolute path to research.html
        file_path = os.path.abspath("research.html")
        page.goto(f"file://{file_path}")

        # Check title
        title = page.title()
        if "Research Areas - Advanced Materials Group" not in title:
            raise AssertionError(f"Unexpected title: {title}")

        # Check heading
        heading = page.query_selector("h2")
        if heading is None:
            raise AssertionError("Heading <h2> not found")
        if heading.inner_text() != "Primary Research Areas":
            raise AssertionError(f"Unexpected heading: {heading.inner_text()}")

        # Check research cards
        cards = page.query_selector_all(".card h3")
        if len(cards) != 3:
            raise AssertionError(f"Expected 3 cards, found {len(cards)}")

        expected_titles = [
            "Composite Sandwich Panels",
            "Bio-inspired Composites",
            "Computational Modeling"
        ]

        card_titles = [card.inner_text() for card in cards]
        for expected in expected_titles:
            if expected not in card_titles:
                raise AssertionError(f"Expected card title '{expected}' not found in {card_titles}")

        print("All tests passed!")
        browser.close()

if __name__ == "__main__":
    try:
        test_research_page()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
