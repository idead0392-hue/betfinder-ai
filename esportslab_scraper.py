"""Compatibility shim for the esports lab scraper.
This re-exports the `scrape_esportslab` function from `esportslab_scrape.py` so
other modules can import `from esportslab_scraper import scrape_esportslab`.
"""
from esportslab_scrape import scrape_esportslab

__all__ = ["scrape_esportslab"]
