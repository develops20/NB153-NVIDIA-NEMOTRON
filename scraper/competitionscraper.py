#!/usr/bin/env python3
"""
Kaggle Competition Scraper

Scrapes competition information via the official Kaggle API (v2) and
Playwright-rendered page content, saving results as clean Markdown files
inside a target directory — ready for ingestion into Claude AI or for
generating a skills.md file.

Prerequisites:
    pip install kaggle requests beautifulsoup4 markdownify python-dotenv playwright
    python -m playwright install chromium

    Authenticate with one of:
      - .env file with KAGGLE_API_TOKEN=... (loaded automatically)
      - KAGGLE_API_TOKEN env var
      - ~/.kaggle/access_token file
      - ~/.kaggle/kaggle.json  (legacy username + key)

Usage:
    python competitionscraper.py nvidia-nemotron-model-reasoning-challenge
    python competitionscraper.py nvidia-nemotron-model-reasoning-challenge -o ./comp-data
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md_convert
from playwright.sync_api import sync_playwright
from kaggle.api.kaggle_api_extended import KaggleApi


def _attr(obj, *names, default="N/A"):
    """Try multiple attribute names (camelCase / snake_case) on an SDK object."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


class CompetitionScraper:
    BASE_URL = "https://www.kaggle.com"

    def __init__(self, slug: str, output_dir: str = "competition-data", delay: float = 2.0):
        self.slug = slug
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

        self.api = KaggleApi()
        self.api.authenticate()

        self._pw = None
        self._browser = None

    # ── browser lifecycle ────────────────────────────────────────────────

    def _ensure_browser(self):
        if self._browser is None:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
        return self._browser

    def _close_browser(self):
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw:
            self._pw.stop()
            self._pw = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _wait(self):
        time.sleep(self.delay)

    def _render_page(self, tab: str) -> str:
        """Use Playwright to render a competition page and return the HTML."""
        url = f"{self.BASE_URL}/competitions/{self.slug}/{tab}"
        print(f"  Rendering {url}")
        self._wait()

        browser = self._ensure_browser()
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(2000)
            html = page.content()
        finally:
            page.close()
        return html

    def _rendered_to_markdown(self, html: str) -> str:
        """Convert fully-rendered page HTML into clean Markdown."""
        soup = BeautifulSoup(html, "html.parser")

        content = (
            soup.select_one("#site-content")
            or soup.select_one("main")
            or soup.find("body")
        )
        if content is None:
            return ""

        for tag in content.find_all(
            ["script", "style", "nav", "header", "footer",
             "noscript", "iframe", "svg", "button", "input",
             "form", "select", "textarea"]
        ):
            tag.decompose()

        raw = md_convert(str(content), heading_style="ATX", strip=["img"])
        raw = self._clean_markdown(raw)
        return raw.strip()

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strip Kaggle page chrome and UI noise from converted markdown."""
        # Cookie banner
        text = re.sub(
            r"Kaggle uses cookies from Google.*?OK, Got it\.\s*",
            "", text, flags=re.DOTALL,
        )
        # Competition header + nav tabs block
        text = re.sub(
            r"NVIDIA\s+·\s+Featured Prediction Competition\s+·.*?\n",
            "", text,
        )
        text = re.sub(
            r"^# NVIDIA Nemotron Model Reasoning Challenge\n+"
            r"Advance reasoning techniques.*?\n+",
            "", text, flags=re.MULTILINE,
        )
        text = re.sub(
            r"## NVIDIA Nemotron Model Reasoning Challenge\n+"
            r"\[Overview\].*?\[Rules\]\([^\)]+\)\s*",
            "", text, flags=re.DOTALL,
        )
        # Material Design icon names leaking into text (with optional backslash-escaped underscores)
        icons = (
            r"push[\\\_]*pin|get[\\\_]*app|fullscreen|chevron[\\\_]*right|"
            r"arrow[\\\_]*right|calendar[\\\_]*view[\\\_]*week|"
            r"navigate[\\\_]*next|minimize|text[\\\_]*snippet|"
            r"emoji[\\\_]*people"
        )
        text = re.sub(rf"\[?(?:{icons})\]?", "", text)
        # Zero-width-space "search" from the search box
        text = re.sub(r"search\u200B", "", text)
        # Standalone UI labels on their own line
        text = re.sub(r"^(?:info|folder)\s*$", "", text, flags=re.MULTILINE)
        # Competition timing sidebar ("Start / 2 days ago / Close / 3 months to go")
        text = re.sub(
            r"Start\n+\d+ days? ago\n+(?:#{1,6}\s+)?Close\n+\d+ months? to go\n+"
            r"(?:Merger & Entry\n+)?",
            "", text,
        )
        # Sort/filter label in discussions
        text = re.sub(r"^Hotness\s*$", "", text, flags=re.MULTILINE)
        # Table-of-contents sidebar that appears on overview
        text = re.sub(
            r"Table of Contents\n+\[Overview\].*$",
            "", text, flags=re.DOTALL,
        )
        # Collapse excessive blank lines
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text

    def _save(self, filename: str, content: str):
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  -> {path}  ({len(content):,} chars)")

    # ── 1. overview ──────────────────────────────────────────────────────

    def scrape_overview(self):
        print("\n[1/6] Overview")

        meta = {}
        try:
            response = self.api.competitions_list(search=self.slug)
            competitions = response.competitions if hasattr(response, "competitions") else response
            if competitions:
                for c in competitions:
                    if self.slug in str(_attr(c, "ref")):
                        meta = {
                            "title": _attr(c, "title"),
                            "category": _attr(c, "category"),
                            "reward": _attr(c, "reward"),
                            "deadline": str(_attr(c, "deadline")),
                            "teams": _attr(c, "team_count", "teamCount"),
                            "metric": _attr(c, "evaluation_metric", "evaluationMetric"),
                        }
                        break
        except Exception as exc:
            print(f"  warn: competitions_list API: {exc}")

        html = self._render_page("overview")
        page_md = self._rendered_to_markdown(html)

        title = meta.get("title", self.slug.replace("-", " ").title())
        lines = [
            f"# {title}",
            "",
            f"**URL:** <https://www.kaggle.com/competitions/{self.slug}>  ",
        ]
        if meta:
            lines += [
                f"**Category:** {meta.get('category', 'N/A')}  ",
                f"**Reward:** {meta.get('reward', 'N/A')}  ",
                f"**Deadline:** {meta.get('deadline', 'N/A')}  ",
                f"**Teams:** {meta.get('teams', 'N/A')}  ",
                f"**Evaluation Metric:** {meta.get('metric', 'N/A')}  ",
            ]
        lines += ["", "---", ""]

        if page_md:
            lines.append(page_md)
        else:
            lines.append("*Could not extract overview content.*")

        self._save("overview.md", "\n".join(lines))

    # ── 2. data ──────────────────────────────────────────────────────────

    def scrape_data(self):
        print("\n[2/6] Data")

        lines = ["# Dataset Description", ""]

        try:
            response = self.api.competition_list_files(self.slug)
            files = response.files if hasattr(response, "files") else response
            if files:
                lines += ["## Competition Files", "",
                           "| File | Size | Created |",
                           "|------|------|---------|"]
                for f in files:
                    name = _attr(f, "name", "ref")
                    raw_bytes = _attr(f, "totalBytes", "total_bytes", default=0)
                    created = _attr(f, "creationDate", "creation_date")
                    try:
                        b = int(raw_bytes)
                        size = (f"{b / 1_000_000:.2f} MB" if b > 1_000_000
                                else f"{b / 1_000:.1f} KB" if b > 1_000
                                else f"{b} B")
                    except (ValueError, TypeError):
                        size = str(raw_bytes)
                    lines.append(f"| {name} | {size} | {created} |")
                lines.append("")
        except Exception as exc:
            print(f"  warn: competition_list_files API: {exc}")

        html = self._render_page("data")
        page_md = self._rendered_to_markdown(html)
        if page_md:
            lines.append(page_md)

        self._save("data.md", "\n".join(lines))

    # ── 3. code (only notebooks with a public score) ─────────────────────

    def scrape_code(self):
        print("\n[3/6] Code / Notebooks (with public score)")

        lines = ["# Code Analysis", "",
                 "Notebooks listed below all have a **public leaderboard score**.", ""]
        fetched_from_api = False

        try:
            scored_kernels = []
            page = 1
            while True:
                batch = self.api.kernels_list(
                    competition=self.slug,
                    sort_by="scoreDescending",
                    page=page,
                    page_size=20,
                )
                if not batch:
                    break
                scored_kernels.extend(batch)
                if len(batch) < 20:
                    break
                page += 1
                self._wait()

            if scored_kernels:
                fetched_from_api = True
                lines.append(f"Found {len(scored_kernels)} scored notebooks:\n")
                for i, k in enumerate(scored_kernels, 1):
                    title = _attr(k, "title")
                    author = _attr(k, "author")
                    votes = _attr(k, "total_votes", "totalVotes", default=0)
                    language = _attr(k, "language", default="?")
                    last_run = _attr(k, "last_run_time", "lastRunTime", default="?")
                    ref = _attr(k, "ref", default="")
                    score = _attr(k, "score", "bestPublicScore", "public_score", default="")

                    lines.append(f"### {i}. {title}")
                    if score and score != "N/A":
                        lines.append(f"- **Public Score:** {score}")
                    lines.append(f"- **Author:** {author}")
                    lines.append(f"- **Votes:** {votes}")
                    lines.append(f"- **Language:** {language}")
                    lines.append(f"- **Last run:** {last_run}")
                    if ref and ref != "N/A":
                        lines.append(f"- **Link:** <https://www.kaggle.com/code/{ref}>")
                    lines.append("")
            else:
                lines.append("*No scored notebooks found yet for this competition.*")
        except Exception as exc:
            print(f"  warn: kernels_list API: {exc}")

        if not fetched_from_api:
            html = self._render_page("code")
            page_md = self._rendered_to_markdown(html)
            if page_md:
                lines.append(page_md)

        self._save("codeanalysis.md", "\n".join(lines))

    # ── 4. models ────────────────────────────────────────────────────────

    def scrape_models(self):
        print("\n[4/6] Models")

        lines = ["# Models", ""]

        html = self._render_page("models")
        page_md = self._rendered_to_markdown(html)

        if page_md:
            lines.append(page_md)
        else:
            lines.append("*No model data found on this page.*")

        self._save("models.md", "\n".join(lines))

    # ── 5. discussions ───────────────────────────────────────────────────

    def scrape_discussions(self):
        print("\n[5/6] Discussions (sorted by hotness)")

        lines = ["# Discussions", ""]

        html = self._render_page("discussion?sort=hotness")
        page_md = self._rendered_to_markdown(html)

        if page_md:
            lines.append(page_md)
        else:
            lines.append("*No discussion data found on this page.*")

        self._save("discussions.md", "\n".join(lines))

    # ── 6. rules ─────────────────────────────────────────────────────────

    def scrape_rules(self):
        print("\n[6/6] Rules")

        lines = ["# Competition Rules", ""]

        html = self._render_page("rules")
        page_md = self._rendered_to_markdown(html)

        if page_md:
            lines.append(page_md)
        else:
            lines.append("*No rules content found on this page.*")

        self._save("rules.md", "\n".join(lines))

    # ── run all ──────────────────────────────────────────────────────────

    def scrape_all(self):
        print(f"Competition : {self.slug}")
        print(f"Output dir  : {self.output_dir.resolve()}")
        print(f"Request gap : {self.delay}s")

        try:
            self.scrape_overview()
            self.scrape_data()
            self.scrape_code()
            self.scrape_models()
            self.scrape_discussions()
            self.scrape_rules()
        finally:
            self._close_browser()

        print(f"\nAll files saved to {self.output_dir}/")
        for f in sorted(self.output_dir.glob("*.md")):
            print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a Kaggle competition into Markdown files.",
        epilog="Example: python competitionscraper.py nvidia-nemotron-model-reasoning-challenge",
    )
    parser.add_argument(
        "competition",
        help="Competition URL slug, e.g. 'nvidia-nemotron-model-reasoning-challenge'",
    )
    parser.add_argument(
        "-o", "--output",
        default="competition-data",
        help="Output directory (default: competition-data)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2.0)",
    )
    args = parser.parse_args()

    scraper = CompetitionScraper(args.competition, args.output, delay=args.delay)
    scraper.scrape_all()


if __name__ == "__main__":
    main()
