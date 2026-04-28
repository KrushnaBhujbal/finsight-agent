import re
import requests
import logging
from pydantic import BaseModel
from bs4 import BeautifulSoup

log = logging.getLogger("finsight.tools")

HEADERS = {"User-Agent": "FinSight research@finsight.com"}

class FilingData(BaseModel):
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: str
    raw_text: str
    word_count: int
    sections: dict[str, str]

def get_cik(ticker: str) -> tuple[str, str]:
    log.info(f"Looking up CIK for {ticker}")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            log.info(f"Found CIK {cik} for {ticker} ({entry['title']})")
            return cik, entry["title"]
    raise ValueError(f"CIK not found for ticker: {ticker}")

def get_latest_10k(cik: str) -> tuple[str, str]:
    log.info(f"Fetching submissions for CIK {cik}")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()
    filings = data["filings"]["recent"]
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i]
            date = filings["filingDate"][i]
            log.info(f"Found 10-K filed {date} — accession {accession}")
            return accession, date
    raise ValueError(f"No 10-K found for CIK {cik}")

def get_filing_url(cik: str, accession: str) -> str:
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{accession}-index.htm"
    log.info(f"Fetching filing index")
    response = requests.get(index_url, headers=HEADERS, timeout=15)
    if response.status_code != 200:
        return _find_doc_from_list(cik_int, accession_clean)
    soup = BeautifulSoup(response.text, "html.parser")
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type == "10-K":
                link = cells[2].find("a")
                if link:
                    url = "https://www.sec.gov" + link["href"]
                    if "/ix?doc=" in url:
                        url = "https://www.sec.gov" + url.split("/ix?doc=")[1]
                    log.info(f"Found 10-K document URL")
                    return url
    return _find_doc_from_list(cik_int, accession_clean)

def _find_doc_from_list(cik_int: int, accession_clean: str) -> str:
    list_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/"
    response = requests.get(list_url, headers=HEADERS, timeout=15)
    if response.status_code != 200:
        raise ValueError("Could not find filing document")
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        name = href.split("/")[-1].lower()
        if name.endswith(".htm") and any(k in name for k in ["10k", "10-k", "annual"]):
            return "https://www.sec.gov" + href
    raise ValueError("No suitable document found")

def extract_sections(soup: BeautifulSoup) -> dict[str, str]:
    sections = {}
    full_text = soup.get_text(separator=" ")
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    patterns = {
        "business": [
            r"item\s*1[\.\s]*business\b(.+?)(?=item\s*1a|item\s*2|\Z)",
            r"ITEM\s+1\b(.+?)(?=ITEM\s+1A|ITEM\s+2|\Z)",
        ],
        "risk_factors": [
            r"item\s*1a[\.\s]*risk factors\b(.+?)(?=item\s*1b|item\s*2|\Z)",
            r"ITEM\s+1A\b(.+?)(?=ITEM\s+1B|ITEM\s+2|\Z)",
        ],
        "mda": [
            r"item\s*7[\.\s]*management.{0,30}discussion(.+?)(?=item\s*7a|item\s*8|\Z)",
            r"ITEM\s+7\b(.+?)(?=ITEM\s+7A|ITEM\s+8|\Z)",
        ],
        "financial_highlights": [
            r"item\s*8[\.\s]*financial statements(.+?)(?=item\s*9|\Z)",
        ]
    }

    for section_name, pats in patterns.items():
        for pat in pats:
            match = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1).strip()
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\x20-\x7E]', '', text)
                if len(text) > 500:
                    sections[section_name] = text[:8000]
                    log.info(f"Extracted section '{section_name}': {len(text.split())} words")
                    break

    if not sections:
        log.warning("No sections found via patterns — using cleaned full text")
        clean = re.sub(r'[^\x20-\x7E]', '', full_text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        skip = _find_content_start(clean)
        sections["full_text"] = clean[skip:skip+30000]

    return sections

def _find_content_start(text: str) -> int:
    markers = [
        "PART I", "Part I", "ITEM 1", "Item 1",
        "BUSINESS", "Business Overview",
        "forward-looking statements",
        "Forward-Looking Statements"
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx > 500:
            log.info(f"Content starts at position {idx} (marker: '{marker}')")
            return idx
    return min(2000, len(text) // 10)

def _clean_html(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "meta", "link", "ix:header",
                     "ix:nonfraction", "ix:nonnumeric"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

def fetch_10k(ticker: str) -> FilingData:
    log.info(f"Starting 10-K fetch for {ticker}")
    try:
        cik, company_name = get_cik(ticker)
        accession, date = get_latest_10k(cik)
        doc_url = get_filing_url(cik, accession)

        log.info(f"Fetching document from EDGAR")
        response = requests.get(doc_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        sections = extract_sections(soup)
        clean_text = _clean_html(soup)

        content_start = _find_content_start(clean_text)
        readable_text = clean_text[content_start:content_start + 40000]

        all_section_text = " ".join(sections.values())
        final_text = all_section_text if len(all_section_text) > 1000 else readable_text

        log.info(f"Final text: {len(final_text.split())} words across {len(sections)} sections")

        return FilingData(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            form_type="10-K",
            filing_date=date,
            raw_text=final_text,
            word_count=len(final_text.split()),
            sections=sections
        )

    except Exception as e:
        log.error(f"10-K fetch failed for {ticker}: {e}")
        return FilingData(
            ticker=ticker,
            company_name="Unknown",
            cik="0000000000",
            form_type="10-K",
            filing_date="unknown",
            raw_text=f"Filing unavailable: {str(e)}",
            word_count=0,
            sections={}
        )