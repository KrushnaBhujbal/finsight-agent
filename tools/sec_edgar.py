import re
import requests
import logging
from pydantic import BaseModel

log = logging.getLogger("finsight.tools")

class FilingData(BaseModel):
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: str
    raw_text: str
    word_count: int

HEADERS = {"User-Agent": "FinSight research@finsight.com"}


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


def get_filing_text(cik: str, accession: str) -> str:
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)

    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{accession}-index.htm"
    log.info(f"Fetching filing index: {index_url}")

    response = requests.get(index_url, headers=HEADERS, timeout=15)

    if response.status_code != 200:
        log.warning(f"Index page returned {response.status_code} — trying document list")
        return _fetch_from_document_list(cik_int, accession_clean)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type == "10-K":
                link = cells[2].find("a")
                if link:
                    doc_url = "https://www.sec.gov" + link["href"]
                    log.info(f"Found 10-K document: {doc_url}")
                    return _fetch_and_clean(doc_url)

    log.warning("10-K link not found in index — trying document list")
    return _fetch_from_document_list(cik_int, accession_clean)


def _fetch_from_document_list(cik_int: int, accession_clean: str) -> str:
    list_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/"
    log.info(f"Fetching document list: {list_url}")
    response = requests.get(list_url, headers=HEADERS, timeout=15)

    if response.status_code != 200:
        log.error(f"Document list returned {response.status_code}")
        return "Filing text unavailable."

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    candidates = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        name = href.split("/")[-1].lower()
        if name.endswith(".htm") and any(k in name for k in ["10k", "10-k", "annual"]):
            candidates.append(href)

    if not candidates:
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".htm") and "ex" not in href.lower():
                candidates.append(href)

    if not candidates:
        log.error("No suitable document found")
        return "Filing text unavailable."

    doc_url = "https://www.sec.gov" + candidates[0]
    log.info(f"Trying document: {doc_url}")
    return _fetch_and_clean(doc_url)


def _fetch_and_clean(url: str) -> str:
    # Strip EDGAR inline XBRL viewer wrapper
    if "/ix?doc=" in url:
        url = "https://www.sec.gov" + url.split("/ix?doc=")[1]
        log.info(f"Stripped XBRL wrapper, fetching: {url}")

    response = requests.get(url, headers=HEADERS, timeout=20)
    if response.status_code != 200:
        return "Filing text unavailable."

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    text = text.strip()

    log.info(f"Extracted {len(text.split())} words from filing")
    return text[:20000]


def fetch_10k(ticker: str) -> FilingData:
    log.info(f"Starting 10-K fetch for {ticker}")
    try:
        cik, company_name = get_cik(ticker)
        accession, date = get_latest_10k(cik)
        text = get_filing_text(cik, accession)

        return FilingData(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            form_type="10-K",
            filing_date=date,
            raw_text=text,
            word_count=len(text.split())
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
            word_count=0
        )