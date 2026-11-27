"""Build the Anpaure Pretraining Index v1 corpus: N=1,000 provably post-cutoff
(Nov 2026) documents balanced evenly across 5 domains matching Elie Bakouch's
Uncheatable Eval (200 docs each):

  1. news      (200): 100 Nov-2026 UK Hansard live human speeches + 100 Nov-2026 Guardian news articles
  2. papers    (200): 150 Nov-2026 arXiv papers (cs/math/physics/stat/q-fin) + 50 Nov-2026 bioRxiv papers
  3. code      (200): 200 Nov-2026 human-reviewed code additions from mature open-source systems repos
  4. reference (200): 200 newly created Nov-2026 English Wikipedia articles
  5. essays    (200): 200 Nov-2026 Guardian Culture / Books / Long Read / Film / Opinion / Literary prose

Every document is strictly verified to have a Nov-2026 timestamp and clean
non-boilerplate content. Output: data/anpaure_v1_corpus_1000.jsonl
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import html as html_lib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data/anpaure_v1_corpus_1000.jsonl"
UA = "anpaure-pretraining-index/1.0 (mailto:research@anpaure.org)"


def http_get(url: str, timeout: int = 15, retries: int = 2, max_bytes: int = 0) -> str:
    headers = {"User-Agent": UA, "Accept": "*/*"}
    for att in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(max_bytes) if max_bytes > 0 else r.read()
                return raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and att + 1 < retries:
                time.sleep(1.5 * (att + 1))
                continue
            raise
        except Exception:
            if att + 1 < retries:
                time.sleep(1.0 * (att + 1))
                continue
            raise
    raise RuntimeError(f"Failed GET {url}")


def clean_prose(text: str) -> str:
    text = html_lib.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_guardian_article(url: str) -> str | None:
    try:
        raw = http_get(url, timeout=20, retries=2)
    except Exception:
        return None
    paras = re.findall(r'<p class="dcr-[^"]+">(.*?)</p>', raw, re.S)
    if len(paras) < 4:
        return None
    cleaned = [clean_prose(p) for p in paras]
    cleaned = [p for p in cleaned if len(p) > 40 and not p.startswith("Sign up to ") and not p.startswith("Photograph:")]
    body = " ".join(cleaned)
    return body if len(body) >= 1000 else None


def fetch_guardian_feeds(feeds: list[str], target_n: int, domain: str, seen_urls: set[str]) -> list[dict]:
    cand_items: list[tuple[str, str, str]] = []
    for feed in feeds:
        try:
            xml_s = http_get(feed, timeout=20)
            root = ET.fromstring(xml_s)
        except Exception as e:
            print(f"  [warn] feed failed {feed}: {e}")
            continue
        for item in root.findall(".//item"):
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            title = (item.findtext("title") or "").strip()
            if not link or link in seen_urls:
                continue
            if "Nov 2026" not in pub and "/2026/nov/" not in link:
                continue
            if any(x in link for x in ("/live/", "/gallery/", "/video/", "/audio/", "/crosswords/", "/ng-interactive/")):
                continue
            seen_urls.add(link)
            cand_items.append((link, pub, title))

    print(f"  [{domain}] guardian candidate URLs: {len(cand_items)} (need {target_n})")
    docs: list[dict] = []

    def _one(row: tuple[str, str, str]) -> dict | None:
        link, pub, title = row
        body = fetch_guardian_article(link)
        if not body:
            return None
        return {
            "doc_id": f"{domain}_guardian_{hashlib.sha1(link.encode()).hexdigest()[:12]}",
            "domain": domain,
            "source": "guardian",
            "published": pub,
            "url": link,
            "title": title[:160],
            "text": body[:4000],
        }

    # Fetch in small waves so we stop immediately upon hitting target_n
    for start in range(0, len(cand_items), 40):
        if len(docs) >= target_n:
            break
        wave = cand_items[start : start + 40]
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
            for res in pool.map(_one, wave):
                if res is not None:
                    docs.append(res)
                    if len(docs) >= target_n:
                        break
    return docs[:target_n]


def fetch_hansard_nov2026(target_n: int) -> list[dict]:
    docs: list[dict] = []
    seen_ids: set[str] = set()
    skip = 0
    while len(docs) < target_n and skip < 2000:
        url = (
            "https://hansard-api.parliament.uk/search/contributions/spoken.json"
            f"?queryParameters.startDate=2026-11-01&queryParameters.endDate=2026-11-25"
            f"&queryParameters.take=100&queryParameters.skip={skip}"
        )
        try:
            d = json.loads(http_get(url, timeout=25))
        except Exception as e:
            print(f"  [warn] hansard skip={skip}: {e}")
            break
        results = d.get("Results") or []
        if not results:
            break
        for r in results:
            ext_id = str(r.get("ContributionExtId") or "")
            if not ext_id or ext_id in seen_ids:
                continue
            txt = clean_prose(r.get("ContributionTextFull") or r.get("ContributionText") or "")
            if len(txt) < 1050:
                continue
            # Avoid procedural boilerplate
            if txt.startswith("I beg to move") and len(txt) < 1300:
                continue
            seen_ids.add(ext_id)
            docs.append(
                {
                    "doc_id": f"news_hansard_{ext_id[:16]}",
                    "domain": "news",
                    "source": "uk_hansard",
                    "published": str(r.get("SittingDate") or "2026-11"),
                    "url": f"https://hansard.parliament.uk/search/Contributions?searchTerm={ext_id}",
                    "title": str(r.get("DebateSection") or "")[:160],
                    "text": txt[:4000],
                }
            )
            if len(docs) >= target_n:
                break
        skip += 100
    print(f"  [news/hansard] collected {len(docs)}/{target_n}")
    return docs


def fetch_papers_nov2026(target_arxiv: int = 150, target_biorxiv: int = 50) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    # 1. arXiv across 5 distinct fields (30 each)
    cats = [
        ("cs.LG+OR+cs.AI+OR+cs.CL+OR+cs.CV", 45),
        ("math.NT+OR+math.PR+OR+math.CO+OR+math.AP+OR+math.AG", 45),
        ("quant-ph+OR+hep-th+OR+cond-mat.mes-hall+OR+astro-ph.CO", 45),
        ("stat.ML+OR+stat.ME+OR+econ.EM+OR+q-fin.ST", 35),
    ]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for query, want in cats:
        got = 0
        url = (
            f"https://export.arxiv.org/api/query?search_query=cat:{query}"
            "&sortBy=submittedDate&sortOrder=descending&max_results=120"
        )
        try:
            root = ET.fromstring(http_get(url, timeout=30))
        except Exception as e:
            print(f"  [warn] arxiv {query}: {e}")
            continue
        for entry in root.findall("a:entry", ns):
            eid = (entry.findtext("a:id", "", ns) or "").strip()
            pub = (entry.findtext("a:published", "", ns) or "").strip()
            if not pub.startswith("2026-11-") or eid in seen:
                continue
            title = clean_prose(entry.findtext("a:title", "", ns) or "")
            summary = clean_prose(entry.findtext("a:summary", "", ns) or "")
            # Combine title + abstract; require substantial length (>= 950 chars)
            full_txt = f"{title}. {summary}"
            if len(full_txt) < 950:
                continue
            seen.add(eid)
            docs.append(
                {
                    "doc_id": f"papers_arxiv_{hashlib.sha1(eid.encode()).hexdigest()[:12]}",
                    "domain": "papers",
                    "source": "arxiv",
                    "published": pub,
                    "url": eid,
                    "title": title[:160],
                    "text": full_txt[:4000],
                }
            )
            got += 1
            if got >= want or len(docs) >= target_arxiv:
                break
        time.sleep(1.0)
        if len(docs) >= target_arxiv:
            break
    print(f"  [papers/arxiv] collected {len(docs)}/{target_arxiv}")

    # 2. bioRxiv Nov 15-25, 2026
    bio_got = 0
    cursor = 0
    while bio_got < target_biorxiv and cursor < 600:
        url = f"https://api.biorxiv.org/details/biorxiv/2026-11-10/2026-11-25/{cursor}"
        try:
            d = json.loads(http_get(url, timeout=30))
        except Exception as e:
            print(f"  [warn] biorxiv cursor={cursor}: {e}")
            break
        coll = d.get("collection") or []
        if not coll:
            break
        for item in coll:
            doi = str(item.get("doi") or "")
            if not doi or doi in seen:
                continue
            title = clean_prose(str(item.get("title") or ""))
            abst = clean_prose(str(item.get("abstract") or ""))
            full_txt = f"{title}. {abst}"
            if len(full_txt) < 1000:
                continue
            seen.add(doi)
            docs.append(
                {
                    "doc_id": f"papers_biorxiv_{hashlib.sha1(doi.encode()).hexdigest()[:12]}",
                    "domain": "papers",
                    "source": "biorxiv",
                    "published": str(item.get("date") or "2026-11"),
                    "url": f"https://doi.org/{doi}",
                    "title": title[:160],
                    "text": full_txt[:4000],
                }
            )
            bio_got += 1
            if bio_got >= target_biorxiv:
                break
        cursor += len(coll)
    print(f"  [papers/biorxiv] collected {bio_got}/{target_biorxiv}")
    return docs


def fetch_wikipedia_new_nov2026(target_n: int = 200) -> list[dict]:
    titles: list[tuple[str, str]] = []
    seen_t: set[str] = set()
    rccontinue = None
    while len(titles) < target_n * 3:
        params = {
            "action": "query",
            "list": "recentchanges",
            "rctype": "new",
            "rcnamespace": "0",
            "rclimit": "500",
            "rcprop": "title|timestamp|sizes",
            "rcshow": "!bot|!redirect",
            "format": "json",
        }
        if rccontinue:
            params["rccontinue"] = rccontinue
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        d = json.loads(http_get(url, timeout=25))
        rcs = (d.get("query") or {}).get("recentchanges") or []
        if not rcs:
            break
        for rc in rcs:
            ts = str(rc.get("timestamp") or "")
            if not ts.startswith("2026-11-"):
                continue
            if int(rc.get("newlen") or 0) < 3500:
                continue
            t = str(rc.get("title") or "")
            if not t or t in seen_t or t.startswith("List of ") or "(disambiguation)" in t:
                continue
            seen_t.add(t)
            titles.append((t, ts))
        rccontinue = (d.get("continue") or {}).get("rccontinue")
        if not rccontinue or (rcs and not str(rcs[-1].get("timestamp", "")).startswith("2026-11-")):
            break

    print(f"  [reference/wiki] candidate new Nov-2026 articles: {len(titles)}")
    docs: list[dict] = []
    # Fetch plain-text extracts in batches of 20
    for i in range(0, len(titles), 20):
        batch = titles[i : i + 20]
        ts_map = {t: ts for t, ts in batch}
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": "|".join(t for t, _ in batch),
            "format": "json",
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            d = json.loads(http_get(url, timeout=25))
        except Exception as e:
            print(f"  [warn] wiki batch {i}: {e}")
            continue
        pages = ((d.get("query") or {}).get("pages") or {}).values()
        for p in pages:
            t = str(p.get("title") or "")
            ext = str(p.get("extract") or "")
            # Drop References / External links tails
            ext = re.split(r"\n(?:References|External links|See also|Further reading)\n", ext)[0]
            txt = clean_prose(ext)
            if len(txt) < 1100:
                continue
            # Ensure rich prose, not tables/lists
            if txt.count(".") < 6:
                continue
            docs.append(
                {
                    "doc_id": f"reference_wiki_{hashlib.sha1(t.encode()).hexdigest()[:12]}",
                    "domain": "reference",
                    "source": "wikipedia_new_nov2026",
                    "published": ts_map.get(t, "2026-11"),
                    "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                    "title": t[:160],
                    "text": txt[:4000],
                }
            )
            if len(docs) >= target_n:
                break
        if len(docs) >= target_n:
            break
    print(f"  [reference/wiki] collected {len(docs)}/{target_n}")
    return docs


CODE_REPOS = [
    ("python/cpython", "main"),
    ("rust-lang/rust", "master"),
    ("golang/go", "master"),
    ("postgres/postgres", "master"),
    ("llvm/llvm-project", "main"),
    ("git/git", "master"),
    ("FFmpeg/FFmpeg", "master"),
    ("nodejs/node", "main"),
    ("numpy/numpy", "main"),
    ("scipy/scipy", "main"),
    ("django/django", "main"),
    ("pytorch/pytorch", "main"),
    ("neovim/neovim", "master"),
    ("zed-industries/zed", "main"),
    ("torvalds/linux", "master"),
    ("systemd/systemd", "main"),
    ("duckdb/duckdb", "main"),
    ("ClickHouse/ClickHouse", "master"),
    ("kubernetes/kubernetes", "master"),
    ("godotengine/godot", "master"),
    ("ziglang/zig", "master"),
    ("servo/servo", "main"),
    ("redis/redis", "unstable"),
    ("curl/curl", "master"),
    ("openssl/openssl", "master"),
    ("pandas-dev/pandas", "main"),
    ("scikit-learn/scikit-learn", "main"),
    ("fastapi/fastapi", "master"),
]

VALID_CODE_EXT = (".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".js", ".ts", ".zig")


def extract_added_code_blocks(patch_text: str) -> list[tuple[str, str]]:
    """Extract contiguous newly added source-code blocks (>= 16 non-empty lines, >= 850 chars)
    from a Nov-2026 git commit patch. Strips diff '+' prefix cleanly."""
    blocks: list[tuple[str, str]] = []
    cur_file = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_lines
        if not cur_file or not cur_lines:
            cur_lines = []
            return
        if not cur_file.endswith(VALID_CODE_EXT):
            cur_lines = []
            return
        low = cur_file.lower()
        if any(x in low for x in ("test", "fixture", "vendor", "generated", "third_party", "lock", "changelog", "docs/")):
            cur_lines = []
            return
        code = "\n".join(cur_lines).strip("\n")
        non_empty = [ln for ln in cur_lines if ln.strip()]
        if len(non_empty) >= 16 and len(code) >= 900:
            # Avoid huge comment/license headers or giant data tables
            alpha_ratio = sum(ch.isalpha() for ch in code) / max(1, len(code))
            if 0.30 <= alpha_ratio <= 0.85:
                blocks.append((cur_file, code[:3500]))
        cur_lines = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            m = re.search(r" b/(.+)$", line)
            cur_file = m.group(1).strip() if m else ""
        elif line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@ "):
            if line.startswith("@@ "):
                flush()
        elif line.startswith("+"):
            cur_lines.append(line[1:])
        else:
            # Context or deleted line breaks a contiguous added block
            if len(cur_lines) >= 16:
                flush()
            else:
                cur_lines = []
    flush()
    return blocks


def fetch_code_nov2026(target_n: int = 200) -> list[dict]:
    def _one_atom(rb: tuple[str, str]) -> list[tuple[str, str, str]]:
        repo, branch = rb
        out_refs: list[tuple[str, str, str]] = []
        try:
            atom = http_get(f"https://github.com/{repo}/commits/{branch}.atom", timeout=12, retries=2)
            root = ET.fromstring(atom)
        except Exception as e:
            print(f"  [warn] atom {repo}: {e}", flush=True)
            return out_refs
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            upd = (entry.findtext("a:updated", "", ns) or "").strip()
            if not upd.startswith("2026-11-"):
                continue
            link_el = entry.find("a:link", ns)
            href = link_el.attrib.get("href", "") if link_el is not None else ""
            m = re.search(r"/commit/([0-9a-f]{40})", href)
            if m:
                out_refs.append((repo, m.group(1), upd))
        return out_refs

    commit_refs: list[tuple[str, str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for batch in pool.map(_one_atom, CODE_REPOS):
            commit_refs.extend(batch)

    print(f"  [code] candidate Nov-2026 commits across {len(CODE_REPOS)} repos: {len(commit_refs)}")
    docs: list[dict] = []
    seen_hashes: set[str] = set()

    def _fetch_one(row: tuple[str, str, str]) -> list[dict]:
        repo, sha, upd = row
        try:
            # Cap at 120 KB per commit patch so huge refactors never stall
            patch = http_get(f"https://github.com/{repo}/commit/{sha}.patch", timeout=12, retries=2, max_bytes=120_000)
        except Exception:
            return []
        out: list[dict] = []
        for fname, code in extract_added_code_blocks(patch)[:2]:  # max 2 blocks per commit for diversity
            h = hashlib.sha1(code[:400].encode()).hexdigest()[:12]
            out.append(
                {
                    "doc_id": f"code_{repo.replace('/', '_')}_{sha[:8]}_{h[:6]}",
                    "domain": "code",
                    "source": f"github:{repo}",
                    "published": upd,
                    "url": f"https://github.com/{repo}/commit/{sha}",
                    "title": f"{repo}:{fname}"[:160],
                    "text": code,
                    "_h": h,
                }
            )
        return out

    # Interleave commits across repos so we get diverse languages immediately!
    by_repo: dict[str, list[tuple[str, str, str]]] = {}
    for row in commit_refs:
        by_repo.setdefault(row[0], []).append(row)
    interleaved: list[tuple[str, str, str]] = []
    max_len = max((len(v) for v in by_repo.values()), default=0)
    for idx in range(max_len):
        for rname in by_repo:
            if idx < len(by_repo[rname]):
                interleaved.append(by_repo[rname][idx])

    for start in range(0, len(interleaved), 50):
        if len(docs) >= target_n:
            break
        wave = interleaved[start : start + 50]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for batch in pool.map(_fetch_one, wave):
                for rec in batch:
                    h = rec.pop("_h")
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    docs.append(rec)
                    if len(docs) >= target_n:
                        break
                if len(docs) >= target_n:
                    break
        print(f"    [code wave {start+len(wave)}/{len(interleaved)}] collected {len(docs)}/{target_n}", flush=True)
    print(f"  [code] collected {len(docs)}/{target_n}", flush=True)
    return docs[:target_n]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen_guardian: set[str] = set()

    print("=== 1/5 NEWS & LIVE SPEECH (200) ===")
    hansard = fetch_hansard_nov2026(100)
    g_news = fetch_guardian_feeds(
        [
            "https://www.theguardian.com/world/rss",
            "https://www.theguardian.com/uk-news/rss",
            "https://www.theguardian.com/us-news/rss",
            "https://www.theguardian.com/business/rss",
            "https://www.theguardian.com/technology/rss",
            "https://www.theguardian.com/science/rss",
            "https://www.theguardian.com/environment/rss",
        ],
        target_n=100,
        domain="news",
        seen_urls=seen_guardian,
    )
    news_docs = hansard + g_news
    print(f"  => NEWS total: {len(news_docs)}")

    print("\n=== 2/5 ACADEMIC PAPERS (200) ===")
    paper_docs = fetch_papers_nov2026(150, 50)
    print(f"  => PAPERS total: {len(paper_docs)}")

    print("\n=== 3/5 HUMAN-REVIEWED SYSTEMS CODE (200) ===")
    code_docs = fetch_code_nov2026(200)
    print(f"  => CODE total: {len(code_docs)}")

    print("\n=== 4/5 NEW WIKIPEDIA REFERENCE ARTICLES (200) ===")
    ref_docs = fetch_wikipedia_new_nov2026(200)
    print(f"  => REFERENCE total: {len(ref_docs)}")

    print("\n=== 5/5 LONG-FORM ESSAYS, CULTURE & LITERARY PROSE (200) ===")
    essay_docs = fetch_guardian_feeds(
        [
            "https://www.theguardian.com/books/rss",
            "https://www.theguardian.com/culture/rss",
            "https://www.theguardian.com/news/series/the-long-read/rss",
            "https://www.theguardian.com/uk/commentisfree/rss",
            "https://www.theguardian.com/film/rss",
            "https://www.theguardian.com/music/rss",
            "https://www.theguardian.com/artanddesign/rss",
            "https://www.theguardian.com/stage/rss",
            "https://www.theguardian.com/lifeandstyle/rss",
            "https://www.theguardian.com/society/rss",
            "https://www.theguardian.com/global-development/rss",
            "https://www.theguardian.com/sport/rss",
        ],
        target_n=200,
        domain="essays",
        seen_urls=seen_guardian,
    )
    print(f"  => ESSAYS total: {len(essay_docs)}")

    all_docs = news_docs + paper_docs + code_docs + ref_docs + essay_docs
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    counts = {}
    for d in all_docs:
        counts[d["domain"]] = counts.get(d["domain"], 0) + 1
    print(f"\nSAVED {len(all_docs)} docs to {OUT_PATH}: {counts}")


if __name__ == "__main__":
    main()
