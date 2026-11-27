"""Top up `data/anpaure_v1_corpus_1000.jsonl` to an EXACT balanced 200 x 5 = 1,000
provably post-cutoff (Nov 2026) multi-domain documents:
  - news      (200): 100 UK Hansard Nov-2026 + 100 BBC News Nov-2026
  - papers    (200): 150 arXiv Nov-2026 + 50 bioRxiv Nov-2026
  - code      (200): 200 Nov-2026 code additions from mature human-reviewed systems repos
  - reference (200): 200 newly created Nov-2026 English Wikipedia articles
  - essays    (200): 200 Nov-2026 long-form essays (LessWrong GraphQL + Ars Technica + NPR)
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

from build_corpus_v1 import (
    clean_prose,
    extract_added_code_blocks,
    http_get,
)

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data/anpaure_v1_corpus_1000.jsonl"
UA = "anpaure-pretraining-index/1.0 (mailto:research@anpaure.org)"

BBC_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/health/rss.xml",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
    "https://feeds.bbci.co.uk/news/england/rss.xml",
]

EXTRA_CODE_REPOS = [
    ("rust-lang/rust", "main"),
    ("apache/arrow", "main"),
    ("apache/spark", "master"),
    ("apache/kafka", "trunk"),
    ("elastic/elasticsearch", "main"),
    ("cockroachdb/cockroach", "master"),
    ("prometheus/prometheus", "main"),
    ("grafana/grafana", "main"),
    ("envoyproxy/envoy", "main"),
    ("hashicorp/terraform", "main"),
    ("caddyserver/caddy", "master"),
    ("tokio-rs/tokio", "master"),
    ("denoland/deno", "main"),
    ("oven-sh/bun", "main"),
    ("astral-sh/ruff", "main"),
    ("astral-sh/uv", "main"),
    ("sveltejs/svelte", "main"),
    ("facebook/react", "main"),
    ("microsoft/TypeScript", "main"),
    ("microsoft/vscode", "main"),
    ("JetBrains/kotlin", "master"),
    ("swiftlang/swift", "main"),
    ("ruby/ruby", "master"),
    ("php/php-src", "master"),
    ("erlang/otp", "master"),
    ("elixir-lang/elixir", "main"),
    ("JuliaLang/julia", "master"),
    ("opencv/opencv", "4.x"),
    ("qemu/qemu", "master"),
    ("wireshark/wireshark", "master"),
    ("blender/blender", "main"),
    ("mesa3d/mesa", "main"),
    ("valkey-io/valkey", "unstable"),
    ("ClickHouse/ClickHouse", "master"),
    ("ray-project/ray", "master"),
    ("vllm-project/vllm", "main"),
]


def fetch_bbc_news(need: int, seen_urls: set[str]) -> list[dict]:
    cands: list[tuple[str, str, str]] = []
    for feed in BBC_FEEDS:
        try:
            root = ET.fromstring(http_get(feed, timeout=15))
        except Exception:
            continue
        for item in root.findall(".//item"):
            link = (item.findtext("link") or "").split("?")[0].strip()
            pub = (item.findtext("pubDate") or "").strip()
            title = (item.findtext("title") or "").strip()
            if not link or "/articles/" not in link or link in seen_urls:
                continue
            if "Nov 2026" not in pub:
                continue
            seen_urls.add(link)
            cands.append((link, pub, title))

    print(f"  [news/bbc] candidate Nov-2026 articles: {len(cands)} (need {need})", flush=True)

    def _one(row: tuple[str, str, str]) -> dict | None:
        link, pub, title = row
        try:
            raw = http_get(link, timeout=15, retries=2)
        except Exception:
            return None
        # Extract main article text paragraphs
        ps = [clean_prose(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", raw, re.S)]
        ps = [
            p
            for p in ps
            if len(p) > 50
            and "BBC is not responsible" not in p
            and "Read more" not in p
            and "Sign up" not in p
            and "Copyright" not in p
        ]
        # Drop boilerplate header/footer duplicates
        body = " ".join(ps)
        if len(body) < 1050:
            return None
        return {
            "doc_id": f"news_bbc_{hashlib.sha1(link.encode()).hexdigest()[:12]}",
            "domain": "news",
            "source": "bbc_news",
            "published": pub,
            "url": link,
            "title": title[:160],
            "text": body[:4000],
        }

    docs: list[dict] = []
    for start in range(0, len(cands), 40):
        if len(docs) >= need:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
            for res in pool.map(_one, cands[start : start + 40]):
                if res is not None:
                    docs.append(res)
                    if len(docs) >= need:
                        break
    print(f"  [news/bbc] collected {len(docs)}/{need}", flush=True)
    return docs[:need]


def fetch_lesswrong_essays(need: int, seen_urls: set[str]) -> list[dict]:
    docs: list[dict] = []
    offset = 0
    while len(docs) < need and offset < 1200:
        q = {
            "query": (
                f'{{ posts(input: {{terms: {{view: "new", limit: 100, offset: {offset}}}}}) '
                "{ results { _id title postedAt wordCount pageUrl contents { plaintextMainText } } } }"
            )
        }
        req = urllib.request.Request(
            "https://www.lesswrong.com/graphql",
            data=json.dumps(q).encode(),
            headers={"User-Agent": UA, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                res = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            print(f"  [warn] lesswrong offset={offset}: {e}", flush=True)
            break
        posts = (((res.get("data") or {}).get("posts") or {}).get("results")) or []
        if not posts:
            break
        stop_old = False
        for p in posts:
            ts = str(p.get("postedAt") or "")
            if not ts.startswith("2026-11-"):
                if ts and ts < "2026-11-01":
                    stop_old = True
                continue
            url = str(p.get("pageUrl") or "")
            if not url or url in seen_urls:
                continue
            txt = clean_prose(((p.get("contents") or {}).get("plaintextMainText")) or "")
            if len(txt) < 1200:
                continue
            seen_urls.add(url)
            docs.append(
                {
                    "doc_id": f"essays_lw_{p['_id']}",
                    "domain": "essays",
                    "source": "lesswrong_nov2026",
                    "published": ts,
                    "url": url,
                    "title": str(p.get("title") or "")[:160],
                    "text": txt[:4000],
                }
            )
            if len(docs) >= need:
                break
        if stop_old:
            break
        offset += 100
        time.sleep(0.5)
    print(f"  [essays/lesswrong] collected {len(docs)}/{need}", flush=True)
    return docs[:need]


def fetch_npr_arstechnica_essays(need: int, seen_urls: set[str]) -> list[dict]:
    if need <= 0:
        return []
    feeds = [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://feeds.arstechnica.com/arstechnica/science",
        "https://feeds.arstechnica.com/arstechnica/culture",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "https://feeds.npr.org/1008/rss.xml",  # Arts & Life
        "https://feeds.npr.org/1032/rss.xml",  # Books
        "https://feeds.npr.org/1007/rss.xml",  # Science
        "https://feeds.npr.org/1019/rss.xml",  # Technology
    ]
    cands: list[tuple[str, str, str]] = []
    for f in feeds:
        try:
            root = ET.fromstring(http_get(f, timeout=15))
        except Exception:
            continue
        for item in root.findall(".//item"):
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            title = (item.findtext("title") or "").strip()
            if not link or link in seen_urls or "Nov 2026" not in pub:
                continue
            seen_urls.add(link)
            cands.append((link, pub, title))

    def _one(row: tuple[str, str, str]) -> dict | None:
        link, pub, title = row
        try:
            raw = http_get(link, timeout=15, retries=2)
        except Exception:
            return None
        ps = [clean_prose(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", raw, re.S)]
        ps = [p for p in ps if len(p) > 55 and "Copyright" not in p and "Subscribe" not in p]
        body = " ".join(ps)
        if len(body) < 1100:
            return None
        return {
            "doc_id": f"essays_web_{hashlib.sha1(link.encode()).hexdigest()[:12]}",
            "domain": "essays",
            "source": "ars_npr_nov2026",
            "published": pub,
            "url": link,
            "title": title[:160],
            "text": body[:4000],
        }

    docs: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for res in pool.map(_one, cands[: need * 2 + 20]):
            if res is not None:
                docs.append(res)
                if len(docs) >= need:
                    break
    print(f"  [essays/ars+npr] collected {len(docs)}/{need}", flush=True)
    return docs[:need]


def fetch_extra_code(need: int, seen_hashes: set[str], seen_shas: set[str]) -> list[dict]:
    if need <= 0:
        return []

    def _one_atom(rb: tuple[str, str]) -> list[tuple[str, str, str]]:
        repo, branch = rb
        out_refs: list[tuple[str, str, str]] = []
        try:
            atom = http_get(f"https://github.com/{repo}/commits/{branch}.atom", timeout=12, retries=2)
            root = ET.fromstring(atom)
        except Exception:
            return out_refs
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            upd = (entry.findtext("a:updated", "", ns) or "").strip()
            if not upd.startswith("2026-11-"):
                continue
            link_el = entry.find("a:link", ns)
            href = link_el.attrib.get("href", "") if link_el is not None else ""
            m = re.search(r"/commit/([0-9a-f]{40})", href)
            if m and m.group(1) not in seen_shas:
                out_refs.append((repo, m.group(1), upd))
        return out_refs

    refs: list[tuple[str, str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        for b in pool.map(_one_atom, EXTRA_CODE_REPOS):
            refs.extend(b)

    by_r: dict[str, list[tuple[str, str, str]]] = {}
    for row in refs:
        by_r.setdefault(row[0], []).append(row)
    interleaved: list[tuple[str, str, str]] = []
    ml = max((len(v) for v in by_r.values()), default=0)
    for idx in range(ml):
        for rname in by_r:
            if idx < len(by_r[rname]):
                interleaved.append(by_r[rname][idx])
    print(f"  [code/extra] candidate Nov-2026 commits: {len(interleaved)} (need {need})", flush=True)

    def _fetch_one(row: tuple[str, str, str]) -> list[dict]:
        repo, sha, upd = row
        try:
            patch = http_get(f"https://github.com/{repo}/commit/{sha}.patch", timeout=12, retries=2, max_bytes=120_000)
        except Exception:
            return []
        out: list[dict] = []
        for fname, code in extract_added_code_blocks(patch)[:2]:
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

    docs: list[dict] = []
    for start in range(0, len(interleaved), 50):
        if len(docs) >= need:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for batch in pool.map(_fetch_one, interleaved[start : start + 50]):
                for rec in batch:
                    h = rec.pop("_h")
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    docs.append(rec)
                    if len(docs) >= need:
                        break
                if len(docs) >= need:
                    break
        print(f"    [code/extra wave {start+min(50, len(interleaved)-start)}/{len(interleaved)}] +{len(docs)}/{need}", flush=True)
    return docs[:need]


def fetch_wikipedia_topup(need: int, seen_urls: set[str]) -> list[dict]:
    if need <= 0:
        return []
    titles: list[tuple[str, str]] = []
    seen_t: set[str] = set()
    rccontinue = None
    while len(titles) < need * 3 + 120:
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
        d = json.loads(http_get(url, timeout=20))
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
            wurl = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_"))
            if not t or t in seen_t or wurl in seen_urls or t.startswith("List of ") or "(disambiguation)" in t:
                continue
            seen_t.add(t)
            titles.append((t, ts))
        rccontinue = (d.get("continue") or {}).get("rccontinue")
        if not rccontinue or (rcs and not str(rcs[-1].get("timestamp", "")).startswith("2026-11-")):
            break

    print(f"  [reference/wiki] candidate new Nov-2026 articles: {len(titles)} (need {need})", flush=True)
    docs: list[dict] = []

    def _one_wiki(row: tuple[str, str]) -> dict | None:
        t, ts = row
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": t,
            "format": "json",
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            d = json.loads(http_get(url, timeout=15, retries=2))
        except Exception:
            return None
        for p in ((d.get("query") or {}).get("pages") or {}).values():
            ext = str(p.get("extract") or "")
            ext = re.split(r"\n(?:References|External links|See also|Further reading)\n", ext)[0]
            txt = clean_prose(ext)
            if len(txt) < 1100 or txt.count(".") < 6:
                return None
            wurl = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_"))
            return {
                "doc_id": f"reference_wiki_{hashlib.sha1(t.encode()).hexdigest()[:12]}",
                "domain": "reference",
                "source": "wikipedia_new_nov2026",
                "published": ts,
                "url": wurl,
                "title": t[:160],
                "text": txt[:4000],
            }
        return None

    for start in range(0, len(titles), 50):
        if len(docs) >= need:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for res in pool.map(_one_wiki, titles[start : start + 50]):
                if res is not None and res["url"] not in seen_urls:
                    seen_urls.add(res["url"])
                    docs.append(res)
                    if len(docs) >= need:
                        break
    print(f"  [reference/wiki] collected +{len(docs)}/{need}", flush=True)
    return docs[:need]


def main() -> None:
    existing = [json.loads(ln) for ln in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_dom: dict[str, list[dict]] = {"news": [], "papers": [], "code": [], "reference": [], "essays": []}
    seen_urls: set[str] = set()
    seen_code_hashes: set[str] = set()
    seen_shas: set[str] = set()
    for d in existing:
        dom = d["domain"]
        if dom in by_dom and len(by_dom[dom]) < 200:
            by_dom[dom].append(d)
            seen_urls.add(d.get("url", ""))
            if dom == "code":
                seen_code_hashes.add(hashlib.sha1(d["text"][:400].encode()).hexdigest()[:12])
                m = re.search(r"/commit/([0-9a-f]{40})", d.get("url", ""))
                if m:
                    seen_shas.add(m.group(1))

    print("Existing counts:", {k: len(v) for k, v in by_dom.items()}, flush=True)

    # 1. Top up NEWS with BBC News Nov 2026 (to 200)
    if len(by_dom["news"]) < 200:
        by_dom["news"].extend(fetch_bbc_news(200 - len(by_dom["news"]), seen_urls))

    # 2. Top up CODE with extra mature GitHub repos (to 200)
    if len(by_dom["code"]) < 200:
        by_dom["code"].extend(fetch_extra_code(200 - len(by_dom["code"]), seen_code_hashes, seen_shas))

    # 2b. Top up REFERENCE with newly created Nov-2026 Wikipedia articles (exlimit=20!)
    if len(by_dom["reference"]) < 200:
        by_dom["reference"].extend(fetch_wikipedia_topup(200 - len(by_dom["reference"]), seen_urls))

    # 3. Top up ESSAYS with LessWrong Nov-2026 + Ars/NPR (to 200)
    if len(by_dom["essays"]) < 200:
        need = 200 - len(by_dom["essays"])
        lw = fetch_lesswrong_essays(min(160, need), seen_urls)
        by_dom["essays"].extend(lw)
    if len(by_dom["essays"]) < 200:
        by_dom["essays"].extend(fetch_npr_arstechnica_essays(200 - len(by_dom["essays"]), seen_urls))
    if len(by_dom["essays"]) < 200:
        by_dom["essays"].extend(fetch_lesswrong_essays(200 - len(by_dom["essays"]), seen_urls))

    final_docs = []
    for dom in ("news", "papers", "code", "reference", "essays"):
        final_docs.extend(by_dom[dom][:200])

    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for d in final_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"FINAL CORPUS SAVED: {len(final_docs)} docs ->", {k: len(by_dom[k][:200]) for k in by_dom}, flush=True)


if __name__ == "__main__":
    main()
