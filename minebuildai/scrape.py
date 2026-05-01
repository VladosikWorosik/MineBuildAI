from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup

from .dataset import ManifestRecord, SCHEMATIC_EXTENSIONS, append_jsonl, clean_prompt, file_sha256


DEFAULT_USER_AGENT = "MineBuildAIDatasetBot/0.1 (+https://example.invalid/minebuildai)"


@dataclass(frozen=True)
class ScrapeConfig:
    seed_urls: tuple[str, ...]
    out_dir: Path
    manifest_path: Path
    allowed_domains: frozenset[str]
    max_pages: int = 1000
    concurrency: int = 16
    timeout_seconds: float = 30.0
    per_host_delay_seconds: float = 0.1
    max_file_mb: int = 256
    user_agent: str = DEFAULT_USER_AGENT
    respect_robots: bool = True


@dataclass
class ScrapeStats:
    pages_seen: int = 0
    pages_fetched: int = 0
    files_seen: int = 0
    files_downloaded: int = 0
    files_skipped: int = 0
    failures: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


class RobotsCache:
    def __init__(self, session: aiohttp.ClientSession, user_agent: str) -> None:
        self._session = session
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            lock = self._locks.setdefault(base, asyncio.Lock())
            async with lock:
                if base not in self._cache:
                    self._cache[base] = await self._load(base)
        return self._cache[base].can_fetch(self._user_agent, url)

    async def _load(self, base: str) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = f"{base}/robots.txt"
        parser.set_url(robots_url)
        try:
            async with self._session.get(robots_url) as response:
                if response.status >= 400:
                    parser.parse([])
                    return parser
                text = await response.text(errors="ignore")
                parser.parse(text.splitlines())
        except (aiohttp.ClientError, asyncio.TimeoutError):
            parser.parse([])
        return parser


class HostThrottle:
    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self._next_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, url: str) -> None:
        if self._delay_seconds <= 0:
            return
        host = urlparse(url).netloc.lower()
        async with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_at.get(host, now) - now)
            self._next_at[host] = max(now, self._next_at.get(host, now)) + self._delay_seconds
        if wait_for:
            await asyncio.sleep(wait_for)


def normalize_url(url: str) -> str:
    url, _fragment = urldefrag(url)
    return url.strip()


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def default_allowed_domains(seed_urls: Iterable[str]) -> frozenset[str]:
    return frozenset(domain_of(seed) for seed in seed_urls if domain_of(seed))


def is_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    host = domain_of(url)
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def is_schematic_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(SCHEMATIC_EXTENSIONS)


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name or "schematic"
    raw_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._") or "schematic"
    prefix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{raw_name}"


def prompt_from_download(url: str, source_title: str | None) -> str:
    filename = Path(urlparse(url).path).name
    stem = filename
    for suffix in sorted(SCHEMATIC_EXTENSIONS, key=len, reverse=True):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = [stem]
    if source_title:
        parts.append(source_title)
    return clean_prompt(" ".join(parts))


async def fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    throttle: HostThrottle,
    robots: RobotsCache | None,
) -> tuple[str | None, str | None]:
    if robots and not await robots.can_fetch(url):
        return None, None
    await throttle.wait(url)
    async with session.get(url, allow_redirects=True) as response:
        if response.status >= 400:
            return None, None
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None, None
        return await response.text(errors="ignore"), str(response.url)


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    source_page: str | None,
    source_title: str | None,
    config: ScrapeConfig,
    throttle: HostThrottle,
    robots: RobotsCache | None,
    stats: ScrapeStats,
) -> None:
    if robots and not await robots.can_fetch(url):
        stats.files_skipped += 1
        return

    filename = safe_filename_from_url(url)
    final_path = config.out_dir / filename
    if final_path.exists():
        stats.files_skipped += 1
        return

    temp_path = final_path.with_suffix(final_path.suffix + ".part")
    max_bytes = config.max_file_mb * 1024 * 1024

    try:
        await throttle.wait(url)
        async with session.get(url, allow_redirects=True) as response:
            if response.status >= 400:
                stats.failures += 1
                return
            downloaded = 0
            config.out_dir.mkdir(parents=True, exist_ok=True)
            with temp_path.open("wb") as file:
                async for chunk in response.content.iter_chunked(1024 * 256):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        stats.failures += 1
                        return
                    file.write(chunk)
        temp_path.replace(final_path)
        digest = file_sha256(final_path)
        append_jsonl(
            config.manifest_path,
            ManifestRecord(
                prompt=prompt_from_download(url, source_title),
                path=str(final_path),
                sha256=digest,
                bytes=final_path.stat().st_size,
                source_url=url,
                source_page=source_page,
            ),
        )
        stats.files_downloaded += 1
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        stats.failures += 1
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def extract_links(html: str, base_url: str) -> tuple[str | None, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    links: list[str] = []
    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href")
        if not href:
            continue
        absolute = normalize_url(urljoin(base_url, href))
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"}:
            links.append(absolute)
    return title, links


async def scrape(config: ScrapeConfig) -> ScrapeStats:
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    headers = {"user-agent": config.user_agent}
    connector = aiohttp.TCPConnector(limit=config.concurrency, ttl_dns_cache=300)
    stats = ScrapeStats()
    queue: asyncio.Queue[str] = asyncio.Queue()
    queued_pages: set[str] = set()
    seen_files: set[str] = set()
    download_tasks: set[asyncio.Task[None]] = set()
    state_lock = asyncio.Lock()

    normalized_seeds = tuple(normalize_url(seed) for seed in config.seed_urls)
    for seed in normalized_seeds:
        if is_schematic_url(seed):
            continue
        queued_pages.add(seed)
        await queue.put(seed)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        robots = RobotsCache(session, config.user_agent) if config.respect_robots else None
        throttle = HostThrottle(config.per_host_delay_seconds)
        download_sem = asyncio.Semaphore(config.concurrency)

        async def schedule_download(url: str, source_page: str | None, source_title: str | None) -> None:
            async with state_lock:
                if url in seen_files:
                    return
                seen_files.add(url)
                stats.files_seen += 1

            async def bounded_download() -> None:
                async with download_sem:
                    await download_file(session, url, source_page, source_title, config, throttle, robots, stats)

            task = asyncio.create_task(bounded_download())
            download_tasks.add(task)
            task.add_done_callback(download_tasks.discard)

        async def enqueue_page(url: str) -> None:
            async with state_lock:
                if len(queued_pages) >= config.max_pages or url in queued_pages:
                    return
                queued_pages.add(url)
                await queue.put(url)

        async def worker() -> None:
            while True:
                page_url = await queue.get()
                try:
                    async with state_lock:
                        stats.pages_seen += 1
                    html, final_url = await fetch_text(session, page_url, throttle, robots)
                    if not html or not final_url:
                        continue
                    stats.pages_fetched += 1
                    title, links = extract_links(html, final_url)
                    for link in links:
                        if not is_allowed(link, config.allowed_domains):
                            continue
                        if is_schematic_url(link):
                            await schedule_download(link, final_url, title)
                        else:
                            await enqueue_page(link)
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    stats.failures += 1
                finally:
                    queue.task_done()

        for seed in normalized_seeds:
            if is_schematic_url(seed):
                await schedule_download(seed, None, None)

        workers = [asyncio.create_task(worker()) for _ in range(config.concurrency)]
        await queue.join()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if download_tasks:
            await asyncio.gather(*download_tasks, return_exceptions=True)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast respectful schematic crawler/downloader.")
    parser.add_argument("--seed", action="append", required=True, help="Seed page URL. Can be passed multiple times.")
    parser.add_argument("--out", default="data/raw", help="Directory for downloaded schematic files.")
    parser.add_argument("--manifest", default="data/manifest.jsonl", help="JSONL manifest output path.")
    parser.add_argument("--allowed-domain", action="append", default=[], help="Allowed crawl domain. Defaults to seed hosts.")
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--per-host-delay", type=float, default=0.1)
    parser.add_argument("--max-file-mb", type=int, default=256)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--ignore-robots", action="store_true", help="Disable robots.txt checks. Only use where you have permission.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ScrapeConfig:
    seeds = tuple(normalize_url(seed) for seed in args.seed)
    allowed = frozenset(args.allowed_domain) if args.allowed_domain else default_allowed_domains(seeds)
    return ScrapeConfig(
        seed_urls=seeds,
        out_dir=Path(args.out),
        manifest_path=Path(args.manifest),
        allowed_domains=allowed,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        per_host_delay_seconds=args.per_host_delay,
        max_file_mb=args.max_file_mb,
        user_agent=args.user_agent,
        respect_robots=not args.ignore_robots,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = asyncio.run(scrape(config_from_args(args)))
    rate = stats.files_downloaded / max(stats.elapsed_seconds, 0.001)
    print(
        "Downloaded "
        f"{stats.files_downloaded}/{stats.files_seen} files, "
        f"fetched {stats.pages_fetched}/{stats.pages_seen} pages, "
        f"skipped {stats.files_skipped}, failures {stats.failures}, "
        f"{rate:.2f} files/sec."
    )


if __name__ == "__main__":
    main()
