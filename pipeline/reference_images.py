"""Reference images: a real photo in place of the generated concept image.

The image model (Z-Image Turbo) only knows what it was trained on, so a
specific product, character or landmark often comes out wrong. Both image-to-3D
models are image-conditioned and remove the background themselves, so a photo
can go straight into reconstruction instead. This module finds one (web image
search), fetches one (a pasted link or a picked search result), or accepts an
upload, and stages it as a clean PNG under config.REFERENCE_IMAGE_DIR until a
job uses it.

HTTP-agnostic like the rest of `pipeline/`: the server and the CLI both call
these functions.
"""

from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from pipeline import config

_REF_ID_RE = re.compile(r"^[0-9a-f]{32}$")
# Many image hosts refuse requests without a browser-like User-Agent.
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
}
_MAX_REDIRECTS = 5
_STAGED_MAX_AGE_S = 7 * 24 * 3600


class ReferenceImageError(ValueError):
    """A reference image couldn't be found, fetched or used. The message is
    meant for the user."""


@dataclass
class ImageSearchResult:
    title: str
    image_url: str
    thumbnail_url: str
    page_url: str
    width: int | None
    height: int | None


# --- Search -------------------------------------------------------------------


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_http_url(url) -> bool:
    return isinstance(url, str) and urlparse(url).scheme in ("http", "https")


def _search_brave(query: str, max_results: int, api_key: str) -> list[ImageSearchResult]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/images/search",
        params={"q": query, "count": min(max_results, 100), "safesearch": "strict"},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=config.REFERENCE_IMAGE_TIMEOUT_S,
    )
    resp.raise_for_status()
    results = []
    for item in resp.json().get("results", []):
        props = item.get("properties") or {}
        image_url = props.get("url")
        if not _is_http_url(image_url):
            continue
        results.append(
            ImageSearchResult(
                title=item.get("title") or "",
                image_url=image_url,
                thumbnail_url=(item.get("thumbnail") or {}).get("src") or image_url,
                page_url=item.get("url") or "",
                width=_int_or_none(props.get("width")),
                height=_int_or_none(props.get("height")),
            )
        )
    return results


def _search_duckduckgo(query: str, max_results: int) -> list[ImageSearchResult]:
    from ddgs import DDGS

    results = []
    for item in DDGS(timeout=config.REFERENCE_IMAGE_TIMEOUT_S).images(
        query, max_results=max_results, safesearch="moderate"
    ):
        image_url = item.get("image")
        if not _is_http_url(image_url):
            continue
        results.append(
            ImageSearchResult(
                title=item.get("title") or "",
                image_url=image_url,
                thumbnail_url=item.get("thumbnail") or image_url,
                page_url=item.get("url") or "",
                width=_int_or_none(item.get("width")),
                height=_int_or_none(item.get("height")),
            )
        )
    return results


def _search_openverse(query: str, max_results: int) -> list[ImageSearchResult]:
    resp = requests.get(
        "https://api.openverse.org/v1/images/",
        params={"q": query, "page_size": min(max_results, 20), "mature": "false"},
        headers={"User-Agent": config.IMAGE_SEARCH_USER_AGENT},
        timeout=config.REFERENCE_IMAGE_TIMEOUT_S,
    )
    resp.raise_for_status()
    results = []
    for item in resp.json().get("results", []):
        image_url = item.get("url")
        if not _is_http_url(image_url):
            continue
        results.append(
            ImageSearchResult(
                title=item.get("title") or "",
                image_url=image_url,
                thumbnail_url=item.get("thumbnail") or image_url,
                page_url=item.get("foreign_landing_url") or "",
                width=_int_or_none(item.get("width")),
                height=_int_or_none(item.get("height")),
            )
        )
    return results


def _search_providers() -> list[tuple[str, Callable[[str, int], list[ImageSearchResult]]]]:
    providers = []
    api_key = os.environ.get(config.BRAVE_SEARCH_API_KEY_ENV)
    if api_key:
        providers.append(("Brave", lambda q, n: _search_brave(q, n, api_key)))
    providers.append(("DuckDuckGo", _search_duckduckgo))
    providers.append(("Openverse", _search_openverse))
    return providers


def search_images(query: str, max_results: int | None = None) -> tuple[str, list[ImageSearchResult]]:
    """Web image search. Returns (provider name, results), trying each
    provider in turn until one returns results."""
    query = query.strip()
    if not query:
        raise ReferenceImageError("Type something to search for.")
    n = max_results or config.IMAGE_SEARCH_MAX_RESULTS
    providers = _search_providers()
    errors = []
    for name, search in providers:
        try:
            results = search(query, n)
        except Exception as exc:  # noqa: BLE001 - any provider failure falls through to the next
            errors.append(f"{name}: {exc}")
            continue
        if results:
            return name, results[:n]
    if len(errors) == len(providers):
        raise ReferenceImageError("Image search failed — " + "; ".join(errors))
    return providers[-1][0], []


# --- Fetch --------------------------------------------------------------------


def _require_public_url(url: str) -> None:
    """Only fetch from the public internet. The server fetches links typed
    into the page, so without this a link could point it at this Mac's own
    services or the local network."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ReferenceImageError("Only http:// and https:// image links are supported.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ReferenceImageError(f"Couldn't find the site {parsed.hostname!r}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise ReferenceImageError("That link points at a local or private address, which isn't allowed.")


def fetch_image_bytes(url: str) -> bytes:
    """Downloads an image, following redirects (each hop re-checked) and
    stopping at config.REFERENCE_IMAGE_MAX_BYTES."""
    for _ in range(_MAX_REDIRECTS + 1):
        _require_public_url(url)
        try:
            resp = requests.get(
                url,
                headers=_FETCH_HEADERS,
                timeout=config.REFERENCE_IMAGE_TIMEOUT_S,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ReferenceImageError(f"Couldn't download the image: {exc}") from exc
        with resp:
            if resp.is_redirect or resp.is_permanent_redirect:
                url = urljoin(url, resp.headers.get("Location", ""))
                continue
            if resp.status_code != 200:
                raise ReferenceImageError(
                    f"The image site refused the download (HTTP {resp.status_code}). Try another image."
                )
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
                raise ReferenceImageError(
                    f"That link is a {content_type} page, not an image. Right-click the picture and copy the image address."
                )
            data = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                data += chunk
                if len(data) > config.REFERENCE_IMAGE_MAX_BYTES:
                    raise ReferenceImageError("That image is too large (over 25MB).")
            return bytes(data)
    raise ReferenceImageError("Too many redirects.")


# --- Normalize + stage --------------------------------------------------------


def normalize_image(data: bytes) -> Image.Image:
    """Decodes any common image format into an upright RGB/RGBA image no
    larger than config.REFERENCE_IMAGE_MAX_SIDE_PX. Re-encoding to PNG later
    also drops metadata (location, camera) from uploaded photos."""
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)  # first frame of an animated GIF/WebP
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ReferenceImageError("That file isn't an image this app can read (use PNG, JPEG or WebP).") from exc
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    if min(img.size) < config.REFERENCE_IMAGE_MIN_SIDE_PX:
        raise ReferenceImageError(
            f"That image is too small ({img.width}×{img.height}px); use one at least "
            f"{config.REFERENCE_IMAGE_MIN_SIDE_PX}px on each side."
        )
    img.thumbnail((config.REFERENCE_IMAGE_MAX_SIDE_PX, config.REFERENCE_IMAGE_MAX_SIDE_PX), Image.LANCZOS)
    return img


def _prune_staged(directory: Path) -> None:
    cutoff = time.time() - _STAGED_MAX_AGE_S
    for path in directory.glob("*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def stage_image_bytes(data: bytes, source: dict, directory: Path | None = None) -> str:
    """Validates and stores an image; returns its reference id."""
    directory = directory or config.REFERENCE_IMAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    _prune_staged(directory)
    img = normalize_image(data)
    ref_id = uuid.uuid4().hex
    img.save(directory / f"{ref_id}.png")
    meta = {**source, "width": img.width, "height": img.height}
    (directory / f"{ref_id}.json").write_text(json.dumps(meta, indent=2))
    return ref_id


def stage_image_url(
    url: str, fallback_url: str | None = None, source: dict | None = None, directory: Path | None = None
) -> str:
    """Fetches and stages an image. `fallback_url` (a search result's
    thumbnail) is tried if the full-size image is blocked or too small."""
    source = {"kind": "url", **(source or {}), "url": url}
    try:
        return stage_image_bytes(fetch_image_bytes(url), source, directory)
    except ReferenceImageError as first:
        if not fallback_url or fallback_url == url:
            raise
        try:
            return stage_image_bytes(
                fetch_image_bytes(fallback_url), {**source, "used_thumbnail": fallback_url}, directory
            )
        except ReferenceImageError:
            raise first from None


def stage_image_file(path: Path, directory: Path | None = None) -> str:
    path = Path(path)
    if not path.is_file():
        raise ReferenceImageError(f"No such image file: {path}")
    if path.stat().st_size > config.REFERENCE_IMAGE_MAX_BYTES:
        raise ReferenceImageError("That image is too large (over 25MB).")
    return stage_image_bytes(path.read_bytes(), {"kind": "file", "filename": path.name}, directory)


def staged_image(ref_id: str, directory: Path | None = None) -> tuple[Path, dict]:
    """The staged PNG for a reference id, plus where it came from."""
    directory = directory or config.REFERENCE_IMAGE_DIR
    if not _REF_ID_RE.match(ref_id or ""):
        raise ReferenceImageError("Unknown reference image.")
    path = directory / f"{ref_id}.png"
    if not path.exists():
        raise ReferenceImageError("That reference image has expired — pick or upload it again.")
    meta_path = directory / f"{ref_id}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return path, meta


def search_result_dicts(results: list[ImageSearchResult]) -> list[dict]:
    return [asdict(r) for r in results]
