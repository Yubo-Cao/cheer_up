import shutil
import zipfile
from pathlib import Path

import aiohttp
import tqdm
from yarl import URL


async def download_extract(url: str, extract_to: Path):
    """Download and extract a zip file."""
    if extract_to.exists():
        print(f"Skipping download of {url} as {extract_to} already exists")
        return

    url = URL(url)
    file = extract_to.parent / url.parts[-1]
    extract_to.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "wb") as f:
        async with (aiohttp.ClientSession() as session, session.get(url) as resp):
            if not resp.ok:
                raise RuntimeError(f"Error downloading {url}: {resp.status}")
            content_length = resp.content_length
            if content_length is None:
                f.write(await resp.read())
            else:
                with tqdm.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {file.stem}",
                    leave=False,
                ) as bar:
                    async for chunk in resp.content.iter_any():
                        f.write(chunk)
                        bar.update(len(chunk))

    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(extract_to.parent)
        shutil.move(extract_to.parent / file.stem, extract_to)
