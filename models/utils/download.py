import asyncio
import shutil
from pathlib import Path

import aiohttp
import tqdm
from yarl import URL


async def download_extract(
    url: str | URL, extract_to: Path, force_extract: bool = False
) -> None:
    """Download and extract a zip file."""
    if extract_to.exists() and not force_extract:
        print(f"Skipping download of {url} as {extract_to} already exists")
        return

    url = URL(url)
    file = extract_to.parent / url.parts[-1]
    extract_to.parent.mkdir(parents=True, exist_ok=True)

    if not file.exists():
        print(f"Downloading {url} to {file}")
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
    else:
        print(f"Skipping download of {url} as {file} already exists")

    def _unpack():
        print(f"Extracting {file} to {extract_to}")
        shutil.unpack_archive(file, extract_to)
        folder = extract_to / file.stem
        if folder.exists():
            for f in folder.iterdir():
                shutil.move(f, extract_to)
            folder.rmdir()

    def unpack(retry: int = 3):
        try:
            _unpack()
        except Exception as e:
            print(f"Error extracting {file}: {e!r}")
            if retry > 0:
                unpack(retry - 1)
            else:
                raise e

    await asyncio.get_running_loop().run_in_executor(None, unpack)
