"""Build the downloads of the flare list feedback page (index.html) in this folder.

- ``STIX_L3_flarelist_showcase.html``: the executed notebook as a static page (not re-executed),
  marked as not to be indexed by search engines;
- ``STIX_L3_flarelist_showcase_with_data.zip``: notebook, requirements.txt and the three data files;
- ``cover.png``: the cover picture of index.html, taken from the notebook output (disc views, section 5.5).

Also checks that every package of requirements.txt is listed on index.html, and prints the file
sizes for index.html.

Run from anywhere:  python build_feedback_bundle.py
"""

import re
import sys
import json
import base64
import zipfile
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "STIX_L3_flarelist_showcase.ipynb"
HTML = HERE / "STIX_L3_flarelist_showcase.html"
ZIP = HERE / "STIX_L3_flarelist_showcase_with_data.zip"
REQUIREMENTS = HERE / "requirements.txt"
INDEX = HERE / "index.html"
COVER = HERE / "cover.png"
COVER_CELL_ID = "bd52ef90"  # notebook cell with the disc views (section 5.5)
DATA_FILES = [
    "solo_L3_stix-flarelist-sdcloc_20230501T000000-20230601T011216_V05.fits",  # month 2023-05
    "solo_L3_stix-flarelist-sdcloc_20230101T000000-20231231T230447_V05.fits",  # year 2023
    "solo_L3_stix-flarelist-sdcloc_20220101T000000-20260831T232151_V05.fits",  # mission
]
ROBOTS = '<meta name="robots" content="noindex, nofollow, noarchive">'
ZIP_FOLDER = "STIX_L3_flarelist_showcase"  # top folder inside the zip


def size(path):
    return (
        f"{path.stat().st_size / 1e6:.0f} MB" if path.stat().st_size >= 1e6 else f"{path.stat().st_size / 1e3:.0f} kB"
    )


def export_html():
    # the saved notebook already has its outputs: export only, no execution
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(NOTEBOOK), "--output", HTML.stem],
        check=True,
        cwd=HERE,
    )
    page = HTML.read_text(encoding="utf-8")
    if ROBOTS not in page:
        page = page.replace("<head>", f"<head>\n{ROBOTS}", 1)
        HTML.write_text(page, encoding="utf-8")


def export_cover():
    # the figure of the disc view cell, as saved in the executed notebook
    cell = next(c for c in json.loads(NOTEBOOK.read_text())["cells"] if c.get("id") == COVER_CELL_ID)
    png = next(o["data"]["image/png"] for o in cell["outputs"] if "image/png" in o.get("data", {}))
    COVER.write_bytes(base64.b64decode(png))


def build_zip():
    missing = [name for name in DATA_FILES if not (HERE / name).exists()]
    if missing:
        sys.exit(f"data files missing in {HERE}: {missing}")
    tmp = ZIP.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in [NOTEBOOK, REQUIREMENTS] + [HERE / name for name in DATA_FILES]:
            z.write(path, f"{ZIP_FOLDER}/{path.name}")
    tmp.replace(ZIP)


def check_requirements_listed():
    # every package of requirements.txt must appear on index.html (the "how to get started" list)
    packages = [
        re.split(r"[<>=~!\[ ]", line.strip(), maxsplit=1)[0]
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    page = INDEX.read_text(encoding="utf-8")
    missing = [p for p in packages if p not in page]
    if missing:
        sys.exit(f"packages of requirements.txt not listed on index.html: {missing}")
    print(f"requirements listed on index.html: {', '.join(packages)}")


if __name__ == "__main__":
    check_requirements_listed()
    export_html()
    export_cover()
    build_zip()
    for path in [HTML, COVER, NOTEBOOK, ZIP, REQUIREMENTS] + [HERE / name for name in DATA_FILES]:
        print(f"{size(path):>8}  {path.name}")
