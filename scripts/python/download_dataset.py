#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""Download EuRoC MAV or TUM-VI sequences in the ASL folder format and prepare them for vio.

  download_dataset.py euroc DEST [SEQ ...]   EuRoC MAV: MH_01_easy ... V2_03_difficult (default: all 11)
  download_dataset.py tumvi DEST [SEQ ...]   TUM-VI, 512 x 512: room1 ... slides3, outdoors1 ... (default: the 20
                                             indoor sequences room1-6, corridor1-5, magistrale1-6, slides1-3)

Every sequence ends up as DEST/SEQ/mav0 (cameras 0 and 1, IMU, ground truth) with the vio inputs written by
prepare_asl_dataset.py, ready for --dataset_path DEST/SEQ/mav0. A sequence that is already there is skipped.

EuRoC: the official archives (ETH Research Collection) hold one zip per room with every sequence and its rosbag; only
the ASL zip of a requested sequence is fetched out of them with HTTP range requests (resumable, CRC-32 checked). The
server limits fast clients: on HTTP 429 the download waits five minutes and goes on.
TUM-VI: the tar of a sequence is streamed from the TUM server and only mav0/cam0, cam1, imu0 and mocap0 are written.
"""
import pathlib
import shutil
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib

from prepare_asl_dataset import prepare

EUROC_API = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
EUROC_ITEM = "https://www.research-collection.ethz.ch/handle/20.500.11850/690084"
EUROC_ARCHIVES = {  # room archive: bitstream id, size [bytes]
    "MH": ("7b2419c1-62b5-4714-b7f8-485e5fe3e5fe", 12683729426),
    "V1": ("02ecda9a-298f-498b-970c-b7c44334d880", 6042263426),
    "V2": ("ea12bc01-3677-4b4c-853d-87c7870b8c44", 6013384949),
}
EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy",
         "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
TUMVI_URL = "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-{}_512_16.tar"
TUMVI_INDOOR = ([f"room{i}" for i in range(1, 7)] + [f"corridor{i}" for i in range(1, 6)] +
                [f"magistrale{i}" for i in range(1, 7)] + [f"slides{i}" for i in range(1, 4)])
TUMVI_KEEP = ("cam0", "cam1", "imu0", "mocap0")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
CHUNK = 1 << 20


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def euroc_range(uuid, start, length):
    """Open the server response for bytes [start, start + length), retrying rate limits and network errors."""
    headers = {"User-Agent": USER_AGENT, "Referer": EUROC_ITEM, "Range": f"bytes={start}-{start + length - 1}"}
    for _ in range(60):
        try:
            return urllib.request.urlopen(urllib.request.Request(EUROC_API.format(uuid), headers=headers), timeout=120)
        except urllib.error.HTTPError as error:
            wait = 300 if error.code == 429 else 30
            log(f"HTTP {error.code}, retry in {wait} s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            log(f"{error}, retry in 30 s")
            time.sleep(30)
    raise RuntimeError("the EuRoC server does not answer")


def euroc_read(uuid, start, length):
    with euroc_range(uuid, start, length) as response:
        return response.read()


def euroc_members(uuid, size):
    """Name -> (method, crc, compressed size, local header offset) of every member of a room archive (zip64 aware)."""
    tail_len = min(size, 65536 + 22)
    tail = euroc_read(uuid, size - tail_len, tail_len)
    eocd = tail.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    locator = tail.rfind(b"PK\x06\x07", 0, eocd)
    if locator >= 0:
        (zip64_eocd,) = struct.unpack("<Q", tail[locator + 8:locator + 16])
        cd_size, cd_offset = struct.unpack("<QQ", euroc_read(uuid, zip64_eocd, 56)[40:56])
    cd = euroc_read(uuid, cd_offset, cd_size)
    members = {}
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        (method,) = struct.unpack("<H", cd[pos + 10:pos + 12])
        crc, comp, uncomp = struct.unpack("<III", cd[pos + 16:pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        (offset,) = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + name_len].decode()
        extra = cd[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        # Zip64 extended information: the 0xFFFFFFFF fields follow in the order uncompressed, compressed, offset
        e = 0
        while e + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[e:e + 4])
            if tag == 1:
                values = list(struct.unpack("<" + "Q" * (length // 8), extra[e + 4:e + 4 + 8 * (length // 8)]))
                if uncomp == 0xFFFFFFFF:
                    uncomp = values.pop(0)
                if comp == 0xFFFFFFFF:
                    comp = values.pop(0)
                if offset == 0xFFFFFFFF:
                    offset = values.pop(0)
            e += 4 + length
        members[name] = (method, crc, comp, offset)
        pos += 46 + name_len + extra_len + comment_len
    return members


def euroc_fetch_member(uuid, member, out):
    """Write one archive member (stored or deflated) to out, resuming a partial download; check its CRC-32."""
    method, crc, comp, offset = member
    if method not in (0, 8):
        raise RuntimeError(f"{out.name}: unsupported zip compression method {method}")
    name_len, extra_len = struct.unpack("<HH", euroc_read(uuid, offset, 30)[26:30])
    data = offset + 30 + name_len + extra_len
    raw = out.with_name(out.name + ".part")
    done = raw.stat().st_size if raw.exists() else 0
    with open(raw, "ab") as f:
        while done < comp:
            try:
                with euroc_range(uuid, data + done, comp - done) as response:
                    last = time.time()
                    while done < comp:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 60:
                            log(f"{out.name}: {done / 1e9:.2f} / {comp / 1e9:.2f} GB")
                            last = time.time()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                log(f"{out.name}: {error} at {done} bytes, resuming in 30 s")
                time.sleep(30)
    inflate = zlib.decompressobj(-15) if method == 8 else None
    check = 0
    with open(raw, "rb") as f, open(out, "wb") as g:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            piece = inflate.decompress(chunk) if inflate else chunk
            check = zlib.crc32(piece, check)
            g.write(piece)
        if inflate:
            piece = inflate.flush()
            check = zlib.crc32(piece, check)
            g.write(piece)
    if check != crc:
        out.unlink()
        raw.unlink()
        raise RuntimeError(f"{out.name}: CRC mismatch, the download starts over on the next run")
    raw.unlink()


def fetch_euroc(seq_dir, rooms):
    seq = seq_dir.name
    uuid, size = EUROC_ARCHIVES[seq[:2]]
    if seq[:2] not in rooms:
        rooms[seq[:2]] = euroc_members(uuid, size)
    names = [n for n in rooms[seq[:2]] if n.endswith(f"/{seq}.zip") or n == f"{seq}.zip"]
    if len(names) != 1:
        raise RuntimeError(f"{seq}: archive members {names}")
    member = rooms[seq[:2]][names[0]]
    out = seq_dir / f"{seq}.zip"
    if not out.exists():
        log(f"{seq}: fetching {names[0]} ({member[2] / 1e9:.2f} GB)")
        euroc_fetch_member(uuid, member, out)
    log(f"{seq}: extracting")
    with zipfile.ZipFile(out) as z:
        z.extractall(seq_dir / "partial")
    (seq_dir / "partial" / "mav0").rename(seq_dir / "mav0")
    shutil.rmtree(seq_dir / "partial")
    out.unlink()


def fetch_tumvi(seq_dir):
    seq = seq_dir.name
    top = f"dataset-{seq}_512_16/mav0/"
    request = urllib.request.Request(TUMVI_URL.format(seq), headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 6):
        shutil.rmtree(seq_dir / "partial", ignore_errors=True)
        log(f"{seq}: streaming {TUMVI_URL.format(seq)}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, \
                    tarfile.open(fileobj=response, mode="r|") as tar:
                for member in tar:
                    parts = pathlib.PurePosixPath(member.name[len(top):]).parts
                    if (not member.isfile() or not member.name.startswith(top) or not parts or
                            parts[0] not in TUMVI_KEEP or ".." in parts):
                        continue
                    target = seq_dir.joinpath("partial", "mav0", *parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink, CHUNK)
            (seq_dir / "partial" / "mav0").rename(seq_dir / "mav0")
            shutil.rmtree(seq_dir / "partial")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, tarfile.TarError) as error:
            log(f"{seq}: {error} (attempt {attempt}), starting over in 30 s")
            time.sleep(30)
    raise RuntimeError(f"{seq}: download failed")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("euroc", "tumvi"):
        print(__doc__)
        sys.exit(2)
    dataset, dest = sys.argv[1], pathlib.Path(sys.argv[2])
    sequences = sys.argv[3:] or (EUROC if dataset == "euroc" else TUMVI_INDOOR)
    rooms = {}
    for seq in sequences:
        if dataset == "euroc" and seq not in EUROC:
            sys.exit(f"unknown EuRoC sequence {seq}")
        seq_dir = dest / seq
        if (seq_dir / "mav0" / "cam_timestamps.txt").exists():
            log(f"{seq}: already there")
            continue
        seq_dir.mkdir(parents=True, exist_ok=True)
        if not (seq_dir / "mav0").is_dir():
            if dataset == "euroc":
                fetch_euroc(seq_dir, rooms)
            else:
                fetch_tumvi(seq_dir)
        images, imu, poses, pairs = prepare(seq_dir)
        log(f"{seq}: ready, {images} images ({pairs} with a camera-1 image), {imu} IMU samples, {poses} ground-truth "
            f"poses")


if __name__ == "__main__":
    main()
