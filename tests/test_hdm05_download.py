import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


subset = module("download_hdm05_subset")
walking = module("download_hdm05_walking")
BVH = b'''HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
}
MOTION
Frames: 2
Frame Time: 0.00833333
0 10 0 0 0 0
1 10 2 0 0 0
'''


def row_for(payload):
    return {"path": "train/9/test.amc", "bytes": len(payload),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()}


def test_amc_raw_hash_and_root_foot_channels():
    payload = b"# M\xfcller\n" + b"".join(
        f"{i}\nroot {i} 0 0 0 0 0\nlfoot 0 0\nrfoot 0 0\nltoes 0\nrtoes 0\n".encode()
        for i in (1, 2))
    assert subset.validate_payload(payload, row_for(payload))["frames"] == 2
    with pytest.raises(ValueError, match="hash"):
        subset.validate_payload(payload + b"x", row_for(payload))
    missing = payload.replace(b"rtoes 0\n", b"")
    with pytest.raises(ValueError, match="bilateral"):
        subset.validate_payload(missing, row_for(missing))


def test_bvh_translation_and_frame_validation():
    result = walking.inspect_bvh(BVH)
    assert result["frames"] == 2
    assert result["root_translation_span_raw_units"] == [1, 0, 2]
    with pytest.raises(ValueError, match="count mismatch"):
        walking.inspect_bvh(BVH.replace(b"Frames: 2", b"Frames: 3"))


def test_only_requested_hdm05_actions():
    infos = [zipfile.ZipInfo(path) for path in (
        "data/hdm05/HDM_bd_walkLeft2Steps_001_120.bvh",
        "data/hdm05/HDM_bd_jogLeftCircle4StepsRstart_001_120.bvh",
        "data/cmu/HDM_bd_walkLeft2Steps_001_120.bvh")]
    assert walking.choose_members(infos) == [infos[0]]


def test_range_reads_cache_and_detect_invalid_response(monkeypatch):
    data = b"abcdefghi"
    requests = []

    class Response(io.BytesIO):
        def __init__(self, request):
            requests.append(request)
            self.headers = {"Content-Length": str(len(data)), "ETag": '"fixture"'}
            self.status = 200
            body = b""
            if request.get_method() != "HEAD":
                start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
                body = data[start:end + 1]
                self.status = 206
                self.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            super().__init__(body)

    monkeypatch.setattr(walking.urllib.request, "urlopen", lambda request, timeout: Response(request))
    remote = walking.RangeReader("https://example.org/data.zip")
    remote.chunk_size = 4
    assert remote.read(6) == b"abcdef"
    remote.seek(-3, 2)
    assert remote.read() == b"ghi"
    remote.seek(0)
    assert remote.read(2) == b"ab"
    assert len(requests) == 4  # HEAD and three cached chunks.
    assert remote.downloaded_bytes == len(data)
    with pytest.raises(ValueError):
        remote.seek(-1)
