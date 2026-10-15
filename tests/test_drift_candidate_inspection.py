import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from inspect_drift_candidates import acquire, longest_run, xlsx_headers


def test_contiguous_duration_is_not_total_samples():
    t = np.arange(7) * .1
    assert longest_run(t, [0, 1, 1, 0, 1, 0, 1]) == pytest.approx(.1)
    assert longest_run(t, [0, 0, 0, 0, 0, 0, 0]) == 0
    assert longest_run(t, [0, 0, 0, 1, 0, 0, 0]) == 0


def test_cached_download_checksum_and_bounds_preserve_file(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"original")
    spec = {"filename": path.name, "max_bytes": 20, "expected_md5": hashlib.md5(b"original").hexdigest()}
    assert acquire(spec, tmp_path)[1] == b"original"
    with pytest.raises(ValueError, match="checksum"):
        acquire({**spec, "expected_md5": "0" * 32}, tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        acquire({**spec, "max_bytes": 1}, tmp_path)
    assert path.read_bytes() == b"original"


def test_xlsx_shared_and_inline_headers():
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}"><si><t>X</t></si></sst>')
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="COM" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Y</t></is></c></row><row r="2"><c r="A2"><v>1.25</v></c></row></sheetData></worksheet>')
    sheet = xlsx_headers(stream.getvalue())[0]
    assert sheet["sheet"] == "COM"
    assert sheet["dimension"] == "A1:B2"
    assert sheet["first_four_rows"] == [{"A1": "X", "B1": "Y"}, {"A2": "1.25"}]
