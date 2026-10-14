"""Query the re3data REST registry and retain auditable motion-source records.

re3data indexes repositories; motion files and action annotations are fetched
from their official providers separately. The v1 list is filtered locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET


BASE = "https://www.re3data.org"
HDM05_ID = "r3d100011968"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "TRON1-research-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(8_000_001)
    if len(payload) > 8_000_000:
        raise ValueError("Registry response exceeds the bounded metadata size")
    return payload


def fields(root, name):
    return [(node.text or "").strip() for node in root.iter()
            if node.tag.split("}")[-1] == name and (node.text or "").strip()]


def parse_index(payload):
    root = ET.fromstring(payload)
    rows = []
    for node in root.iter():
        if node.tag.split("}")[-1] != "repository":
            continue
        values = {child.tag.split("}")[-1]: (child.text or "").strip()
                  for child in node}
        if values.get("id") and values.get("name"):
            rows.append({key: values.get(key, "") for key in ("id", "name", "doi")})
    return rows


def parse_record(raw, identifier):
    record = ET.fromstring(raw)
    if fields(record, "re3data.orgIdentifier") != [identifier]:
        raise ValueError("API response identifier does not match the requested repository")
    url_fields = {}
    for node in record.iter():
        name = node.tag.split("}")[-1]
        value = (node.text or "").strip()
        if name.endswith("URL") and value:
            url_fields.setdefault(name, []).append(value)
    return {"id": identifier, "registry_url": BASE + "/repository/" + identifier,
            "api_url": BASE + "/api/v1/repository/" + identifier,
            "record_sha256": hashlib.sha256(raw).hexdigest(),
            "name": fields(record, "repositoryName"),
            "repository_url": fields(record, "repositoryURL"),
            "description": fields(record, "description"),
            "data_access_type": fields(record, "dataAccessType"),
            "data_license_name": fields(record, "dataLicenseName"),
            "data_license_url": fields(record, "dataLicenseURL"),
            "last_update": fields(record, "lastUpdate"), "url_fields": url_fields,
            "motion_file_links": sorted({value for values in url_fields.values()
                                         for value in values
                                         if urlsplit(value).path.lower().endswith(
                                             (".asf", ".amc", ".c3d", ".bvh", ".zip", ".tar.gz"))})}


def probe_provider(url):
    # Prefer HTTPS; do not disable certificate checks or bypass provider access controls.
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Expected an HTTP(S) provider URL")
    requested = urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))
    request = urllib.request.Request(requested, headers={"User-Agent": "TRON1-research-source-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(200_001)
            return {"registry_url": url, "requested_url": requested,
                    "final_url": response.url, "http_status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "response_prefix_sha256": hashlib.sha256(body).hexdigest(),
                    "prefix_bytes": len(body), "prefix_truncated": len(body) > 200_000}
    except urllib.error.HTTPError as error:
        return {"registry_url": url, "requested_url": requested,
                "http_status": error.code, "error": str(error)}
    except urllib.error.URLError as error:
        return {"registry_url": url, "requested_url": requested,
                "http_status": None, "error": str(error.reason)}


def discover(output: Path, ids, probe=False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserving existing registry audit")
    if not ids or any(not re.fullmatch(r"r3d[0-9]+", identifier) for identifier in ids):
        raise ValueError("Expected re3data identifiers such as r3d100011968")
    url = BASE + "/api/v1/repositories"
    payload = fetch(url)
    rows = parse_index(payload)
    terms = ("motion capture", "mocap", "human motion", "whole-body", "biomechanic")
    matches = [row for row in rows if any(term in row["name"].lower() for term in terms)]
    records = []
    raw_records = {}
    for identifier in dict.fromkeys(ids):
        if identifier not in {row["id"] for row in rows}:
            raise ValueError(f"Requested repository absent from REST index: {identifier}")
        record_url = BASE + "/api/v1/repository/" + identifier
        raw = fetch(record_url)
        record = parse_record(raw, identifier)
        record["provider_probes"] = ([probe_provider(link) for link in record["repository_url"]]
                                     if probe else [])
        records.append(record)
        raw_records[identifier] = raw
    report = {"kind": "re3data_registry_human_motion_source_discovery", "api_doc": BASE + "/api/doc/",
              "index_url": url, "index_sha256": hashlib.sha256(payload).hexdigest(),
              "index_repository_count": len(rows), "local_name_filter": list(terms),
              "matching_repositories": matches, "selected_records": records,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "HDM05 slow lateral stepping, crossover stepping and walking turns; excludes running and skating.",
              "motion_download_status": "Registry metadata only; no motion samples downloaded by this tool.",
              "motion_selection_rule": "Global pelvis position plus bilateral ankle/toe orientation; analyze signed course-to-foot heading independently on each side and near-ground subsets."}
    output.mkdir(parents=True, exist_ok=True)
    for identifier, raw in raw_records.items():
        (output / (identifier + ".xml")).write_bytes(raw)
    (output / "re3data_sources.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-id", nargs="+", default=[HDM05_ID])
    parser.add_argument("--probe-provider", action="store_true",
                        help="Read provider entry pages and record HTTP status (not motion downloads)")
    args = parser.parse_args()
    report = discover(args.output_dir, args.repository_id, args.probe_provider)
    print(json.dumps({"matches": report["matching_repositories"],
                      "selected": [{"id": r["id"], "name": r["name"],
                                    "motion_file_links": r["motion_file_links"],
                                    "provider_probes": r["provider_probes"]}
                                   for r in report["selected_records"]]}))


if __name__ == "__main__":
    main()
