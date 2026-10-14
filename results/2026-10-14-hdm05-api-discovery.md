# HDM05 source discovery through the re3data REST API

The human-motion selection is HDM05 slow lateral stepping, crossover stepping
and walking turns. Running and skating are excluded. These are selection
targets, not downloaded or retargeted experiment results. Human work remains
retarget-only; no human policy is trained in this step.

## Actual API retrieval

The source-discovery script reads these public REST endpoints directly, without
scraping search-engine results:

1. `https://www.re3data.org/api/v1/repositories`
2. `https://www.re3data.org/api/v1/repository/r3d100011968`

The first endpoint returned 3,536 repository records in this retrieval. The
script filters their names locally and validates the requested ID against the
index. The second endpoint returned the HDM05 repository record successfully.
Its original bytes are preserved in
[r3d100011968.xml](hdm05_registry_api/r3d100011968.xml); the
[audit JSON](hdm05_registry_api/re3data_sources.json) stores the retrieval time,
SHA-256 hashes, access/license fields, all nonempty URL fields and provider
reachability. The registry entry describes C3D and ASF/AMC data and lists open
data access and a CC BY-SA 3.0 license URL. These statements describe the
registry record, not an independent verification of individual archive licenses.

The registry returned **no individual motion-file or archive URLs** in this
record. Its repository URL points to the HDM05 provider's index page. A normal
HTTPS GET to that entry point returned **HTTP 403** from this environment.
Thus the REST request succeeded, but motion acquisition has not succeeded.
The registry is not a proxy for the provider's motion files. No raw motion,
GMR trajectory, contact-angle measurement or human rendering was produced by
this metadata retrieval. No access controls or certificate checks were bypassed.

## Reproduce

```bash
python scripts/discover_re3data_human_sources.py \
  --repository-id r3d100011968 \
  --probe-provider \
  --output-dir outputs/hdm05_api_discovery
python -m pytest -q tests/test_re3data_discovery.py
```

Use a fresh output directory: the tool refuses to overwrite an existing audit.
Provider status and live registry hashes may change between retrievals.
The tests run offline with synthetic XML fixtures and check namespace handling,
ID validation, exact raw-record preservation, duplicate-ID handling and the
distinction between registry metadata and motion downloads.

## Next data requirement

An accessible HDM05 ASF/AMC or C3D archive is still needed. Preserve global
translation and the bilateral foot axes, rather than substituting centered
action-classification features. On acquisition, verify coordinate axes, units,
sampling rate, actor skeleton, action labels and archive licensing before GMR.
Measure left/right course-to-foot or course-to-wheel heading offsets separately,
including near-ground subsets; a geometry-based ground mask is not a measured
contact force. Slow-motion speed masks must be reported explicitly, not silently
lowered to manufacture offset coverage. Wheel-axis features are auxiliary:
TRON1 has no independent wheel-yaw actuator.

Sources: [re3data API documentation](https://www.re3data.org/api/doc/),
[HDM05 registry entry](https://www.re3data.org/repository/r3d100011968).
