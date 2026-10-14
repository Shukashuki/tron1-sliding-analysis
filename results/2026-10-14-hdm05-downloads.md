# HDM05 files downloaded locally

Two public research mirrors were used after the official HDM05 provider
continued to return HTTP 403. re3data supplied the repository discovery record,
not the motion payloads. No provider access controls were bypassed.

## Downloaded selection

| Source | Files | Size | Content |
| --- | ---: | ---: | --- |
| JointLearning-HDM05, pinned Git commit | 15 AMC + 5 ASF | 2,809,726 bytes | `sneak2StepsLStart`, five actor skeletons |
| Holden motion-synthesis archive | 180 BVH | 51,994,779 bytes | Side steps, crossover steps, circular walking, turns and shuffles |

The 180 BVH clips contain 54,370 frames with a nominal 120 Hz frame interval.
Their action counts are: `walkLeft2Steps` 16, `walkLeft3Steps` 16,
`walkRightCrossFront2Steps` 16, `walkRightCrossFront3Steps` 13,
`walkLeftCircle4StepsRstart` 18, `walkRightCircle4StepsRstart` 15,
`turnLeft` 30, `turnRight` 30 and left/right two-step shuffles 13 each.
Running, jogging and skating clips were not selected.

Local files are retained under:

```text
outputs/data/hdm05_raw_subset/
outputs/data/hdm05_walking/
```

Raw payloads are excluded from Git. Public reproducibility records contain
filenames, source URLs, hashes, frame counts and channel checks, not the motion
payloads. See the [ASF/AMC audit](hdm05_downloads/asf_amc_audit.json) and
[BVH audit](hdm05_downloads/bvh_audit.json). The ASF/AMC mirror is pinned to commit
`b252ad65598e052c38f49490c1ac9dbadc283fd3`; every downloaded file matched the
expected byte size and Git blob SHA-1. Its AMC clips retain six global root
channels and bilateral foot/toe channels. These checks establish fidelity to
the mirror, not equivalence to inaccessible official originals.

The Holden archive is 5,653,458,561 bytes. HTTP range reads downloaded only
35,712,641 bytes for its ZIP directory and the selected BVH members; the full
archive was not downloaded. ZIP member CRCs, frame counts and channel counts
were checked, and SHA-256 hashes recorded. Root translation channels and
bilateral foot/toe joints are present.

## Important processing distinction

The BVH files are in the archive's `data/processed/hdm05/` directory, not an
original ASF/AMC directory. The archive's `retarget_hdm05.py` explicitly uses
`JacobianInverseKinematics` to standardize the human skeleton, retaining the
source root positions. The retrieved processor SHA-256 is
`5ef7c445ff42962e5a7d0eb7bf6d9f0b9885af6bafbcd318f3cb75a5f26de37b`.

This pre-existing **human-to-human IK derivative is not an unprocessed motion
measurement, not our TRON1 retarget method, and not a GMR result**. Before
foot-heading or ground-proximity conclusions, verify its effect against an
original skeleton/motion pair. No TRON1 retarget or policy training was done
in this acquisition step. A retained global root trajectory alone does not
prove that original foot axes, ground contacts or units were preserved.
The sneak clips are a slow-motion source for checking the ASF/AMC pipeline,
not a substitute for high-offset lateral/turning demonstrations.

## Reproduce acquisition

```bash
python scripts/download_hdm05_subset.py \
  --output-dir outputs/data/hdm05_raw_subset
python scripts/download_hdm05_walking.py \
  --output-dir outputs/data/hdm05_walking
python -m pytest -q tests
```

Both tools preserve existing motion files and reject conflicting contents.
The BVH downloader supports `--list-only` to inspect the selection before
downloading members. Later retrievals must match the published member hashes;
an HTTP ETag is not a substitute for a cryptographic content hash. New audit
files include the retrieved processor hash as well.

HDM05's registry and official site list CC BY-SA 3.0. Preserve the HDM05
attribution and share-alike requirements when publishing derived motion data.
Acknowledgment: The data used in this project was obtained from HDM05.
Cite Müller, Röder, Clausen, Eberhardt, Krüger and Weber,
*Documentation Mocap Database HDM05*, Technical Report CG-2007-2, 2007.

Sources: [re3data HDM05 entry](https://www.re3data.org/repository/r3d100011968),
[JointLearning-HDM05 mirror](https://github.com/zhangchi8620/JointLearning-HDM05/tree/b252ad65598e052c38f49490c1ac9dbadc283fd3),
[Holden research-data download page](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing).
