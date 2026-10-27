"""Download public K700 validation hockey-stop RGB clips and source hashes.

No credentials, YouTube scraping, model weights, remote uploads or inference.
Category labels do not establish a suitable source, calibrated3D, or contact.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request


def sha(data):return hashlib.sha256(data).hexdigest()


def cached(url,path,limit):
    if path.exists():return path.read_bytes()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TRON1-Hockey-Source-Qualification/1.0'}),timeout=60) as response:
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('Public source exceeded the declared download budget')
    with path.open('xb') as f:f.write(data)
    return data


def run(a):
    root=a.output;root.mkdir(parents=True,exist_ok=True)
    url='https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv'
    raw=cached(url,root/'val.csv',5_000_000)
    if sha(raw)!='a6749d9f3bc785620de5ee96025f201b98b2436eae4b782eb3e1db61f50ab7ac':raise ValueError('Pinned annotation release changed')
    table=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));selected=[r for r in table if r['label']=='hockey stop']
    index=sorted({r['label'] for r in table}).index('hockey stop')+1
    archive_url=f'https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_{index:03d}.tar.gz'
    archive=cached(archive_url,root/'hockey_stop_val.tar.gz',80_000_000)
    allowed={(r['youtube_id'],int(r['time_start']),int(r['time_end'])) for r in selected}
    clips=root/'clips';clips.mkdir(exist_ok=True);rows=[]
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        for member in t:
            if not member.isfile() or not member.name.startswith('hockey stop/') or '..' in Path(member.name).parts or not member.name.endswith('.mp4'):
                raise ValueError('Unexpected archive entry; nothing is blindly extracted')
            name=Path(member.name).name;video,start,end=Path(name).stem.rsplit('_',2)
            if (video,int(start),int(end)) not in allowed:raise ValueError('Class shard does not match pinned hockey-stop annotations')
            data=t.extractfile(member).read();target=clips/name
            if target.exists():
                if target.read_bytes()!=data:raise ValueError('Existing clip differs; preserve it')
            else:
                with target.open('xb') as f:f.write(data)
            rows.append(dict(file=name,youtube_id=video,annotation_start_s=int(start),annotation_end_s=int(end),
                source_video_url='https://www.youtube.com/watch?v='+video,archive_member=member.name,
                sha256=sha(data),bytes=len(data),source_type='RGB_only',metric_world_motion_verified=False))
    audit=dict(kind='Kinetics_hockey_stop_RGB_source_download',annotation_url=url,annotation_sha256=sha(raw),
        archive_url=archive_url,archive_sha256=sha(archive),clips=rows,
        visual_motion_review_required=True,raw_video_redistribution_rights_not_inferred=True)
    path=root/'reproducible_download_audit.json'
    if path.exists():
        if json.loads(path.read_text())!=audit:raise ValueError('Existing audit differs')
    else:path.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(clips=len(rows),archive_sha256=sha(archive),output=str(root))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
