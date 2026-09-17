"""
make_subject_disjoint_manifests.py — 根据划分配置生成train/val/test清单

用法:
  python scripts/make_subject_disjoint_manifests.py \
    --source-manifest train.txt \
    --source-manifest val.txt \
    --split-config splits/subject_disjoint_split.yaml \
    --output-dir local_splits/subject_disjoint
"""
import argparse
import os
import re
import sys
import yaml
from collections import defaultdict


def parse_subject(path):
    m = re.search(r'(BT\d{2})', path)
    if not m:
        raise ValueError(f"Cannot parse subject from path: {path}")
    return m.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', action='append', required=True)
    parser.add_argument('--split-config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.split_config, 'r') as f:
        split = yaml.safe_load(f)

    train_subjects = set(split['train_subjects'])
    val_subjects = set(split['val_subjects'])
    test_subjects = set(split['test_subjects'])

    overlap_tv = train_subjects & val_subjects
    overlap_tt = train_subjects & test_subjects
    overlap_vt = val_subjects & test_subjects
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"ERROR: Subject overlap detected:")
        if overlap_tv: print(f"  train ∩ val: {overlap_tv}")
        if overlap_tt: print(f"  train ∩ test: {overlap_tt}")
        if overlap_vt: print(f"  val ∩ test: {overlap_vt}")
        sys.exit(1)

    all_subjects = train_subjects | val_subjects | test_subjects

    all_paths = []
    for manifest_path in args.source_manifest:
        with open(manifest_path, 'r') as f:
            for line in f:
                p = line.strip()
                if p and p not in all_paths:
                    all_paths.append(p)

    subject_paths = defaultdict(list)
    unparseable = []
    for p in all_paths:
        try:
            subj = parse_subject(p)
            subject_paths[subj].append(p)
        except ValueError:
            unparseable.append(p)

    if unparseable:
        print(f"ERROR: {len(unparseable)} paths with unparseable subjects")
        for p in unparseable:
            print(f"  {p}")
        sys.exit(1)

    missing = all_subjects - set(subject_paths.keys())
    if missing:
        print(f"WARNING: Subjects in split config but no CSVs found: {missing}")

    extra = set(subject_paths.keys()) - all_subjects
    if extra:
        print(f"WARNING: CSVs found for subjects not in split config: {extra}")
        print("  These CSVs will be excluded from all manifests.")

    os.makedirs(args.output_dir, exist_ok=True)

    splits = {
        'train.txt': train_subjects,
        'val.txt': val_subjects,
        'test.txt': test_subjects,
    }

    for fname, subjects in splits.items():
        out_path = os.path.join(args.output_dir, fname)
        paths = []
        for subj in sorted(subjects):
            paths.extend(subject_paths.get(subj, []))
        with open(out_path, 'w', encoding='utf-8') as f:
            for p in paths:
                f.write(p + '\n')
        print(f"  {fname}: {len(paths)} CSVs from {len(subjects)} subjects")

    total = sum(len(subject_paths.get(s, [])) for s in all_subjects)
    print(f"\nTotal CSVs in manifests: {total}")
    print(f"Total CSVs in source: {len(all_paths)}")
    if total != len(all_paths):
        diff = len(all_paths) - total
        print(f"WARNING: {diff} CSVs not included (subjects not in split config)")


if __name__ == '__main__':
    main()
