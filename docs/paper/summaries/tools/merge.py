"""Validate provenance, coverage and parent review before publishing the merged corpus."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from batch_io import R, readlines, validate, evidence_overrides
from review_io import pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    for relative, expected in json.loads((R / 'input-integrity.json').read_text()).items():
        assert hashlib.sha256((R.parent / relative).read_bytes()).hexdigest() == expected, relative
    canonical = readlines(R / 'canonical-inputs.jsonl')
    canonical_index = {x['paper_id']: x for x in canonical}
    assert len(canonical_index) == len(canonical)
    ledger = readlines(R / 'source-ledger.jsonl')
    original = readlines(R.parent / 'new-paper-metadata.jsonl')
    assert len(ledger) == len(original)
    assert len({x['source_id'] for x in ledger}) == len(ledger)
    assert {x['source_id'] for x in ledger} == {x['id'] for x in original}
    assert {x['paper_id'] for x in ledger} == set(canonical_index)
    for row in ledger:
        assert row['source_id'] in canonical_index[row['paper_id']]['source_ids']

    manifest = json.loads((R / 'batch-manifest.json').read_text())
    records = readlines(R / 'calibration/approved.jsonl')
    completed, incomplete = [], []
    for batch in manifest:
        name = batch['batch_id']
        input_path = R / 'inputs' / (name + '.jsonl')
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == batch['input_sha256'], name
        assert [x['paper_id'] for x in readlines(input_path)] == batch['paper_ids'], name
        output_path = R / 'batches' / (name + '.jsonl')
        if not output_path.exists():
            incomplete.append(name)
            continue
        # Workers checkpoint partial batches; these do not count as completed.
        try:
            output_records = readlines(output_path)
        except json.JSONDecodeError:
            incomplete.append(name)
            continue
        if len(output_records) < len(batch['paper_ids']):
            incomplete.append(name)
            continue
        validate(name)
        records.extend(output_records)
        completed.append(name)

    records.extend(readlines(R / 'missing-abstract-results.jsonl'))
    assert len({x['paper_id'] for x in records}) == len(records)
    assert {x['paper_id'] for x in records} <= set(canonical_index)
    for result in records:
        source = {**canonical_index[result['paper_id']], **evidence_overrides().get(result['paper_id'], {})}
        assert result['evidence_url'] == source['evidence_url'], result['paper_id']
        assert result['evidence_type'] == source['evidence_type'], result['paper_id']
        if not source['abstract']:
            assert result['status'] == '信息不足'
            assert result['problem'] is None and result['solution'] is None

    rework = {}
    for item in readlines(R / 'reviews/rework-requests.jsonl'):
        rework[item['batch_id']] = item['state']
    unresolved = [name for name, state in rework.items() if state != 'resolved']
    review_pending = pending()
    report = {
        'complete': not incomplete and not review_pending and not unresolved,
        'source_records': len(ledger),
        'canonical_papers': len(canonical),
        'completed_batches': len(completed),
        'total_batches': len(manifest),
        'covered_papers': len(records),
        'pending_papers': len(canonical) - len(records),
        'pending_review_items': len(review_pending),
        'unresolved_rework': unresolved,
        'status_counts_of_completed_records': dict(Counter(x['status'] for x in records)),
        'primary_problem_counts_of_completed_records': dict(Counter(x['primary_problem'] for x in records if x['status'] == '摘要支持')),
    }
    if args.publish:
        assert report['complete'], report
        assert {x['paper_id'] for x in records} == set(canonical_index)
        index = {x['paper_id']: x for x in records}
        merged = []
        for source in canonical:
            row = {k: v for k, v in source.items() if k != 'abstract'}
            row['abstract_available'] = bool(source['abstract'])
            if source['paper_id'] in evidence_overrides():
                row['supplemental_evidence'] = {k: v for k, v in evidence_overrides()[source['paper_id']].items() if k != 'abstract'}
            row.update(index[source['paper_id']])
            merged.append(row)
        (R.parent / 'paper-summaries.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in merged))
        (R / 'final-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
