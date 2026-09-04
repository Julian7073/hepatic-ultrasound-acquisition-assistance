"""Audit saved segmentation runs and the frozen P005 longitudinal decisions.

The audit is read-only with respect to archived experiments. It verifies common
test identities, run settings, metric aggregation, sample exclusions and a
leave-one-gate-out reconstruction. It never retrains or changes thresholds.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd


def source_key(name):
    return re.sub(r'\.rf\.[0-9a-f]+\.[^.]+$', '', str(name), flags=re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output-root', type=Path,
                    default=Path(__file__).resolve().parents[2]/'outputs'/'stage_reproduction'/'audit')
    args = ap.parse_args()
    exp = args.artifact_root/'outputs'/'experimental_segmentation_pipeline'/'experiments'
    independent = (args.artifact_root/'outputs'/'experimental_segmentation_pipeline'/
                   'p005_longitudinal_final_stride1_transfer'/'frame_results.csv')
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows, test_sets = [], {}
    for folder in sorted(exp.glob('*pretrained*50ep*')):
        required = [folder/'config.json', folder/'train_log.csv', folder/'test_metrics.csv',
                    folder/'test_per_image_metrics.csv', folder/'best_model.pth']
        if not all(p.exists() for p in required):
            continue
        config = json.loads(required[0].read_text(encoding='utf-8-sig'))
        log, summary, per_image = pd.read_csv(required[1]), pd.read_csv(required[2]).iloc[0], pd.read_csv(required[3])
        test_sets[folder.name] = {source_key(n) for n in per_image.filename}
        metadata = config['model_metadata']
        deltas = [abs(float(per_image[k].mean())-float(summary['test_'+k]))
                  for k in ['dice', 'iou', 'precision', 'recall']]
        rows.append({
            'experiment': folder.name, 'target': config['class_name'],
            'architecture': metadata['architecture_display'], 'encoder': metadata['encoder'],
            'encoder_initialization': metadata['encoder_weights'], 'fine_tuning': metadata['fine_tuning'],
            'pretrained': config['pretrained'], 'max_epochs': config['epochs'],
            'completed_epochs': len(log), 'seed': config['seed'], 'input_px': config['image_size'],
            'batch_size': config['batch_size'], 'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'], 'augmentation': config['augmentation'],
            'sampling': config['sampling_strategy'], 'split_strategy': config['split_strategy'],
            'train_n': config['split_sizes']['train'], 'valid_n': config['split_sizes']['valid'],
            'test_n': len(per_image), 'test_dice': summary.test_dice,
            'max_reaggregation_delta': max(deltas), 'checkpoint_exists': required[4].exists(),
        })
    runs = pd.DataFrame(rows)
    if len(runs) != 9 or len({tuple(sorted(v)) for v in test_sets.values()}) != 1:
        raise ValueError('The nine runs do not share one normalized test set')
    common = next(iter(test_sets.values()))
    if len(common) != 101:
        raise ValueError('Expected 101 common test images')
    if not (runs[['pretrained', 'checkpoint_exists']].all().all()
            and runs.seed.eq(42).all() and runs.test_n.eq(101).all()
            and runs.max_reaggregation_delta.le(2e-15).all()):
        raise ValueError('Run-record audit failed')
    runs.to_csv(args.output_root/'segmentation_training_record_audit.csv', index=False)

    frame = pd.read_csv(independent)
    if len(frame) != 303 or frame.duplicated(['quality', 'frame_index']).any():
        raise ValueError('Expected 303 unique P005 decisions')
    if frame.groupby('quality').size().to_dict() != {'blurry': 101, 'clear': 101, 'medium': 101}:
        raise ValueError('Unexpected quality denominators')
    gates = {
        'roi_present': frame.roi_present.eq(1), 'liver_present': frame.liver_present.eq(1),
        'liver_ratio': frame.liver_roi_ratio.ge(.15), 'la_present': frame.la_present.eq(1),
        'la_area': frame.la_area_ok.eq(1), 'la_std': frame.la_std_ok.eq(1),
        'la_entropy': frame.la_entropy_ok.eq(1), 'border': frame.border_evidence.eq(1),
    }
    ablation = []
    for omitted in ['none', *gates]:
        accepted = np.ones(len(frame), dtype=bool)
        for name, condition in gates.items():
            if name != omitted:
                accepted &= condition.to_numpy()
        for quality in ['clear', 'medium', 'blurry']:
            use = frame.quality.eq(quality)
            ablation.append({'omitted_gate': omitted, 'quality': quality,
                             'records': int(use.sum()), 'raw_accepted': int(accepted[use].sum())})
    pd.DataFrame(ablation).to_csv(args.output_root/'longitudinal_leave_one_gate_out.csv', index=False)
    pd.DataFrame([{'records': len(frame), 'unique_quality_frame_keys': len(frame),
                   'clear_binary_included': 101, 'blurry_binary_included': 101,
                   'medium_uncertainty_only': 101, 'roi_present': int(frame.roi_present.sum()),
                   'liver_present': int(frame.liver_present.sum()),
                   'la_present_clear': int(frame[frame.quality.eq('clear')].la_present.sum()),
                   'la_present_medium': int(frame[frame.quality.eq('medium')].la_present.sum()),
                   'la_present_blurry': int(frame[frame.quality.eq('blurry')].la_present.sum())}]).to_csv(
                       args.output_root/'longitudinal_sample_reconciliation.csv', index=False)
    print(args.output_root)


if __name__ == '__main__':
    main()
