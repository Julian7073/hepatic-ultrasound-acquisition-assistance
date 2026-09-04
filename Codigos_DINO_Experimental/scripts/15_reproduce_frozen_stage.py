"""Read-only model replay and stage-specific result generation; never calls fit.

Private inputs stay at --artifact-root. Outputs contain coded identifiers and
numeric results only. The frozen release and its archived results are not edited.
"""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.binary_temporal import (
    make_temporal_samples, clear_probabilities, sample_prediction_table,
    aggregate_video_predictions, binary_metric_row, calibrate_abstention_thresholds,
    add_actions, action_metric_row,
)

VIEWS = {'transversal': 'transverse', 'oblicua': 'oblique', 'hepatorrenal': 'hepatorenal'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['thresholds', 'internal_test', 'independent_test'], required=True)
    ap.add_argument('--view', choices=VIEWS, required=True)
    ap.add_argument('--artifact-root', type=Path, default=ROOT.parent)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--output-root', type=Path, default=ROOT.parent/'outputs'/'stage_reproduction')
    args = ap.parse_args()
    base = args.artifact_root/'outputs'/'dino_experimental'/'binary_improvement'
    dest = args.output_root/args.stage/args.view
    dest.mkdir(parents=True, exist_ok=True)
    if args.stage != 'independent_test':
        source = pd.read_csv(base/'reports'/'09_binary_selected_oof_actions.csv')
        source = source[source.view.eq(args.view)].copy()
        if source.empty or source.patient.eq('P005').any() or not source.role.eq('development').all():
            raise ValueError('Expected development-only out-of-fold predictions')
        if args.stage == 'thresholds':
            result = calibrate_abstention_thresholds(source, minimum_action_precision=.90)
            pd.DataFrame([result]).to_csv(dest/'thresholds.csv', index=False)
            acted = add_actions(source, result['adjust_threshold'], result['capture_threshold'])
            # Unlike legacy search-stage fields, these metrics use the final limits.
            pd.DataFrame([action_metric_row(acted, 'deployed')]).to_csv(dest/'post_fallback_action_metrics.csv', index=False)
        else:
            rows = []
            for patient, group in source.groupby('patient'):
                anchors = group[group.true_quality.isin(['clear', 'blurry'])]
                rows.append({'patient': patient, 'view': args.view,
                             **binary_metric_row(anchors.true_quality, anchors.predicted_anchor, 'video')})
            pd.DataFrame(rows).to_csv(dest/'internal_video_metrics.csv', index=False)
        print(dest)
        return

    bundle = joblib.load(base/'models'/f'{args.view}__binary_dinov2.joblib')
    if bundle['temporal_mode'] != 'window5' or bundle['window_size'] != 5:
        raise ValueError('Expected frozen five-embedding model')
    prefix = (base.parent/'embeddings'/'dinov2_small_stride5' if bundle['preprocessing'] == 'full'
              else base/'embeddings'/'dinov2_small_fan_crop_stride5')
    x = np.load(prefix.with_suffix('.npz'))['embeddings']
    meta = pd.read_csv(prefix.parent/(prefix.name+'_metadata.csv'))
    use = meta.patient.eq('P005') & meta.role.eq('external_test') & meta.view.eq(args.view)
    x, meta = x[use], meta.loc[use].reset_index(drop=True)
    if meta.empty:
        raise ValueError('No independent inputs')
    features, sm = make_temporal_samples(x, meta, 'window5')
    labels, probabilities = clear_probabilities(bundle['model'], features)
    windows = sample_prediction_table(sm, labels, probabilities)

    manifest_path = args.manifest or args.artifact_root/'kaggle_hepatic_ultrasound_private'/'upload'/'frames_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.participant.eq('P005') & manifest.view.eq(VIEWS[args.view])].copy()
    keys = ['nominal_quality', 'frame_number']
    if manifest.duplicated(keys).any():
        raise ValueError('Ambiguous manifest mapping: use a video-level key')
    manifest = manifest.set_index(keys)
    joined = []
    for i, row in windows.iterrows():
        ids = list(range(int(row.frame_number), int(row.frame_number)+25, 5))
        ref = manifest.loc[[(row.true_quality, n) for n in ids]]
        if not ref.label_source.eq('expert_radiologist_review').all():
            raise ValueError('Unexpected independent label provenance')
        expected = {'clear': 'informative', 'blurry': 'non_informative', 'medium': ''}[row.true_quality]
        if not ref.binary_anchor_label.fillna('').eq(expected).all():
            raise ValueError('Window contains inconsistent reference labels')
        video_id = ref.video_id.iloc[0]
        if ref.video_id.nunique() != 1:
            raise ValueError('Window crosses video boundary')
        windows.loc[i, 'video_id'] = video_id
        windows.loc[i, 'sample_id'] = f'{video_id}__start_{ids[0]:04d}'
        joined.append({'view': VIEWS[args.view], 'video_id': video_id,
                       'window_start': ids[0], 'window_end': ids[-1],
                       'source_indices': ';'.join(map(str, ids)), 'joined_labels': len(ref),
                       'reference_label': expected, 'label_source': 'expert_radiologist_review',
                       'binary_included': row.true_quality != 'medium'})
    windows = windows.drop(columns=['image_path', 'filename'], errors='ignore')
    windows.to_csv(dest/'window_predictions.csv', index=False)
    pd.DataFrame(joined).to_csv(dest/'label_prediction_join.csv', index=False)
    videos = aggregate_video_predictions(windows)
    videos.to_csv(dest/'video_predictions.csv', index=False)
    metrics = []
    for unit, source in [('window', windows), ('video', videos)]:
        anchors = source[source.true_quality.isin(['clear', 'blurry'])]
        metrics.append({'view': VIEWS[args.view], 'unit': unit,
                        **binary_metric_row(anchors.true_quality, anchors.predicted_anchor, 'binary')})
    pd.DataFrame(metrics).to_csv(dest/'binary_metrics_by_unit.csv', index=False)
    print(dest)


if __name__ == '__main__':
    main()
