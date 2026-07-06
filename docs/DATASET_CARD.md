# Dataset Card — ISL Translator Keypoint Dataset

## Dataset Summary

A small keypoint dataset of isolated Indian Sign Language (ISL) gestures, collected using MediaPipe Holistic. Each clip contains normalized body/hand landmark coordinates — **no raw video is stored**.

## Dataset Details

| Field | Value |
|-------|-------|
| Language | Indian Sign Language (ISL) |
| Task | Isolated word classification |
| Format | NumPy `.npy` arrays, shape `(T, 225)` |
| Vocabulary | 10 words (v1); extendable to 29 (see `config.py`) |
| Collection tool | `data_collection/record_clip.py` |
| License | CC-BY-NC 4.0 |

## Vocabulary (v1)

`hello`, `thank_you`, `please`, `sorry`, `yes`, `no`, `help`, `water`, `food`, `home`

## Data Format

Each clip file: `data/<word>/<word>_<signer>_<timestamp>.npy`

Array shape: `(T, 225)` where:
- `T` = number of captured frames (≤ 50)
- `225` = 99 (pose 33×3) + 63 (left hand 21×3) + 63 (right hand 21×3)

All coordinates are **normalized** relative to shoulder midpoint and width at collection time.

`metadata.csv` columns: `clip_id`, `word`, `signer_id`, `timestamp`, `n_frames`, `quality_flag`, `filepath`

## Collection Methodology

- Recorded via webcam using MediaPipe Holistic with `min_detection_confidence=0.5`
- Signer performs each sign while visible from the waist up
- 15–20 clips per word recommended; clips shorter than 15 frames are discarded automatically
- Sessions varied across lighting conditions and clothing to improve robustness

## Signers

| Signer ID | Description | Sessions |
|-----------|-------------|----------|
| signer_01 | *Update with your details* | *N* |

## Known Limitations and Biases

> [!IMPORTANT]
> **Single signer**: v1 data comes from one signer. The model trained on this data **will not generalize** reliably to other signers, especially those signing different regional dialects of ISL.

- **Regional variation**: ISL has significant dialect variation across India. This dataset captures one regional/personal signing style and should not be treated as "canonical" ISL.
- **Controlled conditions**: recorded in home/office settings; performance will degrade in unusual environments, extreme lighting, or unusual distances from the camera.
- **No sign language expert review**: the signs collected here are based on publicly available ISL references. They have not been validated by a qualified ISL interpreter.
- **Vocabulary is not representative**: 10–29 words is a fraction of the full ISL lexicon. This is a demonstration dataset, not a comprehensive resource.

## Next Steps for Expanding the Dataset

1. Recruit additional signers with diverse ISL backgrounds (different regions, ages)
2. Consult with ISL interpreters or deaf community organizations to verify sign accuracy
3. Add a contribution guide so the community can submit their own recordings
4. Consider institutional ethics review before any large-scale collection

## License

**Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)**

You are free to share and adapt this dataset for non-commercial purposes with attribution. Commercial use requires explicit permission.

## Citation

If you use this dataset, please cite:
```
ISL Translator Keypoint Dataset (2024)
https://github.com/<your-repo>
```
