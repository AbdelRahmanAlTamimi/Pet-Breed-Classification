# Oxford-IIIT Pet Corruption Suite

Run the generator with:

```text
uv run python -m pet_breed_classification.corruptions
```

The suite creates three severities of each corruption for the official test
split. Outputs are written below `data/corrupted/` and described by
`data/processed/corruption_manifest.json`.

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| Gaussian blur radius | 1 | 2 | 4 |
| Brightness up factor | 1.3 | 1.6 | 2.0 |
| Brightness down factor | 0.7 | 0.5 | 0.3 |
| JPEG quality | 50 | 30 | 10 |
| Downscale shorter side | 128 | 96 | 64 |
| Motion blur kernel size | 5 | 9 | 15 |

Downscale/upscale preserves the source aspect ratio: the shorter side is
reduced to the severity target, then the image is resized back to its original
dimensions.
