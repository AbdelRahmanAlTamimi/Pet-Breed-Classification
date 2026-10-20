# Oxford-IIIT Pet Corruption Suite

Run the pipeline with:

```text
dvc repro
```

The suite creates three severities of each corruption for the official test
split. Outputs are written below `data/corrupted/`. The committed split index
is `data/split_index.json`, and the combined manifest at
`data/processed/manifest.json` contains both clean source records and corrupted
records.

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

## Getting the data: `dvc pull` (primary)

The default DVC remote `public` (committed in `.dvc/config`) is a read-only HTTP view
of a Cloudflare R2 bucket at an `r2.dev` URL, so no credentials are needed. It holds
both archives, the corrupted test sets, the manifest and the `train` stage output
(66,053 objects, 4,352,746,975 bytes on 2026-10-13). DVC is not a project dependency;
install it once with `uv tool install dvc`.

```bash
dvc pull data/raw/annotations.tar.gz.dvc data/raw/images.tar.gz.dvc
cd data/raw && tar xzf images.tar.gz && tar xzf annotations.tar.gz
cd ../..
dvc repro
```

`r2.dev` URLs are rate-limited by Cloudflare and meant for light use, so a full pull of
the 66,042 corrupted images can be throttled. If that happens, pull only the archives
(first command) and rebuild the rest with `dvc repro`.

The owner pushes with a separate S3 remote (`storage`) whose endpoint and credentials
live only in the git-ignored `.dvc/config.local`.

## Fallback: rebuild from the official archives

If the DVC remote is unavailable, download the archives from the dataset site and
rebuild. The md5 values are the ones recorded in `data/raw/*.tar.gz.dvc`:

| Archive | URL | md5 |
|---|---|---|
| `images.tar.gz` | https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| `annotations.tar.gz` | https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz | `95a8c909bbe2e81eed6a22bccdf3f68f` |

```bash
cd data/raw
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
md5sum images.tar.gz annotations.tar.gz   # must match the table above
tar xzf images.tar.gz && tar xzf annotations.tar.gz
cd ../..
dvc repro            # corruptions, manifest, validation and training (GPU)
```

`config.py` is not a dependency of the data stages; after changing a setting that
affects the corruption suite, run `dvc repro -f generate_corruptions`.
