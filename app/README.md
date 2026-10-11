# AQ-GNoME screening app

A Streamlit interface over the `aq_gnome` package: element filters, Pourbaix stability criteria
and post-filters as in `notebooks/data_analysis_v2.ipynb`, with a results table, CSV export and
CIF download. Screening uses `aq_gnome.fast_screening`, which keeps the whole precomputed
decomposition-energy grid in memory, so any query over the full database answers in well under a
second.

## Run locally

```bash
pip install -e ".[app]"
streamlit run app/app.py          # opens http://localhost:8501
```

The data folder is found automatically at `<repo>/data` (download `data.zip` from
https://doi.org/10.11583/DTU.30738716 and unzip it there), or point `AQ_GNOME_DATA_DIR` elsewhere.
The first load takes 20–40 s and ~1.8 GB of RAM per dataset variant; afterwards every query is
sub-second.

| Variable                | Default        | Meaning                                                       |
|-------------------------|----------------|---------------------------------------------------------------|
| `AQ_GNOME_DATA_DIR`     | `<repo>/data`  | Folder with the DTU Data deposit files                        |
| `AQ_GNOME_MAX_VARIANTS` | `1`            | Dataset variants (solid filter × functional) kept in RAM      |

## Run with Docker

```bash
cd app
cp .env.example .env              # optional: data path, hostname, RAM budget
docker compose up -d --build      # app + Caddy (HTTP/HTTPS) on ports 80/443
```

`SITE_ADDRESS=:80` serves plain HTTP on the machine's IP. Set it to a hostname
(`aqgnome.example.org`, or `<ip>.sslip.io` if you have no DNS) and Caddy obtains a Let's Encrypt
certificate by itself.

## Deploy on a Google Cloud VM

Sizing: the app needs ~1.8 GB RAM per dataset variant plus ~0.5 GB per active user session.
An `e2-medium` (4 GB) serves a handful of concurrent users; `e2-standard-2` (8 GB) is comfortable.
CPU hardly matters. 32 GB of disk is enough (OS + image ≈ 8 GB, data ≈ 7 GB).

1. Compute Engine → Create instance: `e2-standard-2`, Ubuntu 24.04 LTS, 32 GB balanced disk,
   region `europe-north1` / `europe-west1`, tick *Allow HTTP* and *Allow HTTPS traffic*, and
   reserve a static external IP.
2. SSH in (browser SSH is fine) and install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && newgrp docker
   ```
3. Get the code and the data:
   ```bash
   git clone https://github.com/Matminator/AQ-GNoME.git && cd AQ-GNoME
   # data.zip (4.9 GB) from the DTU Data deposit https://doi.org/10.11583/DTU.30738716
   wget -O data.zip https://ndownloader.figshare.com/files/64699584
   unzip data.zip -d . && rm data.zip
   ```
4. Start it:
   ```bash
   cd app && cp .env.example .env
   echo "SITE_ADDRESS=$(curl -s ifconfig.me).sslip.io" >> .env   # or your own hostname
   docker compose up -d --build
   ```
   The first start pulls dependencies (~5 min) and loads the data (~1 min). Logs:
   `docker compose logs -f app`.
5. Updating: `git pull && docker compose up -d --build`.

Restricting access (e.g. to a research group) is easiest with Cloudflare Access or Google IAP in
front of the VM; Streamlit itself has no login in this version.
