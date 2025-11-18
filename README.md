# Air Quality Prediction Lab – Scalable ML Systems

This repository contains my solution to the Air Quality tutorial lab for the Scalable Machine Learning and Deep Learning Models (ID2223) course. We have built a serverless model that

- Backfills historical air quality and weather data into Hopsworks
- Schedules a daily feature pipeline through GitHub Actions that updates Feature Groups
- Trains a model to predict pm25 for the next 7 days
- Runs a batch inference pipeline that generates a dashboard with forecasts


## How to run the lab locally

1. Create and activate a virtual environment:
   conda create -n aq python=3.11
   conda activate aq
2. Install python dependencies
   pip install -r requirements.txt
3. jupyter notebook


---

# mlfs-book
O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## ML System Examples


[Dashboards for Example ML Systems](https://featurestorebook.github.io/mlfs-book/)




# Run Air Quality Tutorial

See [tutorial instructions here](https://docs.google.com/document/d/1YXfM1_rpo1-jM-lYyb1HpbV9EJPN6i1u6h2rhdPduNE/edit?usp=sharing)
    # Create a conda or virtual environment for your project
    conda create -n book 
    conda activate book

    # Install 'uv' and 'invoke'
    pip install invoke dotenv

    # 'invoke install' installs python dependencies using uv and requirements.txt
    invoke install


## PyInvoke

    invoke aq-backfill
    invoke aq-features
    invoke aq-train
    invoke aq-inference
    invoke aq-clean



## Feldera


pip install feldera ipython-secrets
sudo apt-get install python3-secretstorage
sudo apt-get install gnome-keyring 

mkdir -p /tmp/c.app.hopsworks.ai
ln -s  /tmp/c.app.hopsworks.ai ~/hopsworks
docker run -p 8080:8080 \
  -v ~/hopsworks:/tmp/c.app.hopsworks.ai \
  --tty --rm -it ghcr.io/feldera/pipeline-manager:latest
