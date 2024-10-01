# Suffragists

Python pipeline to extract **Forthcoming meetings** tables from *The Common Cause* journal scans.

## Installation

Python 3.11

```bash
pip install requirements.txt
pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2" --no-build-isolation
```

## Setup Google Cloud API

### Install gcloud client
Move to a directory where gcloud will be installed.
```bash
curl -sL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | tar -xz
./google-cloud-sdk/install.sh

```

### Setup API keys

```bash
gcloud init
gcloud auth application-default login
```
So that the environment variable `$HOME/.config/gcloud/application_default_credentials.json` is defined.

Also get a Gemini API key from [here](ai.google.dev) and either add the following line to you `.bashrc`

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

or directly add the key in the code of `src/prompt_gemini.py`.