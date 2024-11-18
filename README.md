# Suffragists

Python pipeline to extract tables of upcoming events from *The Common Cause*, *Suffragette*, and *Votes For Women* journal scans.

## Installation
Pipeline tested only for Python 3.11 and Ubuntu 22.04.

```bash
pip install requirements.txt
```

## Setup Google Cloud API

### Install gcloud client (for Linux)
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

Also get a Google API key from [here](ai.google.dev) and either add the following line to you `.bashrc`

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

or directly add the key in the code of `src/prompt_gemini.py`.

## Tips

For the issues on which the pipeline throws errors, try rerunning with different `temperature` and `top_p` parameters. For instance:

```bash
python main.py --temperature 1.5 --top_p 0.95
```