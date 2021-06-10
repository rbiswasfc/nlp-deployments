# nlp-deployments
Train and deploy NLP models

# Steps
## Setup GCP VM
* Install gcloud on mac
Ref: https://cloud.google.com/sdk/docs/quickstart
    * Download gcloud sdk 
    * execute `./google-cloud-sdk/install.sh` from the folder containing "google-cloud-sdk"
    * initialize by running `./google-cloud-sdk/bin/gcloud init`
    * [optional] Choose default compute engine zone
        * If zone is not specified while creating/working with compute engine resources, default will be assumed
        *  The default can be changed by running `gcloud config set compute/zone NAME`
            * Replace NAME with zone name e.g. asia-southeast1-c