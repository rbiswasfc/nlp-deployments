# nlp-deployments
Train and deploy NLP models

# Steps

## Install gcloud on mac
Ref: https://cloud.google.com/sdk/docs/quickstart
    * Download gcloud sdk 
    * execute `./google-cloud-sdk/install.sh` from the folder containing "google-cloud-sdk"
    * initialize by running `./google-cloud-sdk/bin/gcloud init`
    * [optional] Choose default compute engine zone
        * If zone is not specified while creating/working with compute engine resources, default will be assumed
        *  The default can be changed by running `gcloud config set compute/zone NAME`
            * Replace NAME with zone name e.g. asia-southeast1-c
        * use `gcloud config list` to check current configurations

## Crete VM with PyTorch 
Ref: 
* https://course19.fast.ai/start_gcp.html
* https://cloud.google.com/deep-learning-vm/docs/quickstart-cli?hl=en_US

Steps:
* Choose deep learning image
    * https://cloud.google.com/deep-learning-vm/docs/images#listing-versions
    * set the image family e.g.
    * `export IMAGE_FAMILY="pytorch-latest-gpu"`
* Set instance type
    * `export INSTANCE_TYPE="n1-highmem-4"` 
    
Execute the following to create an instance with pytorch

```
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="asia-southeast1-c"
export INSTANCE_NAME="pytorch-nlp"
export INSTANCE_TYPE="n1-highmem-4" 

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
```

* useful commands
    * Check status of all compute engine VMs in the current project
        * `gcloud compute instances list` 
    * Check status of a specific instance
        * `gcloud compute instances describe $INSTANCE_NAME`
    * Once the VM is started, access it via
        * `gcloud compute ssh $INSTANCE_NAME` OR
        * `gcloud compute ssh --zone=$ZONE rbiswas@$INSTANCE_NAME -- -L 8080:localhost:8080`
        * If private/public SSH key file does not exist for gcloud, the SSH keygen will be executed to generate keys. These keys will be stored in `~\.ssh` folder. This will generate 
    * Go to `localhost:8080/tree` to access jupyter notebook
    * Stop the instance 
        *`gcloud compute instances stop $INSTANCE_NAME`
    * Restart the VM after stopping
        * `gcloud compute instances start $INSTANCE_NAME`
    * Clean up and delete instance
        * `gcloud compute instances delete $INSTANCE_NAME`

## Connect with github repo
