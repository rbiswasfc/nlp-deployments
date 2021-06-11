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
        * `gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080`
        * If private/public SSH key file does not exist for gcloud, the SSH keygen will be executed to generate keys. These keys will be stored in `~\.ssh` folder. This will generate 
    * Go to `localhost:8080/tree` to access jupyter notebook
    * Stop the instance 
        *`gcloud compute instances stop $INSTANCE_NAME`
    * Restart the VM after stopping
        * `gcloud compute instances start $INSTANCE_NAME`
        * `gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080`
    * Clean up and delete instance
        * `gcloud compute instances delete $INSTANCE_NAME`

* Change scope and service account for the VM [optional] 
    * Create a dedicated [service account](https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances#createanewserviceaccount) for the VM with editor role
    * `gcloud compute instances set-service-account example-instance \
   --service-account compute-engine-vm-nlp-pytorch@ml-deployment-apps-9461.iam.gserviceaccount.com \
   --scopes logging-write,storage-full,monitoring-write,service-control,service-management,trace,pubsub`
   * execute the above command to assign compute-engine-vm-nlp-pytorch@ml-deployment-apps-9461.iam.gserviceaccount.com as service account for the VM and set scopes for this VM
   * scope and IAM role together will determine the capability of the VM

## Connect with github repo
```
cd ~
git clone https://github.com/rbiswasfc/nlp-deployments.git
cd nlp-deployments
git status
```

## Create Google Cloud Bucket & upload model
* In GCP go to storage -> cloud storage -> browser
* Click on create bucket and follow the instructions
* Alternatively the gsutil to create a bucket e.g. 
    * `gsutil mb -l asia-southeast1 gs://rbiswasfc-nlp-bucket` 
        * this will create a bucket named rbiswas-nlp-bucket in asia-southeast1 region
        * `mb` stands for make bucket
* upload trained model to the bucket
    * `gsutil -m cp -r {LOCAL_MODEL_DIR} gs://{BUCKET_NAME}/{GCS_MODEL_DIR}`


## Activate AI Platform Training & Prediction 

## Create Service Account to Access Model
* A service account is an account for an application/VM instead of an individual end user
* Applications use service accounts to make authorized API calls, authorized as
    * service account itself
    * google workspace
    * cloud identity through domain-wide delegation
* A service account is identified by its email address, which is unique to the account
* Service accounts do not have passwords, and cannot log in via browsers or cookies
* Each service account is associated with two sets of public/private RSA key pairs that are used to authenticate to Google
    * Google-managed keys
    * User-managed keys
* You can create as many accounts as needed to represent the different logical components of your application.

* In the GCP Console, go to the create service account key page

## Authentication confirms that users are who they say they are. Authorization gives those users permission to access a resource.