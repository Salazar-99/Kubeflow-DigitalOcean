# Kubeflow-DigitalOcean
Deploying Kubeflow on a Digital Ocean managed Kubernetes Cluster

## Cluster Setup
Use the UI to create a cluster
Download doctl and set it up with an Auth token
Configure your kubectl context with the automated command
Create a namespace for the kubeflow resources
Install Kubeflow with kfctl
 - Download kfctl tarball from releases on repo
 - Download and modify KfDef file
 - Generate the configuration
 - Apply the configuration with kfctl
 - Fix dex by adding env variable due to Kubernetes version issue
Configure a LoadBalancer by patching the existing istio-ingressgateway service
