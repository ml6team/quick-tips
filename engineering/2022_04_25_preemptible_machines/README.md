# Training Machine Learning Models on Preemptible VMs

Training a machine learning models on Google Cloud is not only computationally expensive but also expensive for your wallet 💰. But don't let this stop you from building ML applications. In this repository we will show you this on simple trick that allows you to save a lot of money! ... We promise it isn't a scam. Google offers its excess Compute Engine capacity as preemptible VM instances at much lower rates than standard VM instances.  There is a caveat though, Google can claim these machines back at any time, even in the middle of your computations 😵. To avoid losing all your  progress while training your model, you can save checkpoints to a Google Cloud Storage bucket. We have a Tensorflow and Pytorch example notebook you can use as boilerplate. If you were to lose your preemptible VM, you can then restart training from your latest checkpoint.

More information on preemptible VM instances: 

https://cloud.google.com/compute/docs/instances/preemptible

https://cloud.google.com/compute/docs/instances/create-use-preemptible

## What exactly is in these notebooks?

In the notebooks we train a model to classify handwritten digits using the MNIST dataset. During training we create checkpoints and store these on a Google Cloud Storage bucket. You can use the functions in the notebooks to store, load and move checkpoints on GCS as boilerplate without modification. You'll just need to provide your own GCP parameters such as the project name and bucket name. The Pytorch notebook works out of the box, but for the Tensorflow you'll need to implement a function to store a checkpoint as there are many different ways to do this. The helper functions allow you to store the checkpoint on GCS in an archive or a directory. Check out the documentation for more information:

https://www.tensorflow.org/guide/checkpoint

https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

## How does it work?

To save checkpoints to GCS you can use the `save_[framework]_checkpoint` in your training loop. The function first writes the checkpoint to the local disk of the VM, then uploads it to GCS. To load the latest checkpoint from GCS you can use the  `load_gcs_checkpoint` function. It downloads the checkpoint to the local disk. You can then load the parameters using Tensorflow or Pytorch. The code is a boilerplate and serves as a guideline, you can easily modify the code to use custom save functions or add additional features tailored towards your use case.

PS. If you want to test the code, but don't want to set up a VM instance, you can also try it out on Google Colab.

