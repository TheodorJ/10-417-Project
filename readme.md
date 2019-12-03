The following code more optimally searches for hyperparameters using a transfer-learning based approach.

## Installation

Install requirements in requrirements.txt

Most files will be using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) datasets. CIFAR-10 will be downloaded automaticallty, but Caltech-101 needs to be downloaded manually and placed in the data/ directory.

## Running the Code

The code is broken down into multiple files depending on the hyperparameter search algorithm. 

* nsvm.py uses a neural SVM at the end of training.
* expand.py uses mutation learning based on beam search to continue training hyperparameters that perform wel.
* downsample.py uses downsampling as pretraining to find the best hyperparameters before passing the parameters back up to a higher resolution image set.
