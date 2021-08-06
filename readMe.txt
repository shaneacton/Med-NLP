
* Experiments
    * try backprop through precision
        * could backprop through specific class precision metrics to target classes
    * try scale losses based on label distributions

* change sampling to work with least represented class only
    * if a label includes multiple conditions, weight that example by the rarest conditions frequency
    * stops common conditions from hiding rarer ones in the same example

* log basic metrics on wandb
* config overhaul getting ready for hyperparam search
* plot max precision per class compared to number of examples
* coplot precision per class for train and test data

* tuning
    * higher weight decay for continued learning


* want to see train and test precisions per class.
    * group classes by train accuracy into buckets of size <= 6
    * for each group, plot train vs test accuracy for each class
        * size of dot is total distribution