
* Experiments
    * try backprop through precision
        * could backprop through specific class precision metrics to target classes
    * try scale losses based on label distributions

* change sampling to work with least represented class only
    * if a label includes multiple conditions, weight that example by the rarest conditions frequency
    * stops common conditions from hiding rarer ones in the same example

* put all artifacts of a run inside its own runfolder
* config overhaul getting ready for hyperparam search
* plot max precision per class compared to number of examples
* coplot precision per class for train and test data

* tuning
    * higher weight decay for continued learning