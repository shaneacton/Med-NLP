
* tests
    * try backprop through precision
    * try scale losses based on label distributions

* change sampling to work with least represented class only
    * if a label includes multiple conditions, weight that example by the rarest conditions frequency
    * stops common conditions from hiding rarer ones in the same example

* put all artifacts of a run inside its own runfolder