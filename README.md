
This code accompanies the paper

**Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic and Nassir Navab:
SSD-6D: Making RGB-Based 3D Detection and 6D Pose Estimation Great Again. ICCV 2017.**  [(PDF)]( https://www.dropbox.com/s/k6cbdb77d6ubdfv/kehl2017iccv.pdf?dl=0)


and allows to reproduce parts of our work. Note that due to IP issues we can only 
provide our trained networks and the inference part. This allows to produce the detections and 
the 6D pose pools. **Unfortunately, the code for training as well as 2D/3D refinement cannot be made available.**

In order to use the code, you need to download
* the used datasets (hinterstoisser, tejani) in 
SIXD format (e.g. from [here](http://cmp.felk.cvut.cz/sixd/challenge_2017/) )
* the trained networks from [here](https://www.dropbox.com/sh/08mv6fmzi6dylj2/AACIK-odLEu560UDUs2WY9Sba?dl=0)

and use the run.py script to do the magic. Invoke 'python3 run.py --help' to see the available commands. 
For the correct thresholds you should look at the supplemental material.

Fabian Manhardt now also added a benchmark.py that allows you to properly run the evaluation on a sequence and produce metrics. Note, though, that these numbers are without the final pose refinement.






