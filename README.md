# Fish detection


Extract the data by running `main.py`


#### Train the images with

Run `train.py`

Validation might automatically start after training is done, and will result in an error. This is just a flaw with YOLO, and the real validation should be done as described below


#### Validate the images with

Run `validate.py`



#### Track / Test the images with

Run `track.py`
There may be an issue when the amount of videos are over 15, but this varies. We solev this by running 15 videos at a time

OR

Run my amazingly done code at `compile_predict.py`
If that doesn't work run `predict.py`, that is located in the scripts directory