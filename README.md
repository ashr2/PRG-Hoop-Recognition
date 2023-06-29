# PRG-Hoop-Recognition

## Setting up training data
The hoop images used to train this model are already set up. 
In the assets directory you must create a folder called training_data that contains all your background images that will be used in training.
For my purposes I used the COCO Dataset from 2017 at https://cocodataset.org/#home

If you have a video that you want to collect frames from, you can use the videotjpgs.py utility by modify the path on line 3 to where your video is and 
running the program.

## Training
After your training directory is set up, run visualize.py to begin training the model.
You can modify model_path to save the model where you need it to.
You can also modify the NUM_EPOCHS variable as necessary.
This program will save the model after validation if it is the best performing model up to that point.

## Seeing results
You can install tensorboard and run tensorboard --logdir=logs/fit to view the Tensorboard containing
the results of training, validation, and the images produced by running the model on the training set and the validation set.

If you want to see the model's performance on a video not included in the dataset, you can use the model.py module.
To use this, go to model.py, modify the model_path variable so it is where your model is saved, and then modify your video_path variable
so that it points to where the video you want to see is stored.
Then run model.py to see the side-by-side comparison of the actual video and the model's prediction of where the hoop is.
