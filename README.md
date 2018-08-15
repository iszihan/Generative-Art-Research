
# Generative-Art-Research
Neural networks to extract mid-level such as roundness, or messiness from images, especially abstract images. Training is done on a controlled set of generated images with continuous levels of these parameters.

- **Abstract-Feature-Recognition** Contains the training, testing, and predicted code as well as helper files, and the image spatialization applet.
- **ImageAnalysis** Contains the analyzing function as well as the training, testing, and predicted code.
- **Processing Sketches** Contains the Processing scripts used to generate the training images

## How to Train on an Image Set (Use the code in the Abstract-Feature-Recognition folder)
Run the python file Parameters.py like this:

    python3 Parameters.py train IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -v MODEl_TYPE -e EPOCHS -m PARAMETER -p N_PARAM -g GPU_TO_USE
    EXAMPLE:
    Loading existing VGG19 model:
    python3 Parameters.py train "['Images/MessyRoundRects/','Images/Noise/']" -v 1 -l Models/Trial1 -n Results -e 300 -t tri -m "['r','m']" -p 2 -g 0
    Creating new Hans model:
    python3 Parameters.py train "['Images/MessyRoundRects/','Images/Noise/']" -v 2 -n Results -e 300 -t tri -m "['r','m']" -p 2 -g 1

 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (-l):** If this field is specified, the program will look in the folder provided and start the training with the model loaded from there.
 - **RUN_ID (-n):** The results will be saved in a folder with this name. If this is not specified, a name will be created based on other information.
 - **MODEL_TYPE(-v):** If no MODEL_TO_LOAD is specified, this has to be specified in terms of which model to create (1=VGG19 Model/ 2 = Customized Model). Default value is 1.
 - **EPOCHS (-e):** The number of epochs to train on the data
 - **PARAMETER (-m):** Parameters for training
 - **N_PARAM (-p):** Number of Parameters for training
 - **GPU_TO_USE (-g):** Which GPU/CPU to use for this training operation. (Depending on the devices of the user.)


## How to Predict (Use the code in the Abstract-Feature-Recognition folder)
Run the python file Parameters.py like this:

    python3 Parameters.py predict IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -v MODEl_TYPE -m PARAMETER -p N_PARAM

 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (required)(-l):** Model used to predict.
 - **RUN_ID (-n):** The results will be saved in a folder with this name. If this is not specified, the loaded model's folder will be used
 - **MODEL_TYPE(-v):** If no MODEL_TO_LOAD is specified, this has to be specified in terms of which model to create (1=VGG19 Model/ 2 = Customized Model). Default value is 1.
 - **PARAMETER (-m):** Parameters for training
 - **N_PARAM (-p):** Number of Parameters for training

 Other options aren't included for now for predicting, because prediction images are assumed to be unlabeled.

## How to Test (Use the code in the Abstract-Feature-Recognition folder)
Run the python file Parameters.py like this:

    python3 Parameters.py test IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -v MODEL_TYPE -e EPOCHS  -m PARAMETER -p N_PARAM

 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (required)(-l):** Model used to test
 - **RUN_ID (-n):** Results are saved in a folder with this name if the field is specified, otherwise it uses the existing folder.
 - **MODEL_TYPE(-v):** If no MODEL_TO_LOAD is specified, this has to be specified in terms of which model to create (1=VGG19 Model/ 2 = Customized Model). Default value is 1.
 - **EPOCHS (-e):** The number of epochs to train on the data
 - **PARAMETER (-m):** Parameters for training
 - **N_PARAM (-p):** Number of Parameters for training

## How to Analyze (Use the code in the ImageAnalysis folder)
Run the python file Parameters.py like this:

     python3 Parameters.py analyze IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -v MODEL_TYPE -m PARAMETER -d N_DIVISION
     EXAMPLE:
     To analyze two images in Evo_Art folder with 5*5=25 subimages with customized model:
     python3 Parameters.py analyze "['Evo_Art/image-15.jpg','Evo_Art/image-19.jpg']" -v 1 -l Models/Trial1 -n Results -m "['r']" -d 5

- **IMAGE_DIR (required):** The image paths.
- **MODEL_TO_LOAD (required)(-l):** Model used to analyze
- **MODEL_TYPE(-v):** If no MODEL_TO_LOAD is specified, this has to be specified in terms of which model to create (1=VGG19 Model/ 2 = Customized Model). Default value is 1.
- **RUN_ID (-n):** Results are saved in a folder with this name if the field is specified, otherwise it uses the existing folder.
- **PARAMETER (-m):** Parameters for analysis
- **N_DIVISION (-D):** Number of smaller parts you wish to divide the input image on single axis (x or y) for further analysis.

## How to Spatialize the input images (Use Vis.py in the Abstract-Feature-Extraction folder)
1. Install PyQt4 by 'conda install pyqt=4'
2. Run Parameters.py's Predict function first (See ReadMe.md) for targeted images, and save the result folder in the same directory as Vis.py.
3. Run this applet like below:
    To visualize results in PredictionCSVFolder that has only 1 parameter:
        python3 Vis.py PredictionCSVFolderName -p 1
    To visualize results in PredictionCSVFolder that has 2 parameters and spatialize the images according to both parameters:
        python3 Vis.py PredictionCSVFolderName -p 2
    To visualize results in PredictionCSVFolder that has 2 parameters and spatialize the images according to the first parameters:
        python3 Vis.py PredictionCSVFolderName -p 2 -n 1

## Image Format and Files
The images should be in the format **img-c##-r##-###.png**
The first three letters, **img** are the type of the image. Images generated by the same script should have the same type.
(Note:The above formats are abandoned in this version of the repo. There might still be old codes left associated with it, but it is not necessary here to distinguish and use the images with the first three letters.)
Each parameter **-c##** is designated by a dash followed by a letter. This is how the program recognize the parameter labels. *r* might be *roundness.*
At the end is a string of numbers, **###**, used to keep multiple images with the same parameter values unique.

The data used for the project during J-term is stored on basin, here:
http://www.cs.middlebury.edu/~hgoudey/Generative-Art-Research-Data/
The processing scripts in the repository could also just be run to generate the images again from scratch.

### Remove Junk Images
Using a Firefox extension to download Google Images search results, I ended up with quite a few images that my script can't open. If the quality of the images is unknown, you can run the **Remove_Bad_Images.py** script on the directory first.


#TO DO
- Iterate on the network. Will smaller convolution filters from the start work as well? Explore more hyper-parameters.
- Symmetry Parameter
- Repetition Parameter
- Blur Parameter
