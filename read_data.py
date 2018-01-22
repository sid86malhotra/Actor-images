import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from scipy.misc import imresize

#Convert the RGB image to Grayscale.
def RGBtoGRAY(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

#Read in the training and test CSV file into a numpy array to get the list of file names.
TrainingCSVfile = pd.read_csv("train.csv").as_matrix()
TestCSVfile = pd.read_csv("test.csv").as_matrix()


def convertRGBimagetoGRAY(Filenamelist, CSVoutputname, Datatype):
    """Here we will convert the RGB images listed in the CSV files to grayscale.
    Then we would save them in a numpy array and finally save them back in a csv file.

    filenamelist = CSV file with filenames
    CSVoutputname = Outputname of the CSV file
    Datatype = If it is a training dataset or test dataset
    """

    #Get the list of filenames into a numpy array.
    totalfiles = Filenamelist[:,0]

    #Setup a empty array to collect the data
    IMAGEArray = []

    for idx, image in enumerate(totalfiles): #Loop through all the images.
        IMG = Image.open(''.join([Datatype, image]))
        ConvertedIMG = RGBtoGRAY(np.array(IMG)) #Convert to the RGB image numpy array and then to grayscale
        Resizedimage = imresize(ConvertedIMG, (48,48)) #Resize the image to 48 * 48 pixels.
        Flattenedimage = Resizedimage.flatten() #Flatten the image to 1D array
        IMAGEArray.append(Flattenedimage) #Add the data to our empty array
        if idx % 1000 == 0:
            print("Number of", idx, "files are read in the", Datatype[:-1], "dataset")

    NUMPYIMAGEArray = np.array(IMAGEArray) #Convert the array to Numpy array.
    # print(totalfiles[:, None])
    # print(NUMPYIMAGEArray.shape, totalfiles.shape)
    # print(NUMPYIMAGEArray)

    Finaloutputimage = np.concatenate((Filenamelist, NUMPYIMAGEArray), axis = 1)
    #Merge the filenames + category to the 48 * 48 pixels values

    df = pd.DataFrame(Finaloutputimage) #Convert to DataFrame, easier to export
    df.to_csv(CSVoutputname, index = False) #Export the data to CSV without the Index
    print("Number of", idx, "files are read in the", Datatype[:-1], "dataset")

convertRGBimagetoGRAY(TrainingCSVfile, "Training.csv", "Train/")
convertRGBimagetoGRAY(TestCSVfile, "Test.csv", "Test/")
