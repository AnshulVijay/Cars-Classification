# Cars-Classification
Image classification for three different types of cars

Follow the steps to perform the classification:
1. Clone the repository
2. From Image Dataset Generator firstly run the Dataset Generator.py to download the images from the google, modify the code accordingly to change the dowloading site and no of images.
3. Run augmentation.py to generate more no of images by running python augmentation.py -folder=Cars -limit=10000 
            Augmnetation code referenced from "https://github.com/tomahim/py-image-dataset-generator" 
4. You got your Dataset now, get ready to perform classification.
5. Open Classification Using Defined Model 
6. Run image classification.py to retrain the model or you can directly test by using given weights and yaml files
7. To perform prediction run PredictForMyModel.py by entering python PredictForMyModel.py path to test image
8. You will get the result.
9. To see the Reporting metrics you can see PredictForMyModel.ipynb where i showed the Confusion metrics and classification report.
10. Now to do the classification on predefined weights of resnet50, go to Classification Using Resnet50 
11. Do the same as earlier to retrain and predict the model.
12. Get the trained .h5 file from the link of my drive "https://drive.google.com/open?id=1GhZRYzblgoyNZS_zS6gxzYncuHUn66Aa"
13. I provided some test images to check for the prediction of cars.

Thank you
