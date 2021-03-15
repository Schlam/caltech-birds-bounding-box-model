# Object detection with Transfer Learning

We build the following branch model using a pretrained base
(the top layers and two residual blocks from EfficientNetB3)

## Model

![Brach Model](images/model.png)

## Data

This model is trained using the images of birds and their accompanying bounding boxes
(coordinates for the bounds of a region for a given image).


    @techreport{WelinderEtal2010,
        Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
        Institution = {California Institute of Technology},
        Number = {CNS-TR-2010-001},
        Title = {{Caltech-UCSD Birds 200}},
        Year = {2010}
    }