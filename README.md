# Microbes-Identification-with-Machine-Learning
 Identification of microbes with machine learning
 <img src='https://img.freepik.com/free-vector/bacteria-shapes-realistic-icons_1284-74343.jpg?w=996&t=st=1664897947~exp=1664898547~hmac=3be3c7350d39f8809cfc3ddc6e4695d51e1c568aa1275b50ab9c1f143ba35639'>
 
## Libraries Used
* Pandas
* Matplotlib
* Seaborn
* scikit-learn
* XGBOOST
* treeplot
 
## Contents
* Problem Defination
* Why I am doing this project?
* Proposed Solution
* Objective
* Data Summary
* Approach
* Packages Import
* Exploratory Data Analysis
* Data Visualization
* Data Preprocessing
* ML Modelling
  * Logistic Regression
  * Naive Bayes Classifier
  * Decision Tree Classifier
  * Random Forest Classifier
  * XGBOOST Classifier
  * Features Selection with XGBOOST
  * Training XGBOOST with Best 6 Features
* Saving Model Weights
* XGBOOST Model Deployment with Streamlitm

## 📖 Problem Defination
Bacteria are responsible for causing a wide variety of infectious diseases, including pneumonia, septicemia, meningitis, urinary tract infections, and endocarditis. To identify the specific pathogen causing the infection, doctors often rely on traditional methods such as culturing the organism in a laboratory dish. However, this approach may take weeks or even months before results are known.

Novel DNA sequencing technologies have proliferated over the past two decades. Continuous improvements in "next generation sequencing" (NGS) and "third generation sequencing" (TGS) have increased the fidelity and rate of sequencing, but it still takes hours or days to obtain a complete sequence. Furthermore, there are some clinical applications in which very rapid identification of a particular gene or genetic species becomes necessary, whereas identification of all genes is not necessary.

## Why I am doing this project?
In patients with septic shock from bacterial infection, identification of antibiotic-resistance genes is essential because mortality increases with a delay of 7.6% per hour in administering the correct antibiotics. Unfortunately, it takes more than 24 hours for bacteria recovered from an infected patient's blood to grow, Identifying the kind of microbe responsible for a infection is critical because certain kinds of bacteria are sensitive to particular antibiotics. Once the species of bacteria is identified, doctors can prescribe the right medication to treat the infection.

## Proposed Solution
Bacterial antibiotic resistance is becoming a significant health threat, and rapid identification of antibiotic-resistant bacteria is essential to save lives and reduce the spread of antibiotic resistance.

So we will train a supervised machine learning classification model to identify microbes from Generalized Segmentation Algorithm (GSA) Data

## 🔎 Objective
Identify the ten different micro-forms of life from Generalized Segmentation Algorithm (GSA) Data

## 💾 Data Summary

* Solidity: It is the ratio of area of an object to the area of a convex hull of the object. Computed as Area/ConvexArea.
* Eccentricity: The eccentricity is the ratio of length of major to minor axis of an object.
* EquivDiameter: Diameter of a circle with the same area as the region.
* Extrema: Extrema points in the region. The format of the vector is [top-left top-right right-top right-bottom bottom-right bottom-left left-bottom left-top].
* Filled Area: Number of on pixels in FilledImage, returned as a scalar.
* Extent: Ratio of the pixel area of a region with respect to the bounding box area of an object.
* Orientation: The overall direction of the shape. The value ranges from -90 degrees to 90 degrees.
* Euler number: Number of objects in the region minus the number of holes in those objects.
* Bounding box: Position and size of the smallest box (rectangle) which bounds the object.
* Convex hull: Smallest convex shape/polygon that contains the object.
* Major axis: The major axis is the endpoints of the longest line that can be drawn through the object. Length (in pixels) of the major axis is the largest dimension of the object.
* Minor axis: The axis perpendicular to the major axis is called the minor axis. Length (in pixels) of the minor axis is the smallest line connecting a pair of points on the contour.
* Perimeter: Number of pixels around the border of the region.
* Centroid: Centre of mass of the region. It is a measure of object’s location in the image.
* Area: Total number of pixels in a region/shape.
* microorganisms : [Spirogyra, Volvox, Pithophora, Yeast, Raizopus, Penicillum, Aspergillus sp, Protozoa, Diatom, Ulothrix]

## 🛬 Approach
* Supervised Learning
* Classification
* Multi-class Classification

## Packages Import
 Imported all the required packages
 
## Exploratory Data Analysis
 * No Null Values are Found in dataset
 * Almost all columns are of float datatype
 * Rows in dataset :  30527
 * Columns in dataset :  25
 * Descrptive Stats : Min , Max, Mean and Std are obtained with pandas describe method
 * Distribution plots
## Data Preprocessing

 * Seprating input and output columns
 * Splitting Dataframe into Input and Target dataframe
 * Train, Validation and Test Split
 * Data Normalization

## ML Modelling

### Logistic Regression
### Naive Bayes
### Decision Tree Classifier
    * Hyper Parameter Tunning for Decision Tree Classifier
    * Decision Tree visualization
    * Feature Importance
### Random Forest Classifier
    * Feature Importance
### XGBOOST Classifier
    * Label Encoding target column
### Models Accuracy on Test Data
   * DTC1 Accuracy :  0.9865125240847784
   * RFC Accuracy :   0.9884393063583815
   * XGB Accuracy :  0.9922928709055877

### Features Selection with XGBOOST
    * Top 6 Features :  'ConvexArea', 'raddi', 'FilledArea', 'MinorAxisLength', 'Perimeter', 'Solidity'
### Training XGB with Top 6 Features only

### Saving Model 
Model  and label encoder are saved using joblib library

### Web Deployed App Link
<a href="https://omjiverma-microbes-identi-microbes-identification-webapp-qh6iaz.streamlitapp.com/">Open Webbapp</a>
