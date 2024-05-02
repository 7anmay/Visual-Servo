Notes:

Tried techniques
line fitting
hough transform with line fitting
box fitting
lines via moving a scanning window and calculating intersections
moving std method as per the paper: Towards Autonomous Visual Navigation in Arable Fields, ahmadi2021towards.

To Be Tried techniques:
Box chaining 
point regression
line based scanning
Deep learning model based segmentation



# Project Title
Crop lane visual servoing algorithm - Development Report

The following outlines the progress and challenges encountered during the development of the visual servoing algorithm. The primary objective taken into consideration is ability to be able to detect lanes robustly in crop fields. The focus has been on identifying and implementing various image processing techniques that can reliably detect and follow crops under certain degree of variaitons.

## Implementation Pipeline:
1. Crop Segmentation
    1. Preprocess image ✔️
    2. Segment green colour to detect crops ✔️ 
    3. Denoise mask ✔️
    4. Find contours ✔️
    5. Find centers ✔️
2. Median Lane Detection
    1. shape fitting
        a. Line fitting ❌
        b. Hough transforms ❌
        c. Box fitting ✔️
        d. Centeroid based techniques ⚠️
        e. Sliding window ⚠️ (needs tuning)
    2. Individual crop row detection
        a. Moving variance signal method ⚠️ (needs tuning)
        b. Box Chaining method ⚠️ - Todo
3. Error calculation
    a. Calculated from the median lane vector ✔️

Possible directions that could habe been taken:
- Deep Learning based segmentation methods
- Feature point matching type methods

## Experiments

### Experiment 1
Line fitting through hough trasnforms:
![Line detection](exps/houghLines.png)


### Experiment 2

Box Fitting
![Box on crop lanes](exps/boxOnLanes.png)

## Results

Summary of the results obtained from the experiments.

## Methodology

Explanation of the methodology used in the project.

## Repository Installation and Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/username/repository.git
    ```

2. Install the dependencies:

    ```bash
    npm install
    ```

3. Run the project:

    ```bash
    npm start
    ```

## Conclusion

Closing thoughts and final remarks about the project.