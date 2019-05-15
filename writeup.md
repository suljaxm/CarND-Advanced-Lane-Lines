## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Modularized the previous code
[//]: # (Image References)

[image1]: ./writeup/undistorted.png "Undistorted"
[image2]: ./writeup/undistorted_lane.png "Road Transformed"
[image3]: ./writeup/Binary.png "Binary Example"
[image4]: ./writeup/warped.png "Warp Example"
[image5]: ./writeup/fit.png "Fit Visual"
[image5_1]: ./writeup/peaks.png "Histogram Peaks Visual"
[image6]: ./writeup/Output.png "Output"
[video1]: ./project_video_output.mp4 "Video"

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced_Lane_lines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step. I applied color transforms and gradients to the test image using the `binary_select()`  function and obtained this result.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birds_eye_warp()`, which appears in `Step4`  in the `Advanced_Lane_lines.ipynb`.  The `birds_eye_warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = img.shape[:2]
src = np.float32([[575,464], [707,464], 
                  [1049,682],[258,682]])
dst = np.float32([[150,0],[w-150,0],
                  [w-150,h],[150,h]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575,464      | 150,0      | 
| 707,464      | 1130,0      |
| 1049,682     | 1130,720      |
| 258,682      | 150,720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff in `Step5`. First, I find the pixels that belong to the lane line by the code `hist()` function. I chose the two strongest peaks as the starting point for the initial lane line. Second, using the strategy of  `Sliding Windows`, I selected the lane line pixels for each segment.
![alt text][image5_1]
The last, I fit lane lines with a 2nd order polynomial kinda. the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve is `f(y) = Ay^2 + By + C`, and use `numpy.polyfit()` function.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The [radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) calculated in the code `measure_curvature_real()` function in `Step6`.
The lane-line for the radius of curvature at any point x for the curve` x=f(y)` is given by: 
`R_curve = ((1+( dy/dx)^2)^1.5)  /  (|((d^2)x)/dy^2|)`.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code is
`center_dist = (car_position - lane_center_position) * xm_per_pix`
Since the camera is mounted in the center of the car, this is equivalent to taking half the width of the image, Then `car_position = img.shape[1]/2`. And the `lane_center_position`  is calculated by actually detecting the lane line along the x axis in the img. The last, `xm_per_pix` is the relationship between image pixels and the actual distance.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Step7` and `Step8`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I encountered were almost entirely due to lighting conditions, shadows, and other vehicles. In the binarization, I sometimes cannot eliminate noise well, which leads to the detection of wrong lane line detection.

Therefore, we can do some work on how to reduce the noise. We can improve the quality of the picture by using a method similar to deep learning, or obtain the position of the lane line directly through deep learning. 

As for how to use deep learning to realize lane line detection, my initial idea is to detect lane lines by marking lane lines in images and then detect lane lines with the help of target recognition network.