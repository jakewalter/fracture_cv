# fracture_cv


Use computer vision to identify fractures in a Lidar DEM from Southern Oklahoma at the Wichita Mountains National Wildlife Refuge

1. Get a DEM and load into Python
2. On vertical and horizontal scanlines of the DEM find the local minima from with first derivative of those elevation profiles and save the minima coordinates
3. Those data points are white pixels on a black background of an image.
4. Dilate the image a bit to expand
5. Estimate Probabilistic Hough lines (https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html) (threshold need adjusting)
6. Calculate Rose plots of those fracture orientations
7. Play with skeletonized (https://skeleton-analysis.org/stable/index.html) data and look at branching and connections (more to do here)
