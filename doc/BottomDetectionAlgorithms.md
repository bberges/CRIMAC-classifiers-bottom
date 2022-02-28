# Bottom detection algorithms

This document summarizes some implementation details of the bottom detection algorithms listed in the [README](../README.md).

## The `constant` algorithm

The detected bottom is set to a constant depth.
This algorithm is mainly intended for testing and debugging.
It is very fast since no data is read.

It is possible to specify the bottom depth by appending it to the algorithm name. For example, setting the algorithm
to `constant250` results in a bottom depth of 250 m.

## The `simple` algorithm

This algorithm is fast and works quite well in many cases. A main exception is when there are strong schools of fish
near the seabed.

It consists of the following steps:
1. **Find the *max strength bottom depth*.**<br>
   Find the maximum s<sub>v</sub> in each ping.
   Check that this is at the first bottom echo:
   If the s<sub>v</sub> at half the depth is sufficiently strong,
   then the maximum s<sub>v</sub> is probably at the second bottom echo.
   In this case, use half the *max strength bottom depth*.  
2. **Detect the presence of bottom.**<br>
   If the s<sub>v</sub> at the *max strength bottom depth* is less than a given threshold, then detect no bottom.
3. **Do backstepping.**<br>
   Backstep from the *max strength bottom depth* to the sample with s<sub>v</sub> less than a given fraction
   of the maximum s<sub>v</sub>. This is the detected bottom depth. 

## The `angles` algorithm

This algorithm uses split beam angles to detect the bottom depth, and is mainly intended to be used as a part of 
the [combined](#the-combined-algorithm) algorithm.
It is designed for parts of the data where the bottom is sloping and one or
both of the angles are varying linearly with depth.

1. **Filter the alongship and athwartship angles.**<br>
   This is done by applying a median filter.
   Do not use angles of samples where s<sub>v</sub> has a local minimum.
   Such samples often have outlier angles.
2. **Find the central range of the bottom echo.**<br>
   This is done by finding the 10% and the 90% quantiles of the cumulative s<sub>v</sub> around the first bottom echo.
3. **Compute the angle indicator functions.**<br>
   For each sample, consider the set of samples down to lower end of the *central bottom range*.
   Compute the linear regressions of the alongship and athwartship angles for this set of samples.
   The angle indicator function is set to the
   [R<sup>2</sup>](https://en.wikipedia.org/wiki/Coefficient_of_determination) value of the linear regression.
4. **Select the best angle indicator function.**<br>
   Select the alongship or athwartship angle indicator function that has highest R<sup>2</sup> values.
5. **Find where the angles stop varying linearly.**<br>
   At these locations the angle indicator will drop off significantly.
   Convolve the angle indicator function with an edge detection kernel and search for maxima.
   Possibly step a bit downward from a convolution maximum to corresponding the R<sup>2</sup> breaking point. 
6. **Define bottom candidates.**<br>
   The quality of each candidate is based on the R<sup>2</sup> value and the convolution peak prominence.

If this algorithm is used directly, then the bottom depth is set to the bottom candidate with the highest quality.

## The `edge` algorithm

This algorithm uses the volume back scatter, s<sub>v</sub>, and is mainly intended to be used as a part of the 
[combined](#the-combined-algorithm) algorithm.
It is designed for parts of the data where the bottom is flat and has a clear echo.

1. **Convolve s<sub>v</sub> with an edge detection kernel.**<br>
   The size of the kernel is based on the size of the central range of the bottom echo.
   The kernel is defined asymmetrically with more weight on its first part.
2. **Find convolution peaks.**<br>
   Possibly step a bit upward if s<sub>v</sub> decreases.
3. **Define bottom candidates.**<br>
   The quality of each candidate is based the width and prominence of the convolution peaks.

If this algorithm is used directly, then the bottom depth is set to the bottom candidate with the highest quality.

## The `combined` algorithm

This algorithm combines the [angles](#the-angles-algorithm) and the [edge](#the-edge-algorithm) algorithms.

### Part 1: For each ping

1. **Detect the presence of bottom.**<br>
   Find the *max strength bottom depth* as in the [simple](#the-simple-algorithm) algorithm.
   If the s<sub>v</sub> at the *max strength bottom depth* is less than a given threshold, then detect no bottom.
2. **Select the `angles` or the `edge` algorithm.**<br>
   Use the `angles` algorithm if the angle regression fits well enough
   and the central range of the bottom echo is large enough.
   Otherwise, use the `edge` algorithm.
3. **Define bottom candidates.**<br>
   Use the bottom candidates defined by the selected algorithm.

### Part 2: Across pings

1. **Alternative 1: Use the best bottom candidate for each ping.**<br>
   This is what is currently implemented. 
2. **Alternative 2: Optimize across pings.**<br>
   This is a possible future improvement.
   1. **Define a cost function for a selection of bottom candidates.**<br>
      The cost function can use the bottom candidate qualities
      and the depth difference between bottom candidates at neighbouring pings.
   2. **Minimize the cost function.**<br>
      Find the bottom candidates that minimize the cost function.

## The `work_files` algorithm

This algorithm uses the LSSS work files. The detected bottom is set to the lowest layer boundary.
