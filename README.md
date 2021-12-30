# Canny-edge-detection-from-scratch

Steps performed to get the below attached results: 
- A gray image from Berkeley Segmentation dataset using a python module PIL
- A gaussian filter G was created using the gaussian formula with a appropriate value of sigma (which controls the amount of blurring in the images and even affects the final results).More the value of sigma ,more is the smoothing in the image. (the best value of sigma chosen sigma=0.5)
- Calculated the derivative of gaussian along using x and y direction (Gx and Gy)  by multiplying the gaussian and the differentiation matrices.
- Smoothing of image along X axis  by convolving Image with the gaussian filter to get (Ix) and with the transpose of gaussian to get the smoothing of Image along Y axis (Iy).
- Now to detect the edges we convolve the smoothened image along X and Y axis to get Ix’ and Iy’.
- As we have got the edges in the image ,now we calculate the magnitude and direction of the gradients(edges detected) using sqrt(Ix’(x,y)2+Iy’(x,y)2)  and atan2(Iy’/Ix’)
- Now using the magnitude and the direction , we will be performing Non Maximum Suppression (NMS),where in the directions are thresholded to 4 direction ie 0,45,90 and 135. And depending on these directions we check in the direction of the gradient if it is a edge pixel or not. On performing NMS ,the extra edges are nullified.
- And Finally a hysteresis thresholding is performed to highlight the prominent edges, wherein the pixels greater than threshold are set to 255 and less than the lower threshold are set to 0 ,and the once in the range of high and low threshold are checked for connectivity with the high threshold pixel ,if found then it is set to 255 else 0.

## Below mentioned are the results obtained during the canny edge detection process

Order of images( Input,Ix,Ix’, Iy, Iy’, Mxy,NMS, Hysteresis )

![image](https://user-images.githubusercontent.com/95454351/147712887-772b16c7-4737-4b69-93da-dedcbe8a7efa.png)

Looking into the effect of sigma on the final results:
It is noticed that higher the value of sigma only a few edges are detected and in case of low sigma ,better fine edges are detected .Hence a lower value of sigma=0.5 has been chosen to perform the edge detection . Hence I would say if ( depending on the application ie fine grained or loosely spaced edges) fine grained and continuous edges are preferred then it is better that we go with lower value of sigma.
Image order(0.2,0.5,1,3,5)
Image set 1 demonstrating the affect of sigma

![image](https://user-images.githubusercontent.com/95454351/147712937-1ab38b59-b2a8-4aba-8c81-9522a3216477.png)





