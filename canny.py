from PIL import Image
import numpy as np
import cv2
import math 

def gaussian_filter(sig):
    side = int(np.ceil(3 * sig)) # values upto  3sigma
    x = np.linspace(-side,side, 2*side + 1, dtype=float) # creating a one dim from +3sigma to -3sigma
    kernel = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-x**2 / (2 * sig**2))  #applying gaussian to all the values
    kernel = kernel / np.sum(kernel) # make sure the sum of weights in kernel = 1
    return kernel


def derivative_x():# computes derivative along x axis but gives verticle edges
    Filter=gaussian_filter(3)
    Filter=np.array(Filter)[np.newaxis]
    diff=np.array([[1,0,-1]])
    return np.dot(Filter.T,diff)

def derivative_y(): #computes derivative along y axis but gives horizontal edges
    Filter=gaussian_filter(3)
    Filter=np.array(Filter)[np.newaxis]
    diff=np.array([[1],[0],[-1]])
    return np.dot(diff,Filter)


# def convolve(I,F,stride=1): #applicable for sqaure filters and images only
#     output_shape=int(((I.shape[0]-F.shape[0])/stride)+1)
#     assert output_shape==int(output_shape),"Convolution not possible"
#     output=np.zeros([output_shape,output_shape], dtype = int)
#     x=0
#     for con1 in range(output_shape): #this lines accounts for convolution along the column
#         for con in range(output_shape): # this line accounts for convolution along the row
#             for row in range(con1,con1+F.shape[0]):  # next 3 lines constitutes for just 1 convolution
#                 for col in range(con,con+F.shape[1]):
#                     print(I[row][col],F[row-con1][col-con])
#                     x+=I[row][col]*F[row-con1][col-con]
#                 print("done with one conv")
#             output[con1][con]=x
#             x=0
#     return output

# def convolve_X(I,F,stride=1):
#     output_shape=int(((I.shape[0]-F.shape[1])/stride)+1)
#     assert output_shape==int(output_shape),"Convolution not possible"
#     output=np.zeros([I.shape[0],output_shape], dtype = int)
#     x=0
#     for con1 in range(I.shape[0]): #this lines accounts for convolution along the column
#         for con in range(output_shape): # this line accounts for convolution along the row
#             for row in range(con1,con1+F.shape[0]):  # next 3 lines constitutes for just 1 convolution
#                 for col in range(con,con+F.shape[1]):
#                     x+=I[row][col]*F[row-con1][col-con]
#             output[con1][con]=x
#             x=0
#     return output

# def convolve_Y(I,F,stride=1):
#     output_shape=int(((I.shape[0]-F.shape[0])/stride)+1)
#     assert output_shape==int(output_shape),"Convolution not possible"
#     output=np.zeros([output_shape,I.shape[0]], dtype = int)
#     x=0
#     for con1 in range(output_shape): #this lines accounts for convolution along the column
#         for con in range(I.shape[0]): # this line accounts for convolution along the row
#             for row in range(con1,con1+F.shape[0]):  # next 3 lines constitutes for just 1 convolution
#                 for col in range(con,con+F.shape[1]):
#                     x+=I[row][col]*F[row-con1][col-con]
#             output[con1][con]=x
#             x=0
#     return output

def convolve_test(I,F,stride=1):
    output_shape1=int(((I.shape[0]-F.shape[0])/stride)+1)
    output_shape2=int(((I.shape[1]-F.shape[1])/stride)+1)
    output=np.zeros([output_shape1,output_shape2], dtype = int)
    x=0
    for con1 in range(output_shape1): #this lines accounts for convolution along the column
        for con in range(output_shape2): # this line accounts for convolution along the row
            for row in range(con1,con1+F.shape[0]):  # next 3 lines constitutes for just 1 convolution
                for col in range(con,con+F.shape[1]):
                    x+=I[row][col]*F[row-con1][col-con]
            output[con1][con]=x
            x=0
    return output


def padding(Image,pad): #complete row and column padding
    output_shape1=Image.shape[0]+pad*2
    output_shape2=Image.shape[1]+pad*2
    output=np.zeros([output_shape1,output_shape2], dtype = int)
    for i in range(pad,output_shape1-pad):
        for j in range(pad,output_shape2-pad):
            output[i][j]=Image[i-pad][j-pad]
    return output

def column_padding(Image,pad): 
    output=[]
    zeros=np.zeros((1,pad))
    for i in range(len(Image)):
        x=np.hstack((Image[i],zeros[0]))
        output.append(x)
    return np.array(output)

def row_padding(Image,pad):
    output=[]
    zeros=np.zeros((1,Image.shape[1]))
    x=Image
    for i in range(pad):
        x=np.vstack((x,zeros))
    output.append(x)
    return np.array(output[0])



def magDir(I1,I2): #calculate the magnitude of edge response
    if I1.shape[0]>I2.shape[0]: #padding to make the shape of 2 matrices same
        pad_row=I1.shape[0]-I2.shape[0]
        I2=row_padding(I2,pad_row)
    if I2.shape[0]>I1.shape[0]:
        pad_row=I2.shape[0]-I1.shape[0]
        I1=row_padding(I1,pad_row)
    if I1.shape[1]>I2.shape[1]:
        pad_col=I1.shape[1]-I2.shape[1]
        I2=column_padding(I2,pad_col)
    if I2.shape[1]>I1.shape[1]:
        pad_col=I2.shape[1]-I1.shape[1]
        I1=column_padding(I1,pad_col)
    output_m=np.empty([I1.shape[0],I1.shape[1]], dtype = int)
    output_d=np.empty([I1.shape[0],I1.shape[1]], dtype = int)
    for i in range(0,I1.shape[0]):
        for j in range(0,I1.shape[1]):
            mag=math.sqrt(math.pow(I1[i][j],2)+math.pow(I2[i][j],2)) #magnitude of each pixel
            direc=math.atan2(I2[i][j],I1[i][j]) #direction of gradients in radians
            direc=direc*180/math.pi
            direc=direction_threshold(direc)
            output_m[i][j],output_d[i][j]=mag,direc
    return output_m,output_d

def direction_threshold(dir): # thresholding the direction to one of the 4 directions 0,45,90,135
    if dir<0:
        dir+=180
    if 0<= dir <22.5 or 157.5 <= dir<=180:
        return 0
    elif 22.5 <= dir < 67.5:
        return 1
    elif 67.5 <= dir < 112.5:
        return 2
    elif 112.5 <= dir < 157.5:
        return 3


def NMS_filter(Im1,Im2): # if neighbouring pixel magnitude is greater than the current pixel then make the current pixel zero
    I1=padding(Im1,1) # padding the image to make the pixel comparison easier
    I2=padding(Im2,1)
    output=np.zeros([I1.shape[0],I1.shape[1]], dtype = int)
    for i in range(1,I1.shape[0]-1):
        for j in range(1,I1.shape[1]-1):
            if I2[i][j]==0: # comparing pixels depending upon its gradient direction
                if I1[i][j] > I1[i][j-1] and I1[i][j] > I1[i][j+1]: #checking neighbours along the 0 degrees 
                    output[i][j]=I1[i][j]
            elif I2[i][j]==1:
                if I1[i][j] > I1[i-1][j+1] and I1[i][j] > I1[i+1][j-1]: #checking neighbours along the 45 degrees 
                    output[i][j]=I1[i][j]
            elif I2[i][j]==2:
                if I1[i][j] > I1[i-1][j] and I1[i][j] > I1[i+1][j]: #checking neighbours along the 90 degrees 
                    output[i][j]=I1[i][j]
            elif I2[i][j]==3:
                if I1[i][j] > I1[i-1][j-1] and I1[i][j] > I1[i+1][j+1]: #checking neighbours along the 135 degrees 
                    output[i][j]=I1[i][j]
    cv2.imwrite("output_nms.jpg",np.uint8(output))
    thresholded=Thresholding(output)
    return output,thresholded

def Thresholding(Image):
    High_thresh=Image.max()*0.3 #setting the threshold for hysterisis thresholding
    low_thresh=High_thresh*0.05
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if Image[i][j]>High_thresh:
                Image[i][j]=255
            elif low_thresh< Image[i][j] <= High_thresh: #performing connected component analysis in here for the pixels in between high and low threshold
                if Image[i-1][j-1]>High_thresh or Image[i-1][j]>High_thresh or Image[i-1][j+1]>High_thresh or Image[i][j-1]>High_thresh or Image[i][j+1]>High_thresh or Image[i+1][j-1]>High_thresh or Image[i+1][j+1]>High_thresh or Image[i+1][j]>High_thresh:
                    Image[i][j]=255
            elif Image[i][j]<low_thresh:
                Image[i][j]=0
    return Image



if __name__=='__main__':
    I=Image.open("123.jpg") #reading input image
    G=gaussian_filter(3) #gaussian filter for image smoothing
    G=np.array(G)[np.newaxis]
    Gx=derivative_x() #derivative of gaussian along x direction
    Gy=derivative_y() #derivative of gaussian along y direction
    x=convolve_test(np.array(I),G) #Smoothing of imaging in x direction
    Ix=np.uint8(x) # convert to unsigned int8 to display or save the image in opencv
    y=convolve_test(np.array(I),G.T) #Smoothing of imaging in y direction
    Iy=np.uint8(y)
    cv2.imwrite("Ix.jpg",Ix)
    cv2.imwrite("Iy.jpg",Iy)
    x_dash= convolve_test(x,Gx) #convolve to get edge detection along x direction (in numpy array format)
    y_dash= convolve_test(y,Gy) #convolve to get edge detection along y direction (in numpy array format)
    Ix_dash=np.uint8(x_dash) #to save or plot the resultant image
    Iy_dash=np.uint8(y_dash)
    cv2.imwrite("Ix_dash.jpg",Ix_dash)
    cv2.imwrite("Iy_dash.jpg",Iy_dash)
    Mag,dir=magDir(x_dash,y_dash)
    nms_output,thresholded=NMS_filter(Mag,dir) #nms filtering for extra thick edges removal
    cv2.imwrite("Mxy.jpg",np.uint8(Mag))
    cv2.imwrite("Thresh.jpg",np.uint8(thresholded))
    


    

    