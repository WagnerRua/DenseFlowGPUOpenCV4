## Usage
This repo is for using GPU and opencv4 to extract videos' dense optical flow fields

## Authorship
I updated the code from [Yang Wang's repo](https://github.com/yangwangx/denseFlow_gpu) so that it could be used with the latest version of opencv.  

## Interface
Option | Name | Default | Note 
:---   | :--- | :---    | :---
f  | vidFile    | ex.avi  | filename of video
x  | xFlowFile  | x       | filename prefix of flow x component
y  | yFlowFile  | y       | filename prefix of flow x component
i  | imgFile    | i       | filename prefix of image
b  | bound      | 20      | specify the maximum (px) of optical flow
t  | type       | 1       | specify the optical flow algorithm
d  | device_id  | 0       | specify gpu id
s  | step       | 1       | specify the step for frame sampling
h  | height     | 0       | specify the height of saved flows, 0: keep original height
w  | width      | 0       | specify the width of saved flows,  0: keep original width

## Example
```
mkdir flow
./denseFlowGpu -f=ex.avi -x=flow/x -y=flow/y -i=flow/i -b=20 -t=1 -d=0 -s=1 -h=0 -w=0
```
> ex.avi is the input video  
> flow/ is the folder containing the extracted RGB frames and optical flows

## Compile
If you have the opencv in /usr/local/include/opencv4:
```
mkdir build
cd build
cmake .. 
make
```
The executable will be inside the build folder.

