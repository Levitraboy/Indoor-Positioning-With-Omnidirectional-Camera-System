# Indoor Positioning With-Omnidirectional Camera System
Highly accurate indoor positioning using computer vision

# ABSTRACT
In this project, we worked on finding the location of an AGV equipped with an 
omnidirectional camera in a 10x10 meter room by using the indoor positioning method and 
displaying its position and orientation on a visual map interface. To determine the location of 
the AGV, we placed markers with precise coordinates on the walls to use as reference points 
in the room. We detected the markers by processing the panoramic images of room obtained 
from the omnidirectional on the AGV. Then we used the Angle of Arrival (AOA) model, which 
is a positioning system, whose method is based on measuring the angular directions between 
one or more points with known coordinates and an unknown point to find the vehicle's position. 
After we produced the algorithm of the model according to our own working conditions, we 
gave the already known marker coordinates, the azimuth angle measurements between the 
markers and the AGV and the rotation angle of the AGV as input to the software and got the 
position of the AGV as output. As a final step, we showed the position and the orientation angle 
of AGV on the map produced using the Unity physic engine.

![UI](https://user-images.githubusercontent.com/33463788/182940079-93210d7d-b210-4eb4-aa58-47cd81d074f2.gif)
