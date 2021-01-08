Using the framework pytorch to develop a neural network for early detection of lung cancer.

This project is a reproduction of the project described in the second part on the book "Deep Learning With
Pytorch", available for free in the page www.pytorch.org . This is my first large scale project, made with the
purpose of honing my skills in Deep learning and coding in general.

There will be two Conv. Neural Networks in this project. The first will be used to segment ie read an entire CT
scan and spot all possible nodes in it. The second will read scan candidates and see if wicht of them are worth
to further investigate. Thus, this project will cover the development of such networks; the application that
will integrate both of them and the scrips necessary for reading raw training data and, of course, training.

THE DATA----------------------------------------

The raw data used in this project comes from the LUNA dataset ( download at https://luna16.grand-challenge
.org/Description ). The set contains dozens of real lif CT scans and annotations about them.

A CT scan is basicaly a 3D greyscale image of a human torso and its internal organs. Or, to put in a more
pratical terms, a set of cross sections of someones body with each stacke in top of the other. Thus, it should
be represented on a computer like an tridimentional array. Also, each one has its own identification, with the
purpose of ditiguish from one to another. With that logic in mind each CT scan in this file is stored in two 
files, the first containing the raw image and the other containing its metadata( like the dimentions of the 
array and how to map a x,y,z location in milimiters to the array )

The annotations are a set of weird spots in the scans that might be nodules or early tummors. It is dived in
two files. The first contais a list of suspects, marked by their ct scans identification and the milimetric 
positions inside of it. The seconds have the confirmed suspects, each ove have the same information of the other set plus the radius of the nodule/tumor. 

THE FILES -------------------------------------

The files of this project will be divided in three parts. First, the code necessary to transform the raw data
into proto-data that the networks con work with. Second, each network will have its own folder, with scripts to transform the proto-data into actual input, the code of the network itself and scripts to train each network. Finally, the application that will implement and manege both networks. Also, there will be an extra folder with
benchmarks and tests for the system.