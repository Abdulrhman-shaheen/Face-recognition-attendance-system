# Face Recognition Automatic Attendance System
This project is an automatic attendance taker that recognize the face from the 'Known Faces' folder and match it with its name specefied in the code to finally ouput a text file with the Name, ID, date and time of taking the attendance. 

To add your own data, remove the file in the known faces folder, add the targeted faces. Declare a variable for each face in the folder for example `Shaheen_face_encoding = face_recognition.face_encodings(Shaeen_image)` and create two arrays `known_face_encodings` and `known_face_names` with the index of both to match each other. 

### Optional
Add an excel file containing the Names and the IDs (or any other datat) of the targeteed set to be included in the output file.

<sub> This was part of a project in one of my courses so I hope you enjoy it </sub>
