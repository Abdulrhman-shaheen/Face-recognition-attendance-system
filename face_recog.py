import face_recognition

import cv2
import numpy as np
import datetime
import pandas as pd
import sys
import time


# Get a form of dictionary to get the IDS
excel_file = 'Excel Data\Data.xlsx'
df = pd.read_excel(excel_file)
name_to_id = dict(zip(df["Name"], df["ID"]))

# Get the date to be used in the text file
current_date = datetime.datetime.now().strftime("%Y-%m-%d")


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)





# Loading the data for students images
Shaeen_image = face_recognition.load_image_file(r'Known Faces\Shaheen.jpeg')
Shaheen_face_encoding = face_recognition.face_encodings(Shaeen_image)[0]


hendy_image= face_recognition.load_image_file(r'Known Faces\Abo Hendy.jpg   ')
hendy_face_encoding= face_recognition.face_encodings(hendy_image)[0]

Nora_image = face_recognition.load_image_file(r'Known Faces\Nora.jpg')
Nora_face_encoding= face_recognition.face_encodings(Nora_image)[0]

Nada_image = face_recognition.load_image_file(r'Known Faces\Nada.jpg')
Nada_face_encoding= face_recognition.face_encodings(Nada_image)[0]

Menna_image = face_recognition.load_image_file(r'Known Faces\Menna.jpg')
Menna_face_encoding= face_recognition.face_encodings(Menna_image)[0]

Yehia_image = face_recognition.load_image_file(r'Known Faces\Yehia.jpg')
Yehia_face_encoding= face_recognition.face_encodings(Yehia_image)[0]

David_image = face_recognition.load_image_file(r'Known Faces\David.jpg')
David_face_encoding= face_recognition.face_encodings(David_image)[0]

Trial_image = face_recognition.load_image_file(r'Known Faces\Haytham.jpg')
Trial_face_encoding = face_recognition.face_encodings(Trial_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Shaheen_face_encoding,
    hendy_face_encoding,
    Nora_face_encoding,
    Nada_face_encoding,
    Menna_face_encoding,
    Yehia_face_encoding,
    David_face_encoding,
    Trial_face_encoding
]
known_face_names = [
    "Abdelrhman Shaheen",
    "Ali AboHendy",
    "Nora Wael",
    "Nada Mubarak",
    "Menna Ramadan",
    "Mohamed Yehia",
    "David George",
    "Dr. Haytham"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
my_dict= {}


while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()


    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            filename = f"{current_date}.txt"
            existing_variables = set()

            # Making sure 4 seconds have passed with the same person on the camera to avoid inaccuracies
            if name in my_dict:
                if time.time() - my_dict[name] > 4:
                    # Read existing variables from the file if it exists
                    try:
                        with open(filename, "r") as file:
                            existing_variables = set(file.read().splitlines())
                    except FileNotFoundError:
                        pass

                    id_value = name_to_id.get(name, "Unknown ID")    

                    current_time = datetime.datetime.now().strftime("%I:%M %p")

                    pattern = f"{name} - ID: {id_value}"

                    # Check if the variable name already exists in the file
                    if not any(pattern in line for line in existing_variables) and name != "Unknown":

                        # Append the variable name on a new line
                        with open(filename, "a") as file:
                            file.write(f"{name} - ID: {id_value} - Time: {current_time}\n")
            else:
                    my_dict={}
                    my_dict[name] = time.time()

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!d
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()