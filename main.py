# All import
import cv2
import pandas as pd
from tracker import *
from tracker_speed import *

# Initialize Tracker เรียกใช้ class
tracker = EuclideanDistTracker()
tracker1 = EuclideanDistTracker1()


# Initialize the videocapture object
cap = cv2.VideoCapture("./VDOdata/ThaiBuri_CMA13_7.30.mp4")
# cap = cv2.VideoCapture("./wuVDO/cam16")
input_size = 350 #กำหนดขนาดทั้ง W H 320
if not cap.isOpened():
    print("Failed to open video file")

# Get the original frame dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the new dimensions
new_width = 750 #790
new_height = int((new_width / width) * height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to save the resized video
cap1 = cv2.VideoWriter('CAM16-17.00-18.00 = 9_1_2023.mp4', fourcc, 30, (new_width, new_height))


# Detection confidence threshold ความถูกต้อง
confThreshold = 0.7 # ค่าความเชื่อมั่น
nmsThreshold = 0.6  #(bounding box ที่ดีที่สุด)

font_color = (255, 255, 255)
font_warn = (0, 0, 255)
font_size = 0.7
font_thickness = 2

# Middle cross line position
middle_line_position =  95 #352
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# # Define the angle of the lines in degrees
# angle_degrees = 30
#
# # Calculate the slope of the lines
# slope = math.tan(math.radians(angle_degrees))
#
# # Define the x-coordinate of the left edge of the video
# left_edge_x = 0
#
# # Define the x-coordinate of the skew point (for example, 1/3 of the width of the video)
# skew_x = 750
#
# # Define the y-coordinate of the middle line
# middle_line_y = middle_line_position + 250
#
# # Define the y-coordinates of the upper and lower lines
# upper_line_y = up_line_position + 250
# lower_line_y = down_line_position + 250
#
# # Calculate the y-coordinates of the skew point on each line
# middle_line_skew_y = int(middle_line_y + slope * (left_edge_x - skew_x))
# upper_line_skew_y = int(upper_line_y + slope * (left_edge_x - skew_x))
# lower_line_skew_y = int(lower_line_y + slope * (left_edge_x - skew_x))
#
# # Define the starting and ending points of the middle line
# middle_line_start = (left_edge_x, middle_line_y)
# middle_line_end = (skew_x, middle_line_skew_y)
#
# # Define the starting and ending points of the upper line
# upper_line_start = (left_edge_x, upper_line_y)
# upper_line_end = (skew_x, upper_line_skew_y)
#
# # Define the starting and ending points of the lower line
# lower_line_start = (left_edge_x, lower_line_y)
# lower_line_end = (skew_x, lower_line_skew_y)

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')


# class index for our required detection classes
required_class_index = [2, 3, 5, 7]
detected_classNames = []

## Model Files
modelConfiguration = 'yolov4.cfg'
modelWeigheights = 'yolov4.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)


# Define random colour for each class สุ่มสีของclsass
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# List for store vehicle count information
speed_car = []
speed_motorbike = []
speed_bus = []
speed_truck = []

# Function for count vehicle
def count_vehicle(box_id, resized_frame):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center


    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(resized_frame, center, 2, (0, 0, 255), -1)  # end here
    # print(lower_line_skew_y)

# Function for finding the detected objects from the network output
def postProcess(outputs, resized_frame):
    global detected_classNames
    height, width = resized_frame.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 11.8

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds) --------------------------------------------------------------------------------------------
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)
            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            # cv2.putText(resized_frame,f'{name.upper()} {int(confidence_scores[i]*100)}%',
            #       (x, y-10 ), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
        # Draw bounding rectangle
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), color, 2) #draw the bounding box
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    #Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, resized_frame)
    boxes_ids = tracker1.update(detection)
    for box in boxes_ids:
        x, y, w, h, id = box
        # print("position :", [x, y], "FPS :", fps, "ID: ", id)
        SpeedEstimatorTool = SpeedEstimator([x, y], fps)
        speed = SpeedEstimatorTool.estimateSpeed()
        # print(speed)
        if speed > 0:
            if name == 'car':
                speed_car.append(speed)
            elif name == 'motorbike':
                speed_motorbike.append(speed)
            elif name == 'bus':
                speed_bus.append(speed)
            else:
                speed_truck.append(speed)

            if speed > 60:
                cv2.putText(resized_frame, f'{name.upper()} {str(speed)+"Km/h"}',
                        (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, font_warn, 1)
            else:
                cv2.putText(resized_frame, f'{name.upper()} {str(speed) + "Km/h"}',
                        (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def realTime():
    vdoout = cv2.VideoWriter('Results2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 11.8, (new_width, new_height))
    while (True):
    # Capture frame-by-frame
        success, img = cap.read()

    # Resize the frame
        resized_frame = cv2.resize(img, (new_width, new_height))
        ih, iw, channels = resized_frame.shape


    # print(iw,ih,channels)
        blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False) #[0, 0, 0] ดังนั้นจึงไม่มีการลบค่าเฉลี่ย

    # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
#   # Feed data to the network
        outputs = net.forward(outputNames)

    # Find the objects from the network output
        postProcess(outputs, resized_frame)


    # Draw the crossing lines
        cv2.line(resized_frame, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(resized_frame, (0, up_line_position), (iw, up_line_position), (237, 43, 42), 2)
        cv2.line(resized_frame, (0, down_line_position), (iw, down_line_position), (255, 211, 176), 2)

        # Draw counting texts in the frame
        cv2.putText(resized_frame, "Up", (150, 290), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness) #115
        cv2.putText(resized_frame, "Down", (210, 290), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness) #160
        cv2.putText(resized_frame, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
        cv2.putText(resized_frame, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 350), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
        cv2.putText(resized_frame, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 380), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
        cv2.putText(resized_frame, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 410),cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)

        # df = pd.DataFrame(list(zip(speed_car,speed_motorbike,speed_bus,speed_truck)),columns=['speed_car','speed_motorbike','speed_bus','speed_truc'])
        # print(df)
    # Assume that up_list and down_list are your data lists
        data = np.array([['Direction', 'car', 'motorbike', 'bus', 'truck'],
                     ['Up'] + up_list,
                     ['Down'] + down_list])
    # Save the data to a CSV file using np.savetxt()
        np.savetxt('data.csv', data, delimiter=',', fmt='%s')

    # Show the frames
    # cv2.imshow('Output', resized_frame)
    #
        df = pd.read_csv('data.csv')
        df0 = pd.DataFrame(speed_car, columns=['speed_car'])
        df1 = pd.DataFrame(speed_motorbike, columns=['speed_motorbike'])
        df2 = pd.DataFrame(speed_bus, columns=['speed_bus'])
        df3 = pd.DataFrame(speed_truck, columns=['speed_truck'])


        # Concatenate the dataframes vertically
        df_combined = pd.concat([df0, df1, df2, df3], axis=0)

        # Replace null values with zeros
        df_combined = df_combined.fillna(0)

        # Save the combined dataframe to a CSV file with two decimal places for the speed data
        df_combined.to_csv('combined_speed_data.csv', index=False, float_format='%.2f')

        # Top speed
        max_speed_car = df_combined['speed_car'].max()
        max_speed_motorbike = df_combined['speed_motorbike'].max()
        max_speed_bus = df_combined['speed_bus'].max()
        max_speed_truck = df_combined['speed_truck'].max()

        # avg speed
        avg_speed_car = df0['speed_car'].mean()
        if pd.isna(avg_speed_car):
            avg_speed_car = 0
        avg_speed_motorbike = df1['speed_motorbike'].mean()
        if pd.isna(avg_speed_motorbike):
            avg_speed_motorbike = 0
        avg_speed_bus = df2['speed_bus'].mean()
        if pd.isna(avg_speed_bus):
            avg_speed_bus = 0
        avg_speed_truck = df3['speed_truck'].mean()
        if pd.isna(avg_speed_truck):
            avg_speed_truck = 0

        # print('avg_car', avg_speed_car)
        # print('avg_motorbike', avg_speed_motorbike)

        # Group by the "Direction" column and sum the values for each group
        sums = df.groupby('Direction').sum()

    # Create a dictionary of the summary data
        summary_dict = {'Type': ['car', 'motorbike', 'bus', 'truck'],
                        'Total': [sums['car'].sum(), sums['motorbike'].sum(), sums['bus'].sum(), sums['truck'].sum()],
                        'max_speed': [max_speed_car, max_speed_motorbike, max_speed_bus, max_speed_truck],
                        'avg_speed': [avg_speed_car, avg_speed_motorbike, avg_speed_bus, avg_speed_truck]}

        # Create a dataframe from the summary data and save it to a CSV file
        summary_df = pd.DataFrame(summary_dict)
        summary_df = summary_df.round({'avg_speed': 2})
        summary_df.to_csv('combined_speed_summary.csv', index=False)


    # Write the frame into the file 'output.avi'
        if success == True:
            vdoout.write(resized_frame)
        # Display the resulting frame
            cv2.imshow('Output', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realTime()


