# Import the necessary modules.
import cv2
import mediapipe as mp

# Visualization utilities
MARGIN = 10  # pixels
HAND_BOUNDARY_COLOR = (88, 205, 54)  # vibrant green
HAND_BOUNDARY_THICKNESS = 2


# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread(
    "data/fingerspelling_detection/alphabet_data/a/video_61_168_signer_frame_54.jpg"
)
# tracked_image = image.copy()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = hands.process(image_rgb)

# Check if hands are detected
if result.multi_hand_landmarks:
    x_coordinates = []
    y_coordinates = []
    for hand_landmarks in result.multi_hand_landmarks:
        # Draw landmarks on the frame
        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = image.shape
        # x_coordinates = []
        # y_coordinates = []
        for landmark in hand_landmarks.landmark:
            x_coordinates.append(landmark.x)
            y_coordinates.append(landmark.y)

    left_x = int(min(x_coordinates) * width) - MARGIN
    bottom_y = int(min(y_coordinates) * height) - MARGIN
    right_x = int(max(x_coordinates) * width) + MARGIN
    top_y = int(max(y_coordinates) * height) + MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.rectangle(
        image,
        (left_x, bottom_y),
        (right_x, top_y),
        HAND_BOUNDARY_COLOR,
        HAND_BOUNDARY_THICKNESS,
    )

# Display the frame with hand landmarks
cv2.imshow("Hand Recognition", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
