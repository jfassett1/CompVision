import cv2

# Global list to store clicked points (optional)
clicked_points = []

# Mouse callback function
def get_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # Check for left mouse button click
        print(f"Clicked at: x={x}, y={y}")
        clicked_points.append((x, y)) # Store the coordinates

# Load an image
image_path = "testimg.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
# Create a window to display the image
    cv2.namedWindow("Image")

# Set the mouse callback function for the "Image" window
    cv2.setMouseCallback("Image", get_pixel_coordinates)

    # Display the image
    cv2.imshow("Image", image)

    # Wait for a key press (0 means wait indefinitely)
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    print(f"All clicked points: {clicked_points}")
