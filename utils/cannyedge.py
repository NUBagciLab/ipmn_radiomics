import cv2
import numpy as np

# Read the input image
image_path = "/home/lys6602/Radiomics/utils/CTpancreas.jpeg"  # Replace with the actual path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)  # Adjust the threshold values for better results

# Convert edges to Kenny Edge Candy style
# kenny_edge_candy = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# kenny_edge_candy[np.where((kenny_edge_candy == [255, 255, 255]).all(axis=2))] = [0, 255, 255]  # Replace white with yellow
edges = cv2.bitwise_not(edges)
# Save the Kenny Edge Candy image as a JPEG file
output_path = "/home/lys6602/Radiomics/utils/Canny_edge.jpg"  # Replace with the desired output path and filename
# cv2.imwrite(output_path, kenny_edge_candy)
cv2.imwrite(output_path, edges)

print("Kenny Edge Candy image saved successfully!")

