import cv2
import pytesseract

# --- Configuration (you may need to set this path manually on Windows) ---
# Example: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Load and preprocess the tile image ---
img = cv2.imread("tile_A.jpg")              # input image of a single Scrabble tile
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# --- Perform OCR (single character mode) ---
custom_config = r'--oem 3 --psm 10 -l eng'
text = pytesseract.image_to_string(gray, config=custom_config)

# --- Clean and print the result ---
text = text.strip().upper()
print(f"Detected character: {text}")