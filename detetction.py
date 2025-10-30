import cv2
import pytesseract

# # --- Configuration (you may need to set this path manually on Windows) ---
# # Example: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # --- Load and preprocess the tile image ---
# img = cv2.imread("tile_A.jpg")              # input image of a single Scrabble tile
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# # --- Perform OCR (single character mode) ---
# custom_config = r'--oem 3 --psm 10 -l eng'
# text = pytesseract.image_to_string(gray, config=custom_config)
#
# # --- Clean and print the result ---
# text = text.strip().upper()
# print(f"Detected character: {text}")


def get_black_white_proportion_grid(warped_board, num_squares=15, white_thresh=120, black_thresh=20):
    """
    Compute the proportion of near-white and near-black pixels in each cell of a warped board.
    Returns two 15x15 numpy arrays: white_proportion_grid, black_proportion_grid.
    """
    gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
    board_size = gray.shape[0]
    cell_size = board_size // num_squares

    white_grid = np.zeros((num_squares, num_squares), dtype=float)
    black_grid = np.zeros((num_squares, num_squares), dtype=float)

    letter_mat = np.zeros((15, 15))
    for i in range(num_squares):
        for j in range(num_squares):
            y1, y2 = i * cell_size, (i + 1) * cell_size
            x1, x2 = j * cell_size, (j + 1) * cell_size
            cell = gray[y1 - 20:y2 - 20, x1:x2]

            total_pixels = cell.size
            white_pixels = np.sum(cell >= white_thresh)
            black_pixels = np.sum(cell <= black_thresh)

            white_grid[i, j] = white_pixels / total_pixels
            black_grid[i, j] = black_pixels / total_pixels

            if white_grid[i, j] + black_grid[i, j] > 0.7:
                # Detection method
                custom_config = r'--oem 3 --psm 10 -l eng'
                letter_mat[i,j] = pytesseract.image_to_string(cell, config=custom_config)
                letter_mat[i,j] = letter_mat[i,j].strip().upper()

    return letter_mat