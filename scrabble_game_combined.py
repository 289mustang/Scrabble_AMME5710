# ================================================================================
# Combined Scrabble Project Script
# Includes: Point Scoring Code + Scrabble Detector
# ================================================================================

# %% ## Point Scoring Module
# ============================================================================
# Auto-converted from: Point Scoring Code.ipynb
# Converted on: 2025-10-27 23:36:32
# Cell markers use Spyder/VS Code convention: '# %% ## <section title>'
# ============================================================================

# %% ## import numpy as np
import numpy as np

# Score for each letter (b for blank tile)
LETTER_POINTS = {
"A":1, "B":3, "C":3, "D":2, "E":1, "F":4, "G":2, "H":4, "I":1,
"J":8, "K":5, "L":1, "M":3, "N":1, "O":1, "P":3, "Q":10, "R":1,
"S":1, "T":1, "U":1, "V":4, "W":4, "X":8, "Y":4, "Z":10, "b":0
}

# Load real Scrabble dictionary (ENABLE)
with open("enable1.txt", "r") as f:
    VALID_WORDS = set(w.strip().upper() for w in f)

# Functions
def reset_game():
    """Resets the letter and word multiplier"""

    # Ensure they are global variables
    global letter_multiplier, word_multiplier

    # The letter bonuses
    letter_multiplier = np.array([
        [1,1,1,2,1,1,1,1,1,1,1,2,1,1,1],
        [1,1,1,1,1,3,1,1,1,3,1,1,1,1,1],
        [1,1,1,1,1,1,2,1,2,1,1,1,1,1,1],
        [2,1,1,1,1,1,1,2,1,1,1,1,1,1,2],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,3,1,1,1,3,1,1,1,3,1,1,1,3,1],
        [1,1,2,1,1,1,2,1,2,1,1,1,2,1,1],
        [1,1,1,2,1,1,1,1,1,1,1,2,1,1,1],
        [1,1,2,1,1,1,2,1,2,1,1,1,2,1,1],
        [1,3,1,1,1,3,1,1,1,3,1,1,1,3,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,1,1,1,1,1,1,2,1,1,1,1,1,1,2],
        [1,1,1,1,1,1,2,1,2,1,1,1,1,1,1],
        [1,1,1,1,1,3,1,1,1,3,1,1,1,1,1],
        [1,1,1,2,1,1,1,1,1,1,1,2,1,1,1]
    ])

    # The word bonuses
    word_multiplier = np.array([
        [3,1,1,1,1,1,1,3,1,1,1,1,1,1,3],
        [1,2,1,1,1,1,1,1,1,1,1,1,1,2,1],
        [1,1,2,1,1,1,1,1,1,1,1,1,2,1,1],
        [1,1,1,2,1,1,1,1,1,1,1,2,1,1,1],
        [1,1,1,1,2,1,1,1,1,1,2,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [3,1,1,2,1,1,1,2,1,1,1,2,1,1,3],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,2,1,1,1,1,1,2,1,1,1,1],
        [1,1,1,2,1,1,1,1,1,1,1,2,1,1,1],
        [1,1,2,1,1,1,1,1,1,1,1,1,2,1,1],
        [1,2,1,1,1,1,1,1,1,1,1,1,1,2,1],
        [3,1,1,1,1,1,1,3,1,1,1,1,1,1,3]
    ])

def find_main_word(diff_indices, board):
    """Finds the main word that has been added"""

    # Normalize to list of tuples
    diff = [(int(r), int(c)) for r, c in diff_indices]
    rows = [r for r, c in diff]
    cols = [c for r, c in diff]

    # horizontal if all rows same
    if len(set(rows)) == 1:
        r = rows[0]
        start_c = min(cols)
        end_c = max(cols)
        # expand left
        while start_c > 0 and board[r, start_c - 1] != "":
            start_c -= 1
        # expand right
        while end_c < 14 and board[r, end_c + 1] != "":
            end_c += 1
        positions = [(r, c) for c in range(start_c, end_c + 1)]
        word = "".join(board[r, c] for (r, c) in positions)
        return word, positions

    # vertical if all cols same
    if len(set(cols)) == 1:
        c = cols[0]
        start_r = min(rows)
        end_r = max(rows)
        # expand up
        while start_r > 0 and board[start_r - 1, c] != "":
            start_r -= 1
        # expand down
        while end_r < 14 and board[end_r + 1, c] != "":
            end_r += 1
        positions = [(r, c) for r in range(start_r, end_r + 1)]
        word = "".join(board[r, c] for (r, c) in positions)
        return word, positions

    # not aligned: try to detect orientation from adjacency of the first new tile
    r0, c0 = diff[0]
    # horizontal adjacency?
    if (c0 > 0 and board[r0, c0 - 1] != "") or (c0 < 14 and board[r0, c0 + 1] != ""):
        start_c = c0
        while start_c > 0 and board[r0, start_c - 1] != "":
            start_c -= 1
        end_c = c0
        while end_c < 14 and board[r0, end_c + 1] != "":
            end_c += 1
        positions = [(r0, c) for c in range(start_c, end_c + 1)]
        word = "".join(board[r0, c] for (r0, c) in positions)
        return word, positions

    # vertical adjacency?
    if (r0 > 0 and board[r0 - 1, c0] != "") or (r0 < 14 and board[r0 + 1, c0] != ""):
        start_r = r0
        while start_r > 0 and board[start_r - 1, c0] != "":
            start_r -= 1
        end_r = r0
        while end_r < 14 and board[end_r + 1, c0] != "":
            end_r += 1
        positions = [(r, c0) for r in range(start_r, end_r + 1)]
        word = "".join(board[r, c0] for (r, c0) in positions)
        return word, positions

    # isolated single tile (no extension) -> return single letter word (will be invalid in VALID_WORDS)
    return board[r0, c0], [(r0, c0)]

def find_all_words(diff_indices, board):
    """Return a list of (word, positions) for all words formed this turn."""
    words = []
    main_word, main_pos = find_main_word(diff_indices, board)
    if main_word:
        words.append((main_word, main_pos))

    diff_set = set(diff_indices)
    for r, c in diff_indices:
        # Check perpendicular direction
        if all(pos[0] == r for pos in main_pos):
            # main word is horizontal → check vertical word
            start_r = r
            while start_r > 0 and board[start_r-1, c] != "":
                start_r -= 1
            end_r = r
            while end_r < 14 and board[end_r+1, c] != "":
                end_r += 1
            if end_r != start_r:  # there is a vertical word
                positions = [(rr, c) for rr in range(start_r, end_r+1)]
                word = "".join(board[rr, c] for rr in range(start_r, end_r+1))
                if (word, positions) not in words:
                    words.append((word, positions))
        else:
            # main word is vertical → check horizontal word
            start_c = c
            while start_c > 0 and board[r, start_c-1] != "":
                start_c -= 1
            end_c = c
            while end_c < 14 and board[r, end_c+1] != "":
                end_c += 1
            if end_c != start_c:  # there is a horizontal word
                positions = [(r, cc) for cc in range(start_c, end_c+1)]
                word = "".join(board[r, cc] for cc in range(start_c, end_c+1))
                if (word, positions) not in words:
                    words.append((word, positions))
    return words

def is_word_real(all_words):
    """Checks all words to ensure they are real"""

    for word, positions in all_words:
        # Skip validation if the word contains a blank tile (trust player)
        if "b" in word:
            continue

        # Check validity against the word list
        if word not in VALID_WORDS:
            print(f"'{word}' is not a valid word!")

    return

def calculate_score(all_words, diff_indices, current_board):
    """Calculates the score from the words in this turn"""

    # Extracting the words
    diff_set = set((int(r), int(c)) for r, c in diff_indices)

    # Finding the score of each word
    score = 0
    for word, positions in all_words:
        word_mult = []
        word_score = 0
        for r, c in positions:
            letter = current_board[r, c]
            base_points = LETTER_POINTS.get(letter, 0)
            if (r, c) in diff_set:
                base_points *= letter_multiplier[r, c]
                word_mult.append(word_multiplier[r, c])
            word_score += base_points
        score += word_score*max(word_mult)

    # Reset multipliers for the squares used in this turn
    for r, c in diff_set:
        letter_multiplier[r, c] = 1
        word_multiplier[r, c] = 1

    # Bingo bonus (for all seven tiles used in one turn)
    if len(diff_set) == 7:
        score += 50

    return score

def update_score(p1_score, p2_score, previous_board, current_board, turn):
    """Returns updated points for players"""
    
    # Find newly placed tiles
    new_word = previous_board != current_board
    diff_indices = list(zip(*np.where(new_word))) 
    all_words = find_all_words(diff_indices, current_board)

    # Validate that the words are real
    is_word_real(all_words)

    # Calculate score for all words
    score = calculate_score(all_words, diff_indices, current_board)

    # Find which player's score to add to
    if turn % 2 == 1:
        p1_score += score
    else:
        p2_score += score

    return p1_score, p2_score

def print_board(board):
    """Prints the Scrabble board"""

    # Print board
    print("    " + " ".join([f"{i+1:2}" for i in range(15)]))
    print("   " + "---" *15)
    for i, row in enumerate(board):
        line = "  ".join(letter if letter != "" else "." for letter in row)
        print(f"{i+1:2} | {line}")

# %% ## Example Game
# Example Game
reset_game()
p1_score = 0; p2_score = 0
board0 = np.array([[""]*15 for _ in range(15)])
board1 = board0.copy()
board1[7,7] = "A"; board1[7,8] = "R"; board1[7,9] = "T"

# Turn 1
turn = 1
p1_score, p2_score = update_score(p1_score, p2_score, board0, board1, turn)
print_board(board1)
print(f"P1: {p1_score}pts, P2: {p2_score}pts.\n")

board2 = board1.copy()
board2[7,10] = "S"
board2[7,6] = "P" 

# Turn 2
turn = 2
p1_score, p2_score = update_score(p1_score, p2_score, board1, board2, turn)
print_board(board2)
print(f"P1: {p1_score}pts, P2: {p2_score}pts.\n")

board3 = board2.copy()
board3[4,10] = "K"
board3[5,10] = "I"
board3[6,10] = "T"

# Turn 3
turn = 3
p1_score, p2_score = update_score(p1_score, p2_score, board2, board3, turn)
print_board(board3)
print(f"P1: {p1_score}pts, P2: {p2_score}pts.\n")

board4 = board3.copy()
board4[6,9] = "I"
board4[8,9] = "b"

# Turn 4
turn = 4
p1_score, p2_score = update_score(p1_score, p2_score, board3, board4, turn)
print_board(board4)
print(f"P1: {p1_score}pts, P2: {p2_score}pts.\n")



# %% ## Imported Modules
# Imported Modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# %% ## Extracts video frames
# Extracts video frames
def get_frames(video, frame_interval):
    '''Extracts one frame every `frame_interval` frames from the video.'''
    cap = cv2.VideoCapture(video)
    all_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        all_frames.append(frame)
        frame_num += frame_interval
    
    return all_frames

# Get a frame every 10 seconds
frame_interval = 500
all_frames1 = get_frames("Scrabble Game 1.mp4", frame_interval)
# all_frames2 = get_frames("Scrabble Game 2.mp4", frame_interval)
# all_frames3 = get_frames("Scrabble Game 3.mp4", frame_interval)

# %% ## Isolates board
# Isolates board
def filter_background(im):
    '''Filters the background leaving just the board'''
    im_filtered = im.copy()
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([80, 255, 255])
    mask = cv2.inRange(im, lower_blue, upper_blue)
    im_filtered[(mask == 0),:] = (0,0,0)

    return im_filtered

def find_largest_contour(im_filtered):
    '''Find the largest contour in the image which will be the board'''

    # Change image to grayscale and threshold for best results
    im_gray = cv2.cvtColor(im_filtered, cv2.COLOR_RGB2GRAY)
    thresh_val_used, thresh_im = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Biggest contour will be board
    largest_contour = max(contours, key = cv2.contourArea)

    return largest_contour

def find_corners(largest_contour):
    '''Finds corners of board'''

    # Find corners of board
    epsilon = 0.04*cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    corners = corners.reshape(-1, 2)
    corners = np.unique(corners, axis = 0)  # Sometimes poly function finds more than 4 corners
    
    # Rearrange to match reference data
    rearranged_corners = np.zeros((4,2), dtype = int)
    s = corners.sum(axis = 1)
    rearranged_corners[0] = corners[np.argmin(s)]     # Top left
    rearranged_corners[3] = corners[np.argmax(s)]     # Bottom right
    diff = np.diff(corners, axis = 1)
    rearranged_corners[1] = corners[np.argmin(diff)]  # Top right
    rearranged_corners[2] = corners[np.argmax(diff)]  # Bottom left

    return rearranged_corners

def generate_warped_board(im, corners):
    '''Uses the corners to generate a warped 800x800 image of the board'''
    
    # Changing corners order to work best with functions
    corners = corners[[0, 1, 3, 2], :]
    corners = corners.astype(np.float32)
    transfer_size = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype = np.float32)

    # Generating warped perspective
    M = cv2.getPerspectiveTransform(corners, transfer_size)
    warped = cv2.warpPerspective(im, M, (800, 800))

    return warped

def get_scrabble_board(all_frames):
    '''Isolates the scrabble board from the frames'''
    all_im = []
    all_im_filtered = []
    all_warped = []
    for i in range(len(all_frames)):
        im = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB)
        im_filtered = filter_background(im)
        largest_contour = find_largest_contour(im_filtered)
        rearranged_corners = find_corners(largest_contour)
        warped = generate_warped_board(im, rearranged_corners)

        all_im.append(im)
        all_im_filtered.append(im_filtered)
        all_warped.append(warped)

    return all_im, all_im_filtered, all_warped  

# Get scrabble board data
all_im1, all_im_filtered1, all_warped1 = get_scrabble_board(all_frames1)
# all_im2, all_im_filtered2, all_warped2 = get_scrabble_board(all_frames2)
# all_im3, all_im_filtered3, all_warped3 = get_scrabble_board(all_frames3)

plt.imshow(all_warped1[6])



# %% Board segmentation
# =========================
# OPTION A: Inner play area warp from white grid lines
# =========================
import cv2, numpy as np
# %% Board segmentation (fixed)
# ==============================
# Board segmentation + grid find
# (with corner printouts)
# ==============================
import numpy as np
import cv2

# ---------- helpers: masks & lines ----------

def _central_mask(shape, frac=0.05):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    y0, y1 = int(H*frac), int(H*(1-frac))
    x0, x1 = int(W*frac), int(W*(1-frac))
    m[y0:y1, x0:x1] = 255
    return m

def grid_white_mask(rgb):
    rim = _central_mask(rgb.shape, frac=0.05)  # <-- smaller crop; keep more border

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(lab)
    mask_lab = (L > 150) & (np.abs(a - 128) < 14) & (np.abs(b - 128) < 16)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    Hc, S, V = cv2.split(hsv)
    mask_hsv = (S < 20) & (V > 180)             # <-- looser thresholds

    mask = ((mask_lab | mask_hsv).astype(np.uint8) * 255) & rim

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))
    mask |= ((tophat > 20).astype(np.uint8) * 255) & rim

    # connect lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    mask_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1,10)), 1)
    mask_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (10,1)), 1)
    mask = cv2.bitwise_or(mask_v, mask_h)

    cv2.imwrite("dbg_mask_tuned.png", mask)  # save to inspect
    plt.imshow(mask)
    return mask


def hough_grid_segments(mask):
    """Return long near-vertical and near-horizontal segments from mask."""
    H, W = mask.shape[:2]
    edges = cv2.Canny(mask, 20, 100, L2gradient=True)
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                           minLineLength=int(0.5*min(H,W)), maxLineGap=25)
    if segs is None:
        print("[hough] No segments found.")
        return [], []
    vertical, horizontal = [], []
    for x1,y1,x2,y2 in segs[:,0,:]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang > 82:      vertical.append((x1,y1,x2,y2))
        elif ang < 8:     horizontal.append((x1,y1,x2,y2))
    print(f"[hough] {len(vertical)} vertical, {len(horizontal)} horizontal segments.")
    return vertical, horizontal

def _fit_line_from_segments(segments):
    """Fit infinite line (vx,vy,x0,y0) to segment endpoints."""
    if not segments:
        return None
    pts = []
    for x1,y1,x2,y2 in segments:
        pts.append([x1,y1]); pts.append([x2,y2])
    pts = np.array(pts, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)

def _intersect_lines(l1, l2):
    """Intersection of two (vx,vy,x0,y0) lines."""
    vx1, vy1, x1, y1 = l1; vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2],[vy1, -vy2]], np.float64)
    b = np.array([x2-x1, y2-y1], np.float64)
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([x1 + vx1*t[0], y1 + vy1*t[0]], np.float32)

def find_inner_play_corners_from_mask(mask):
    """Fit L/R and T/B grid borders and return TL,TR,BR,BL (float32)."""
    vertical, horizontal = hough_grid_segments(mask)
    if len(vertical) < 6 or len(horizontal) < 6:
        print("[corner detect] Not enough segments —",
              len(vertical), "vert,", len(horizontal), "horiz")
        return None

    v_x = np.array([0.5*(x1+x2) for x1,y1,x2,y2 in vertical], np.float32)
    h_y = np.array([0.5*(y1+y2) for x1,y1,x2,y2 in horizontal], np.float32)

    Kv = max(6, len(vertical)//5)
    Kh = max(6, len(horizontal)//5)
    left_idx  = np.argsort(v_x)[:Kv]
    right_idx = np.argsort(v_x)[-Kv:]
    top_idx   = np.argsort(h_y)[:Kh]
    bot_idx   = np.argsort(h_y)[-Kh:]

    L = _fit_line_from_segments([vertical[i] for i in left_idx])
    R = _fit_line_from_segments([vertical[i] for i in right_idx])
    T = _fit_line_from_segments([horizontal[i] for i in top_idx])
    B = _fit_line_from_segments([horizontal[i] for i in bot_idx])
    if any(x is None for x in (L,R,T,B)):
        print("[corner detect] One or more line fits failed.")
        return None

    def ang(vx,vy): return np.degrees(np.arctan2(vy, vx))
    if abs(ang(*L[:2]) - ang(*R[:2])) > 5 or abs(ang(*T[:2]) - ang(*B[:2])) > 5:
        print("[corner detect] Line pairs not parallel enough.")
        return None

    TL = _intersect_lines(T, L); TR = _intersect_lines(T, R)
    BR = _intersect_lines(B, R); BL = _intersect_lines(B, L)

    # small refinement: move to strongest mask pixel nearby
    def refine(p, rad=4):
        x, y = int(p[0]), int(p[1])
        x0, x1 = max(0, x-rad), min(mask.shape[1]-1, x+rad)
        y0, y1 = max(0, y-rad), min(mask.shape[0]-1, y+rad)
        patch = mask[y0:y1+1, x0:x1+1].astype(np.float32)
        if patch.size == 0: return p
        j = np.argmax(patch); j = np.unravel_index(int(j), patch.shape)
        return np.array([x0 + j[1], y0 + j[0]], np.float32)

    corners = np.array([refine(TL), refine(TR), refine(BR), refine(BL)], np.float32)

    print("[corner detect] Raw corners (TL, TR, BR, BL):")
    for name, pt in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
    return corners

def _order_quad_tl_tr_br_bl(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)

def sanitize_corners(corners, img_shape, min_area_frac=0.10):
    """Clip/order/validate 4 points; print sanitized coords."""
    if corners is None: return None
    c = np.array(corners, np.float32).reshape(-1,2)
    if c.shape[0] != 4 or not np.all(np.isfinite(c)): return None
    H, W = img_shape[:2]
    c[:,0] = np.clip(c[:,0], 0, W-1); c[:,1] = np.clip(c[:,1], 0, H-1)
    c = _order_quad_tl_tr_br_bl(c)

    # area check
    x, y = c[:,0], c[:,1]
    area = 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
    if area < (min_area_frac * min(H, W))**2: 
        print("[sanitize] Area too small; rejecting.")
        return None

    print("[extract_play_area] Sanitized corners (TL, TR, BR, BL):")
    for name, pt in zip(["TL", "TR", "BR", "BL"], c):
        print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
    return c

def warp_play_area_from_corners(rgb, corners, out_size=800):
    dst = np.array([[0,0],[out_size,0],[out_size,out_size],[0,out_size]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    flat = cv2.warpPerspective(rgb, M, (out_size, out_size))
    return flat, M

def extract_play_area(rgb, out_size=800):
    """One-call: mask -> corners (print) -> warp; fallback to central crop."""
    mask = grid_white_mask(rgb)
    raw = find_inner_play_corners_from_mask(mask)
    corners = sanitize_corners(raw, rgb.shape)
    if corners is None:
        print("[extract_play_area] Using fallback (central crop).")
        H, W = rgb.shape[:2]; m = int(0.07*min(H,W))
        inner = cv2.resize(rgb[m:H-m, m:W-m], (out_size, out_size),
                           interpolation=cv2.INTER_CUBIC)
        return inner, {"mask": mask, "corners": None, "fallback": True}
    flat, M = warp_play_area_from_corners(rgb, corners, out_size)
    return flat, {"mask": mask, "corners": corners, "fallback": False}

def draw_corners_overlay(img_rgb, corners, out_path):
    vis = img_rgb.copy()
    if corners is not None:
        for (x,y) in corners:
            cv2.circle(vis, (int(round(x)), int(round(y))), 8, (255,0,0), -1, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

# ---------- grid detection on flattened board ----------

def _proj_peaks_1d(img, axis, approx_period_px, snap_radius=6):
    """Find 16 equal-spaced grid lines along one axis by 1D projection + snapping."""
    H, W = img.shape[:2]
    if axis == 1:  # vertical lines: score per column
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9,1)))
        score = th.sum(axis=0).astype(np.float64); length = W
    else:          # horizontal lines: score per row
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1,9)))
        score = th.sum(axis=1).astype(np.float64); length = H

    score = cv2.GaussianBlur(score.reshape(1,-1), (1,9), 0).ravel()
    score -= score.min(); 
    if score.max() > 0: score /= score.max()

    K = 16
    a0 = approx_period_px
    b0 = (length - a0*(K-1)) / 2.0

    def snap(x):
        l = int(np.clip(np.floor(x - snap_radius), 0, length-1))
        r = int(np.clip(np.ceil (x + snap_radius), 0, length-1))
        j = l + np.argmax(score[l:r+1])
        if 1 <= j < length-1:
            y0,y1,y2 = score[j-1], score[j], score[j+1]
            d = (y0 - 2*y1 + y2)
            if abs(d) > 1e-6: j = j + 0.5*(y0 - y2)/d  # subpixel
        return float(np.clip(j, 0, length-1))

    xk = np.array([snap(b0 + a0*k) for k in range(K)], dtype=np.float64)

    # refine spacing with least squares, then enforce perfect equal spacing
    k = np.arange(K, dtype=np.float64)
    A = np.stack([k, np.ones_like(k)], axis=1)
    a,b = np.linalg.lstsq(A, xk, rcond=None)[0]
    xk = np.array([snap(b + a*i) for i in range(K)], dtype=np.float64)
    a,b = np.linalg.lstsq(A, xk, rcond=None)[0]
    return b + a*np.arange(K, dtype=np.float64)

def detect_scrabble_grid(play_flat_rgb):
    """
    Return (xs, ys, vis) for 16 vertical & 16 horizontal line positions on play_flat_rgb.
    """
    gray = cv2.cvtColor(play_flat_rgb, cv2.COLOR_RGB2GRAY) if play_flat_rgb.ndim==3 else play_flat_rgb.copy()
    gray_eq = cv2.equalizeHist(gray)
    period = play_flat_rgb.shape[1] / 15.0

    xs = _proj_peaks_1d(gray_eq, axis=1, approx_period_px=period, snap_radius=6)
    ys = _proj_peaks_1d(gray_eq, axis=0, approx_period_px=period, snap_radius=6)

    vis = play_flat_rgb.copy()
    for x in xs:
        cv2.line(vis, (int(round(x)), 0), (int(round(x)), vis.shape[0]-1), (255,255,255), 1, cv2.LINE_AA)
    for y in ys:
        cv2.line(vis, (0, int(round(y))), (vis.shape[1]-1, int(round(y))), (255,255,255), 1, cv2.LINE_AA)
    return xs, ys, vis

def iter_cells_from_lines(im_rgb, xs, ys, inset_frac=0.12):
    """Yield ((r,c), RGB crop) for each 15×15 cell using detected line coords."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    for r in range(15):
        for c in range(15):
            x0, x1 = xs[c], xs[c+1]
            y0, y1 = ys[r], ys[r+1]
            w = x1 - x0; h = y1 - y0
            dx = inset_frac * w; dy = inset_frac * h
            xi0 = int(max(0, np.floor(x0 + dx)))
            xi1 = int(min(im_rgb.shape[1], np.ceil (x1 - dx)))
            yi0 = int(max(0, np.floor(y0 + dy)))
            yi1 = int(min(im_rgb.shape[0], np.ceil (y1 - dy)))
            yield (r, c), im_rgb[yi0:yi1, xi0:xi1]

# ---------- usage example ----------
# Assumes you already have a warped RGB board (e.g., all_warped1[6]):

try:
    warped = all_warped1[6]  # RGB warped board from your earlier pipeline
except Exception as e:
    raise RuntimeError("Ensure 'warped' RGB image is available (e.g., all_warped1[6]).") from e

# 1) Extract & print corners
play_flat, dbg = extract_play_area(warped, out_size=800)
draw_corners_overlay(warped, dbg["corners"], "dbg_corners_on_warped.png")
cv2.imwrite("dbg_mask.png", grid_white_mask(warped))

# 2) Detect grid lines & save overlay
xs, ys, grid_vis = detect_scrabble_grid(play_flat)
cv2.imwrite("dbg_grid_on_playflat.png", cv2.cvtColor(grid_vis, cv2.COLOR_RGB2BGR))
