import cv2
import numpy as np
import pytesseract
import random
# Initialize webcam video capture
cap = cv2.VideoCapture(0)  # '0' is the default webcam
#pytesseract.pytesseract.tesseract_cmd = '/Users/adarshpachori/Desktop/CV_Sudoku_Solver/Sudoku/lib/python3.12/site-packages/tesseract'
# Preprocessing the frame (grayscale, blur, edge detection)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

# Finding the contour of the Sudoku board
def find_sudoku_contour(edged_frame):
    contours, _ = cv2.findContours(edged_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Adjust threshold based on video quality
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Assuming the Sudoku board is square
                if area > max_area:
                    max_area = area
                    sudoku_contour = approx
    return sudoku_contour

# Transform the perspective to get a top-down view of the Sudoku grid
def transform_perspective(frame, contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    return warped

# Recognize a digit using pytesseract
def recognize_digit(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    config = "--psm 10"  # Single character recognition
    digit = pytesseract.image_to_string(thresholded, config=config)
    
    try:
        return int(digit)
    except ValueError:
        return 0  # If recognition fails, return 0 (empty cell)

# Sudoku solver (backtracking algorithm)
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

# Overlay the solution on the original frame
def overlay_solution(frame, solution_grid, sudoku_contour):
    step_x = frame.shape[1] // 9
    step_y = frame.shape[0] // 9
    for i in range(9):
        for j in range(9):
            if solution_grid[i][j] != 0:
                x = j * step_x + step_x // 3
                y = i * step_y + step_y // 2
                cv2.putText(frame, str(solution_grid[i][j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# Function to identify individual Sudoku cells from the transformed grid
def get_cell_images(warped_grid):
    cell_images = []
    step_x = warped_grid.shape[1] // 9
    step_y = warped_grid.shape[0] // 9

    for i in range(9):
        for j in range(9):
            cell = warped_grid[i * step_y:(i + 1) * step_y, j * step_x:(j + 1) * step_x]
            cell_images.append(cell)
    return cell_images


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    edged_frame = preprocess_frame(frame)
    sudoku_contour = find_sudoku_contour(edged_frame)

    if sudoku_contour is not None:
        transformed = transform_perspective(frame, sudoku_contour)
        
        # Get cell images from the transformed grid
        cell_images = get_cell_images(transformed)

        # Recognize digits from each cell
        sudoku_grid = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                cell_image = cell_images[i * 9 + j]
                sudoku_grid[i][j] = recognize_digit(cell_image)

        # Solve the Sudoku
        solved_grid = sudoku_grid.copy()
        if solve_sudoku(solved_grid):
            frame_with_solution = overlay_solution(frame, solved_grid, sudoku_contour)
            cv2.imshow('Live Sudoku with Solution', frame_with_solution)
        else:
            cv2.imshow('Live Sudoku', frame)
    else:
        cv2.imshow('Live Sudoku', frame)
        if random.random()<0.1:
            print("No Sudoku board detected. Waiting...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()