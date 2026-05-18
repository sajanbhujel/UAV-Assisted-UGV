import numpy as np
import cv2
import glob
import os


CHECKERBOARD = (8, 6) 


SQUARE_SIZE_MM = 23.45  


IMG_PATH = 'images/*.jpg' 



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objpoints = [] 
imgpoints = [] 


objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

# --- 3. PROCESSING IMAGES ---
images = glob.glob(IMG_PATH)

if not images:
    print("No images found! Check your path.")
    exit()

print(f"Found {len(images)} images. Processing...")
valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        valid_images += 1
        objpoints.append(objp)
        
     
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration Check', img)
        cv2.waitKey(1000)
    else:
        print(f"Skipping {fname} - Corners not found.")

cv2.destroyAllWindows()


if valid_images > 0:
    print(f"\nCalibrating with {valid_images} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration Complete! RMSE Error: {ret:.4f}")

    
    print("\nCamera Matrix (K):")
    print(mtx)
    print("\nDistortion Coefficients (D):")
    print(dist)

    np.save("camera_matrix.npy", mtx)
    np.save("dist_coeffs.npy", dist)
    print("\nSaved parameters to .npy files.")
else:
    print("Calibration Failed. No valid checkerboards detected.")