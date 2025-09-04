# this is storing the co-ordinates of pixels as well which is not required.
import cv2
import numpy as np
import math

patch_size = 32
patches = []
points_so_far = []

def orb_match(full_image, patch):
    """
    ORB match, fallback to template matching if fail.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(patch, None)
    kp2, des2 = orb.detectAndCompute(full_image, None)

    if des1 is None or des2 is None or len(kp1)==0 or len(kp2)==0:
        return template_match(full_image, patch)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return template_match(full_image, patch)

    best_match = min(matches, key=lambda x: x.distance)
    x, y = map(int, kp2[best_match.trainIdx].pt)

    return (x - patch_size // 2, y - patch_size // 2)

def template_match(full_image, patch):
    """
    Template matching fallback for first patch.
    """
    result = cv2.matchTemplate(full_image, patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

def get_neighbors(center, step=patch_size, img_shape=(0,0)):
    x, y = center
    neighbors = []
    offsets = [(-1,-1), (-1,0), (-1,1),
               (0,-1),   (0,0)   , (0,1),
               (1,-1),  (1,0),  (1,1)]
    for dx, dy in offsets:
        nx = x + dx*step
        ny = y + dy*step
        if nx >= 0 and ny >= 0 and nx + step <= img_shape[1] and ny + step <= img_shape[0]:
            neighbors.append((nx, ny))
    return neighbors

def click_points(event, x, y, flags, param):
    """
    Click-based patch selection (trackpad friendly)
    """
    # global patches, points_so_far
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = min(img.shape[1], x1 + patch_size)
        y2 = min(img.shape[0], y1 + patch_size)
        patch = img[y1:y2, x1:x2].copy()

        # # --- Add jitter/noise/blur for testing ---
        # # Gaussian blur
        # patch = cv2.GaussianBlur(patch, (3, 3), 0)

        # # Random noise
        # noise = np.random.normal(0, 10, patch.shape).astype(np.uint8)
        # patch = cv2.add(patch, noise)

        # # Random small shift (±2 px)
        # shift_x, shift_y = np.random.randint(-2, 3), np.random.randint(-2, 3)
        # M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        # patch = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))

        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            patches.append(patch)
            # points_so_far.append((x, y))
            cx = x1 + patch_size // 2
            cy = y1 + patch_size // 2
            points_so_far.append((cx, cy))
            print(f"[INFO] Patch {len(patches)} added at {x},{y}")

def main(moon_image_path):
    # global patches, points_so_far
    # patches, points_so_far = [], []

    full_image = cv2.imread(moon_image_path)
    if full_image is None:
        print("Error: Image not found!")
        return

    # Phase 1: Capture clicks
    cv2.namedWindow("Moon Map")
    cv2.setMouseCallback("Moon Map", click_points, full_image)
    

    print("[INFO] Click on image to set trajectory points. Press 'q' when done.")

    while True:
        temp = full_image.copy()
        # Draw trajectory live
        if len(points_so_far) > 1:
            for i in range(1, len(points_so_far)):
                cv2.line(temp, points_so_far[i-1], points_so_far[i], (0,255,0), 6)
        cv2.imshow("Moon Map", temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # for x in range(len(patches)):
    #     for y in range(len(patches[x])):
    #         print(patches[x][y])

    if len(patches) < 2:
        print("[INFO] Need at least 2 patches!")
        return
    

        # ---- Rotate image before reconstruction ----
    # angle = 30  # change this to any rotation angle you want
    # h, w = full_image.shape[:2]
    # center = (w // 2, h // 2)

    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # full_image = cv2.warpAffine(full_image, M, (w, h))

    # Phase 2: Reconstruct trajectory
    trajectory_points = []
    start_loc = orb_match(full_image, patches[0])
    trajectory_points.append(start_loc)

    for i in range(1, len(patches)):
        current_loc = trajectory_points[-1]
        neighbors = get_neighbors(current_loc, patch_size, full_image.shape)
        best_loc = None
        best_score = -1

        # ORB match with neighbors first
        for (nx, ny) in neighbors:
            candidate = full_image[ny:ny+patch_size, nx:nx+patch_size]
            # candidate = full_image[ny, nx]
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(patches[i], None)
            kp2, des2 = orb.detectAndCompute(candidate, None)

            if des1 is None or des2 is None: continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) == 0: continue

            score = -np.mean([m.distance for m in matches])
            if score > best_score:
                best_score = score
                best_loc = (nx, ny)

        if best_loc is None:
            best_loc = orb_match(full_image, patches[i])

        if best_loc is None:
            print(f"[WARNING] Patch {i+1} could not be matched!")
            break

        cx = best_loc[0] + patch_size // 2
        cy = best_loc[1] + patch_size // 2
        trajectory_points.append((cx, cy))

    rmse, errors = compute_rmse(points_so_far, trajectory_points) 
    comparison_image = draw_trajectories_with_error(full_image, points_so_far, trajectory_points,rmse) #stroing as well as generating the cell of compa
    #comparison images



    # print(f"points of actual trajectory {points_so_far}")
    # print(f"points of reconstructed trajectory : {trajectory_points}")  

    # Phase 3: Visualization
    # output = full_image.copy()
    # for i, (x, y) in enumerate(trajectory_points):
    #     center = (x + patch_size // 2, y + patch_size // 2)
    #     cv2.circle(output, center, 4, (0, 0, 255), -1)
    #     if i > 0:
    #         prev = (trajectory_points[i-1][0] + patch_size // 2, trajectory_points[i-1][1] + patch_size // 2)
    #         cv2.line(output, prev, center, (0, 255, 0), 1)

    

    # cv2.imshow("Reconstructed Trajectory", output)


    # output generation 

     # --- Prepare outputs ---
    # Ground truth only
    gt_img = full_image.copy()
    for i in range(1, len(points_so_far)):
        cv2.line(gt_img, points_so_far[i-1], points_so_far[i], (0, 255, 0), 4)  # green line

    # Reconstructed only
    rec_img = full_image.copy()
    for i in range(1, len(trajectory_points)):
        cv2.line(rec_img, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)  # red line

    # Resize all to same height
    h = 500  # target height for output
    def resize(img): 
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*ratio), h))
    gt_img, comparison_img, rec_img = map(resize, [gt_img, comparison_image, rec_img])

    # Concatenate side by side
    final_output = cv2.hconcat([gt_img, comparison_img, rec_img]) 

    # Save result
    cv2.imwrite("trajectory_results.jpg", final_output)
    print("[INFO] Results saved as 'trajectory_results.jpg'")


    # for i in range(1, len(trajectory_points)):
    #     cv2.line(full_image.copy(), trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)  # red
    # print("[INFO] Reconstructed trajectory saved as 'reconstructed_trajectory_final.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # compute_accuracy(points_so_far, trajectory_points)
    


# draw trajectory with errors

def draw_trajectories_with_error(image, ground_truth, reconstructed,rmse):
    """
    Draws ground truth trajectory (blue), reconstructed trajectory (green),
    and displays average error between them on the image.
    """
    img_copy = image.copy()

    # --- Draw ground truth (blue line) ---
    for i in range(1, len(ground_truth)):
        cv2.line(img_copy, ground_truth[i-1], ground_truth[i], (0, 255, 0), 2)  # green

    # --- Draw reconstructed (green line) ---
    for i in range(1, len(reconstructed)):
        cv2.line(img_copy, reconstructed[i-1], reconstructed[i], (0, 0, 255), 2)  # red

    # --- Compute error ---
    if len(ground_truth) == len(reconstructed):
        errors = [
            np.linalg.norm(np.array(gt) - np.array(rec))
            for gt, rec in zip(ground_truth, reconstructed)
        ]
        avg_error = np.mean(errors)
        max_error = np.max(errors)
    else:
        avg_error, max_error = -1, -1  # trajectory mismatch

    

    # --- Put error text on image ---
    text = f"Avg Error: {avg_error:.2f} px RMSE Error : {rmse}"
    cv2.putText(img_copy, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Show image ---
    cv2.imshow("Trajectory Comparison", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_copy



# def compute_accuracy(points_so_far, trajectory_points):
#     """
#     Compare original clicked points and reconstructed trajectory points
#     """
#     if len(points_so_far) != len(trajectory_points):
#         print(f"[INFO] Original points = {len(points_so_far)}, Reconstructed = {len(trajectory_points)}")
#         min_len = min(len(points_so_far), len(trajectory_points))
#     else:
#         min_len = len(points_so_far)

#     distances = []
#     for i in range(min_len):
#         # Original center (clicked)
#         x1, y1 = points_so_far[i]
#         # Reconstructed center
#         x2 = trajectory_points[i][0] + patch_size // 2
#         y2 = trajectory_points[i][1] + patch_size // 2

#         dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#         distances.append(dist)

#     avg_dist = np.mean(distances)
#     max_dist = np.max(distances)
#     within_10px = np.sum(np.array(distances) <= 10) / len(distances) * 100

#     print("\n====== Accuracy Report ======")
#     print(f"Total Points Compared: {min_len}")
#     print(f"Average Distance: {avg_dist:.2f} pixels")
#     print(f"Max Distance: {max_dist:.2f} pixels")
#     print(f"Points within 10px: {within_10px:.2f}%")
#     print("==============================\n")

#     return avg_dist, max_dist, within_10px

def compute_rmse(ground_truth, predicted):
    # Convert to numpy arrays
    gt = np.array(ground_truth, dtype=np.float32)
    pred = np.array(predicted, dtype=np.float32)

    # Ensure same length
    min_len = min(len(gt), len(pred))
    gt = gt[:min_len]
    pred = pred[:min_len]

    # Euclidean distances per point
    errors = np.linalg.norm(gt - pred, axis=1)

    


    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    print(f"RMSE between trajectories: {rmse:.2f} pixels")
    return rmse, errors




if __name__ == "__main__":
    main("Moon_image_dataset/Equirectangular/Equi_3.png")

    # compute_accuracy(points_so_far, trajectory_points) 