import cv2
import numpy as np
import math
import os
from datetime import datetime
# ---------- PARAMETERS ----------
patch_size = 256
patches = []
points_so_far = []

# ---------- ORB + TEMPLATE MATCH ----------
def orb_match(full_image, patch):
    """
    ORB match, fallback to template matching if fail.
    Returns top-left corner of best match.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(patch, None)
    kp2, des2 = orb.detectAndCompute(full_image, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("falling back to template matching !!")
        return template_match(full_image, patch)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        print("length of matches array is zero")
        return template_match(full_image, patch)

    best_match = min(matches, key=lambda x: x.distance)
    x, y = map(int, kp2[best_match.trainIdx].pt)

    return (x - patch_size // 2, y - patch_size // 2)

def template_match(full_image, patch):
    """Template matching fallback."""
    result = cv2.matchTemplate(full_image, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

# ---------- NEIGHBOR SEARCH ----------
orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def get_sliding_neighbors(center, patch_size, img_shape, radius=250, step=32):
    """
    Sliding window around current center.
    Returns list of (tlx, tly) top-left candidate patch coords.
    """
    cx, cy = center
    h, w = img_shape[:2]
    candidates = []

    for dy in range(-radius, radius+1, step):
        for dx in range(-radius, radius+1, step):
            ccx = int(cx + dx)
            ccy = int(cy + dy)
            tlx = ccx - patch_size // 2
            tly = ccy - patch_size // 2
            if tlx >= 0 and tly >= 0 and tlx + patch_size <= w and tly + patch_size <= h:
                candidates.append((tlx, tly))
    return candidates

# ---------- CLICK PATCHES ----------
def click_points(event, x, y, flags, param):
    """Click to select patches and store trajectory ground truth points."""
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = min(img.shape[1], x1 + patch_size)
        y2 = min(img.shape[0], y1 + patch_size)
        patch = img[y1:y2, x1:x2].copy()

        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            patches.append(patch)
            cx = x1 + patch_size // 2
            cy = y1 + patch_size // 2
            points_so_far.append((cx, cy))
            print(f"[INFO] Patch {len(patches)} added at {cx},{cy}")

# ---------- RMSE ----------
def compute_rmse(ground_truth, predicted):
    gt = np.array(ground_truth, dtype=np.float32)
    pred = np.array(predicted, dtype=np.float32)

    min_len = min(len(gt), len(pred))
    gt = gt[:min_len]
    pred = pred[:min_len]

    errors = np.linalg.norm(gt - pred, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    print(f"RMSE between trajectories: {rmse:.2f} pixels")
    return rmse, errors

# ---------- VISUALIZATION ----------
def draw_trajectories_with_error(image, ground_truth, reconstructed, rmse):
    img_copy = image.copy()

    # Draw ground truth (green)
    for i in range(1, len(ground_truth)):
        cv2.line(img_copy, ground_truth[i-1], ground_truth[i], (0, 255, 0), 2)

    # Draw reconstructed (red)
    for i in range(1, len(reconstructed)):
        cv2.line(img_copy, reconstructed[i-1], reconstructed[i], (0, 0, 255), 2)

    # Error text
    cv2.putText(img_copy, f"RMSE: {rmse:.2f} px", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Trajectory Comparison", img_copy)
    cv2.imwrite("trajectory_results.jpg", img_copy)
    print("[INFO] Results saved as 'trajectory_results.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# --- create output folder if not exists ---
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_results(image, actual_points, predicted_points, rmse_value):
    # Create unique filename based on time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(OUTPUT_DIR, f"trajectory_{timestamp}.png")
    log_path = os.path.join(OUTPUT_DIR, f"log_{timestamp}.txt")

    # Draw actual (green) and predicted (red) trajectories
    for i, (x, y) in enumerate(actual_points):
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)  # green actual
        if i > 0:
            cv2.line(image, actual_points[i-1], (x, y), (0, 255, 0), 2)

    for i, (x, y) in enumerate(predicted_points):
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)  # red predicted
        if i > 0:
            cv2.line(image, predicted_points[i-1], (x, y), (0, 0, 255), 2)

    # Put RMSE value on the image
    cv2.putText(image, f"RMSE: {rmse_value:.2f}px", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Save visualization image
    cv2.imwrite(img_path, image)
    print(f"[SAVED] Trajectory visualization saved to {img_path}")

    # Save log file
    with open(log_path, "w") as f:
        f.write(f"RMSE: {rmse_value:.2f} pixels\n\n")
        f.write("Actual Points:\n")
        for pt in actual_points:
            f.write(f"{pt}\n")
        f.write("\nPredicted Points:\n")
        for pt in predicted_points:
            f.write(f"{pt}\n")
    print(f"[SAVED] Log file saved to {log_path}")

# ---------- MAIN ----------
def main(moon_image_path):
    full_image = cv2.imread(moon_image_path)
    if full_image is None:
        print("Error: Image not found!")
        return

    # --- Phase 1: Collect clicks ---
    cv2.namedWindow("Moon Map")
    cv2.setMouseCallback("Moon Map", click_points, full_image)

    print("[INFO] Click on image to set trajectory points. Press 'q' when done.")
    while True:
        temp = full_image.copy()
        if len(points_so_far) > 1:
            for i in range(1, len(points_so_far)):
                cv2.line(temp, points_so_far[i-1], points_so_far[i], (0,255,0), 2)
        cv2.imshow("Moon Map", temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    if len(patches) < 2:
        print("[INFO] Need at least 2 patches!")
        return

    # --- Phase 2: Reconstruct trajectory ---
    trajectory_points = []

    # First patch: global match
    start_tl = orb_match(full_image, patches[0])
    if start_tl is None:
        print("[ERROR] Could not find starting patch globally.")
        return
    start_center = (start_tl[0] + patch_size // 2, start_tl[1] + patch_size // 2)
    trajectory_points.append(start_center)

    # Next patches: sliding neighbor search
    radius, step = 200, 32
    for i in range(1, len(patches)):
        prev_center = trajectory_points[-1]
        kp_q, des_q = orb.detectAndCompute(patches[i], None)
        if des_q is None:
            print(f"[WARNING] Patch {i+1} has no descriptors; skipping.")
            continue

        candidates = get_sliding_neighbors(prev_center, patch_size, full_image.shape, radius, step)
        best_score, best_loc_tl = -1, None

        for tlx, tly in candidates:
            candidate = full_image[tly:tly+patch_size, tlx:tlx+patch_size]
            kp_c, des_c = orb.detectAndCompute(candidate, None)
            if des_c is None:
                continue

            try:
                matches = bf.knnMatch(des_q, des_c, k=2)
            except:
                continue

            good, dists = [], []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
                    dists.append(m.distance)

            if len(good) == 0:
                continue

            score = len(good) - np.mean(dists)
            if score > best_score:
                best_score = score
                best_loc_tl = (tlx, tly)

        if best_loc_tl is None:
            print(f"[WARNING] Patch {i+1} not matched; using global fallback.")
            fallback_tl = orb_match(full_image, patches[i])
            if fallback_tl:
                cx = fallback_tl[0] + patch_size // 2
                cy = fallback_tl[1] + patch_size // 2
                trajectory_points.append((cx, cy))
        else:
            cx = best_loc_tl[0] + patch_size // 2
            cy = best_loc_tl[1] + patch_size // 2
            trajectory_points.append((cx, cy))

    # --- Phase 3: RMSE + Visualization ---
    rmse, _ = compute_rmse(points_so_far, trajectory_points)
    draw_trajectories_with_error(full_image, points_so_far, trajectory_points, rmse)
    save_results(full_image.copy(), points_so_far, trajectory_points, rmse)

# ---------- RUN ----------
if __name__ == "__main__":
    main("Moon_image_dataset/Equirectangular/Equi_3.png")