import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

def is_blurry(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def extract_frames(video_path, output_folder, frames_per_second, blur_threshold=20):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get frames per second

    try:
        # Iterate through video frames until the end
        while success:
            if count % int(fps / frames_per_second) == 0:
                # Check if the frame is too blurry
                if not is_blurry(image, blur_threshold):
                    # Write the frame image to file
                    cv2.imwrite(f"{output_folder}/frame{count:04d}.jpg", image)
            success, image = vidcap.read()
            count += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Release the video object
        vidcap.release()

# Define your quality criteria weights
weight_resolution = 1.0
weight_sharpness = 1.0
weight_aesthetic = 1.0  # Weight for aesthetic quality
# Add weights for other criteria as needed

# Define your quality criteria (modify these functions as needed)
def evaluate_resolution(image):
    return image.shape[0] * image.shape[1]

def evaluate_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Define aesthetic evaluation function
def evaluate_aesthetic(image):
    # Calculate variance of color channels
    b, g, r = cv2.split(image)
    b_var = np.var(b)
    g_var = np.var(g)
    r_var = np.var(r)

    # Compute texture features using Gabor filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 1, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_features.extend([np.mean(filtered), np.var(filtered)])

    # Combine color variance and texture features
    aesthetic_score = (b_var + g_var + r_var) + sum(gabor_features)

    return aesthetic_score

# Combine individual quality scores into a single metric
def quality_score(image):
    resolution_score = evaluate_resolution(image)
    sharpness_score = evaluate_sharpness(image)
    aesthetic_score = evaluate_aesthetic(image)
    combined_score = (resolution_score * weight_resolution) + \
                     (sharpness_score * weight_sharpness) + \
                     (aesthetic_score * weight_aesthetic)
    return combined_score

def select_best_quality(cluster):
    best_image = None
    best_score = -float('inf')
    for image in cluster:
        score = quality_score(image)
        if score > best_score:
            best_image = image
            best_score = score
    return best_image

# Define input and output folders
input_folder = "input_images"
output_folder = "output_images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load your image dataset from input folder
def load_images_from_folder(folder):
    image_list = os.listdir(folder)
    images = [cv2.imread(os.path.join(folder, image_path)) for image_path in image_list]
    return images

@app.post("/process_video/")
async def process_video(frames_per_second: int, num_clusters: int, video_file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        video_path = os.path.join("uploads", video_file.filename)
        with open(video_path, "wb") as buffer:
            buffer.write(video_file.file.read())

        # Extract frames from the uploaded video
        extract_frames(video_path, input_folder, frames_per_second)

        # Load images from the input folder
        images = load_images_from_folder(input_folder)

        # Clustering
        features = np.array([image.flatten() for image in images])
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features)
        cluster_labels = kmeans.labels_

        # Select best image from each cluster and save to output folder
        for cluster_id in range(num_clusters):
            cluster = [images[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            best_image = select_best_quality(cluster)
            output_path = os.path.join(output_folder, f"best_image_cluster_{cluster_id}.jpg")
            cv2.imwrite(output_path, best_image)

        return {"message": "Video processed successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
