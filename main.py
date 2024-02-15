from exif import Image
from datetime import datetime
import cv2
import math
import numpy as np


def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
        # Convert descriptors to NumPy arrays
    descriptors_1 = np.array(descriptors_1, dtype=np.uint8)
    descriptors_2 = np.array(descriptors_2, dtype=np.uint8)

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    #@ zavrit okno klavesou "0"
    cv2.waitKey(0)
    cv2.destroyWindow('matches')
    
#@ tqdm se musime zbavit pred odevzdanim
from tqdm import tqdm
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in tqdm(matches):
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = (feature_distance * GSD)/ 100000
    speed = distance / time_difference
    return speed

def main():
    image_1 = "imgs/photo_0676.jpg"
    image_2 = "imgs/photo_0677.jpg"

    show_matches = False
    time_difference = get_time_difference(image_1, image_2) # zjisti casovy rozdil
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # preformatovani obrazku na opencv objekty
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # najde keypointy a descriptory
    keypoints_celkem = len(keypoints_1)

    if show_matches: # pokud je potreba vdet keypointy
        match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None) 
        resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
        cv2.imshow('pred_filtrem', resize)
    keypoints_po_filtru = len(keypoints_1)
    matches = calculate_matches(descriptors_1, descriptors_2) # najde kaypointy co jsou na opou obrazcich
    if show_matches: # zase pokud je potreba vydet keypointy
        display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) 
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
    speed = speed*1.06396
    print(f"keypoints celkem: {keypoints_celkem}")
    print(f"keypoints po filtru: {keypoints_po_filtru}")
    print(f"prumerna vzdalenost features (km) - {average_feature_distance}")
    print(f"rozdil casu (s) - {time_difference}")
    print(f"rychlost (km/s) - {speed}")
    #TODO dat vysledek do iss_speed.txt
    #TODO vyresit cesty k obrazkum
    #TODO chytit exceptions kdyz nejsou obrazky
    #TODO otestovat podle: https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/5

if __name__ == "__main__":
    main()
