import collections
import csv
import glob
import os.path
import pickle

import cv2
import numpy as np

def prepare_im(path):
    # loads the image from a path and converts it to greyscale
    img = cv2.imread(path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.rotate(grey, cv2.ROTATE_90_CLOCKWISE)
    grey = grey.astype(np.float64)
    return grey

def does_have_xandy(pixel_sample):
    has_front = False
    has_side = False

    for angle_name in pixel_sample:
        if "front" in angle_name or "back" in angle_name:
            has_front = True
        elif "left" in angle_name or "right" in angle_name:
            has_side = True

    return has_front and has_side


def get_brightest_front_side(brights):
    front, f_val = None, 0
    side, s_val = None, 0
    for angle_name in brights:
        if "front" in angle_name or "back" in angle_name:
            if brights[angle_name] > f_val:
                f_val = brights[angle_name]
                front = angle_name
        elif "left" in angle_name or "right" in angle_name:
            if brights[angle_name] > s_val:
                s_val = brights[angle_name]
                side = angle_name
    return [front, side]


if __name__ == "__main__":
    # so long as the brightness is this many units greater than the background, we detected a light
    threshold = 50

    pixels_samples = collections.defaultdict(dict)
    pixels_samples_brightness = collections.defaultdict(dict)

    sides = ["front1", "left1", "back1", "right1"]
    for side in sides:
        # load the background image
        background = prepare_im(f"./data/{side}/background.jpg")


        # for each of the images we have in data
        for img_path in glob.glob(f"./data/{side}/*.jpg"):
            light_im = prepare_im(img_path)

            # remove the background so that we are looking at the change in lighting conditions
            diff = np.clip(light_im - background, 0, None)

            # eroding to remove reflections
            kernel = np.ones((6, 6), np.uint8)
            diff = cv2.erode(diff, kernel, cv2.BORDER_REFLECT)

            #  threshold out any change below our defined threshold
            threshed = np.zeros_like(diff, dtype=np.uint8)
            threshed[diff > threshold] = diff[diff > threshold]

            # get the location of the maximum brightness in the image
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(threshed)

            # if we detected a change then
            if maxVal > 0:
                filename = os.path.basename((img_path))
                pixel_id = int(filename.replace(".jpg", ""))

                pixels_samples[pixel_id][side]= [maxLoc[0], maxLoc[1]]
                pixels_samples_brightness[pixel_id][side]= maxVal

            # show the images and the detections
            image = light_im.copy().astype(np.uint8)
            image = np.dstack([image, image, image])
            cv2.circle(image, maxLoc, 10, (255, 0, 0), 2)
            cv2.imshow("im", image)
            cv2.imshow("diff", diff.astype(np.uint8))
            cv2.waitKey(1)


    # for each angle we imaged from, normalize the range of the locations
    for side in sides:
        locations = []
        for pixel_id in pixels_samples:
            xy_loc = pixels_samples[pixel_id].get(side, None)
            if xy_loc:
                locations.append(xy_loc)

        min_pos = np.min(locations, axis=0)
        max_pos = np.max(locations, axis=0)



        for pixel_id in pixels_samples:
            xy_loc = pixels_samples[pixel_id].get(side, None)
            if xy_loc:
                # normalize the coordinate
                xy_loc = (xy_loc - min_pos) / (max_pos - min_pos)

                # flip the coordinates if we are in the back or the right hand side
                if "back" in side or "right" in side:
                    xy_loc[0] = 1 - xy_loc[0]


                #flip vertically because the top of an image is 0, and we want the bottom of the tree to be zero
                xy_loc[1] = 1 - xy_loc[1]

                # rescale the coordinate to the dimensions of the tree
                xy_loc[0] *= 100
                xy_loc[1] *= 300

                pixels_samples[pixel_id][side] = xy_loc


    pixel_coords = []
    for pixel_id in pixels_samples:
        if does_have_xandy(pixels_samples[pixel_id]):
            best_samples = get_brightest_front_side(pixels_samples_brightness[pixel_id])

            # get all X samples
            X_samples = [pixels_samples[pixel_id][s][0] for s in best_samples if "front" in s or "back" in s]
            x = np.array(X_samples).mean()

            Y_samples = [pixels_samples[pixel_id][s][0] for s in best_samples if "left" in s or "right" in s]
            y = np.array(Y_samples).mean()

            Z_samples = [pixels_samples[pixel_id][s][1] for s in [best_samples[0],]]
            z = np.array(Z_samples).mean()

            pixel_coords.append((pixel_id, x, y, z))

    with open("./data/coords.csv", "w", newline="\n") as f:
        writer = csv.writer(f)
        for row in pixel_coords:
            writer.writerow(row)