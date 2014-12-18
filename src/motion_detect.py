#!/usr/bin/env python2

import logging
import argparse
import numpy as np
import pandas as pd
import cv2

def read_motion_vector(filename):
    """ read the motion vector file under csv format
    :param filename: input path to input csv file
    :return: data under DataFrame format (see pandas lib)
    """
    logging.info('reading vectors from {}'.format(filename))
    return pd.read_csv(filename)


def write_motion_vector(filename, data):
    """
    write DataFrame to file
    :param filename: input path to output csv file
    :param data: input data to be writen at DataFrame format
    """
    logging.info('writing vectors to {}'.format(filename))
    data.to_csv(filename, index=False)


def filter_out_static(input_data, reprojection_threshold=2.0):
    """
    :param input_data: input DataFrame of motion vectors of all frames.
    :return: list of boolean, True when the input vector has moved
    """
    frame_count = input_data['framenum'].max()
    motion_all_mask = []
    for frame_number in range(1, frame_count+1):
        logging.info('loading frame {}'.format(frame_number))

        #select motion vector for the current frame only
        info_frame = input_data[input_data['framenum'] == frame_number]
        if info_frame.empty: # happens for keyframes, just skip it
            logging.info('no vector info for frame {}'.format(frame_number))
            continue

        # reformat source and destination coordinates
        src_pts = np.float32(info_frame[['srcx', 'srcy']]).reshape(-1, 1, 2)
        dst_pts = np.float32(info_frame[['dstx', 'dsty']]).reshape(-1, 1, 2)

        # compute 2D transformation using RANSAC method
        M, homography_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojection_threshold)
        # object in motions are the ones that does not fit the homography
        # a logical not should be enough (for first try)
        motion_mask = np.logical_not(homography_mask).ravel().tolist()
        # then accumulate the results
        motion_all_mask += motion_mask

    return motion_all_mask


def main():
    parser = argparse.ArgumentParser(description='create a video of the vector field.')
    parser.add_argument('input_path', help='input path to motion vector file (.csv)')
    parser.add_argument('output_path', help='output path to video')
    parser.add_argument('-t', '--threshold', type=float, default=2.0, help='threshold value in pixel for motion detection')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    # logging handling
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    input_motion_vectors = read_motion_vector(args.input_path)
    mask = filter_out_static(input_motion_vectors, args.threshold)
    output_motion_vectors = input_motion_vectors[mask]
    write_motion_vector(args.output_path, output_motion_vectors)


if __name__ == "__main__":
    main()
