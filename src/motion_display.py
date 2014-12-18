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


def create_motion_field_video(vector_data, video_filename, output_filename, display=False):
    """

    :param vector_data:
    :param video_filename:
    :param output_filename:
    :return:
    """
    info_all_frames = vector_data
    frame_count = info_all_frames['framenum'].max()

    logging.info("opening video {}".format(video_filename))
    vin = cv2.VideoCapture(video_filename)
    logging.info("retreiving video info")
    width = int(vin.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    heigh = int(vin.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = vin.get(cv2.cv.CV_CAP_PROP_FPS)
    if not fps == fps: # isnan
        logging.warning('unable to read fps value from input video, set default 30fps.')
        fps = 30 # set 30 as default value
    logging.info("info: {}x{}@{}".format(width, heigh, fps))

    logging.info('creating video file {}'.format(output_filename))
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    vout = cv2.VideoWriter()
    vout.open(output_filename, fourcc, fps, (width, heigh))

    for frame_number in range(1, frame_count+1):
        logging.info('loading frame {}'.format(frame_number))
        ret, image = vin.read()
        if not ret:
            logging.info('no more images in video')
            break

        #select motion vector for the current frame only
        info_frame = info_all_frames[info_all_frames['framenum'] == frame_number]
        if info_frame.empty: # happens for keyframes, just skip it
            logging.info('no vector info for frame {}'.format(frame_number))
        #     continue

        logging.info('drawing {} motion vectors'.format(info_frame.shape[0]))
        # todo: move out plot function
        for index, vector in info_frame.iterrows():
            x1, y1, x2, y2, w, h = (int(v) for v in vector[['srcx', 'srcy', 'dstx', 'dsty', 'blockw', 'blockh']].tolist())
            c = (255, 127+5*(x2-x1), 127+5*(y2-y1))
            # cv2.rectangle(image, (x2, y2), (x2+w, y2+h), c, -1)
            cv2.line(image, (x1, y1), (x2, y2), c, 2)

        if display:
            cv2.imshow('image', image)
            cv2.waitKey(100)

        logging.info('writing image')
        vout.write(image)  # probably threaded: make sure image exists a long time... very confusing.

    logging.info('exiting')
    vout.release()
    vin.release()


def main():
    parser = argparse.ArgumentParser(description='create a video of the vector field.')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('-d', '--display', action='store_true', help='debug will display image while processing')
    parser.add_argument('-f', '--video', required=True, help='input video')
    parser.add_argument('-i', '--input_path', required=True, help='input path to motion vector file (.csv)')
    parser.add_argument('-o', '--output_path', required=True, help='output path to video')
    args = parser.parse_args()

    # logging handling
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    input_motion_vectors = read_motion_vector(args.input_path)
    input_video_filename = args.video
    create_motion_field_video(input_motion_vectors, input_video_filename, args.output_path, display=args.display)

if __name__ == "__main__":
    main()
