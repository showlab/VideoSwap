import os

import imageio


def frames_to_video(frames_folder, output_video_path, fps):
    frame_files = sorted(os.listdir(frames_folder))

    writer = imageio.get_writer(output_video_path, fps=fps)

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = imageio.imread(frame_path)
        writer.append_data(frame)

    writer.close()


if __name__ == '__main__':
    root = 'results/1007_final_ss0897_T07_Iter50/visualization/source/frames'
    output_video_path = 'results/1007_final_ss0897_T07_Iter50/visualization/source_fps30.mp4'
    fps = 30
    frames_to_video(root, output_video_path, fps)
