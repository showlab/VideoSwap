import cv2


def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print('无法打开视频文件')
        return

    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output_path = os.path.join(output_folder, f'{frame_count:05d}.jpg')

        cv2.imwrite(output_path, frame)

        frame_count += 1

    video.release()
    print(f'已成功提取{frame_count}帧到文件夹：{output_folder}')


video_path = 'datasets/paper_evaluation/source_videos/object/pexels_3859462_v2.mp4'
output_folder = 'datasets/paper_evaluation/object/pexels_3859462_v2/frames'
extract_frames(video_path, output_folder)
