# from moviepy.video.fx.all import crop
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip

clip = VideoFileClip('pexels_3859462_v2.mp4')
(w, h) = clip.size

new_clip = clip.fx(resize, width=768, height=448)
# new_clip = crop(clip, width=768, height=448, x_center=w / 2, y_center=h / 2)
new_clip.write_videofile('pexels_3859462_v2_1.mp4')
