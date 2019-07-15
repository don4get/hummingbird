import os
from moviepy.editor import *


def create_gifs():
    list_avi = [avi for avi in os.listdir(os.getcwd()) if "avi" in avi]
    list_gif = [avi for avi in os.listdir(os.getcwd()) if "gif" in avi]
    print(list_avi)

    for avi in list_avi:
        short_name = avi.split(".")[0]
        if (short_name+".gif") not in list_gif:
            vid = VideoFileClip(avi)
            dur = vid.duration
            clip = (vid.subclip(0.2, dur-0.2).speedx(5).resize(0.5))
            clip.write_gif(short_name+'.gif', fps=15)


if __name__ == "__main__":
    create_gifs()
