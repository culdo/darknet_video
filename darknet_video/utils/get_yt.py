from youtube_dl import YoutubeDL


def get_yt_vid(yt_url, download=False, **kwargs):
    ydl_opts = {"format": "[ext=mp4]", "outtmpl": "yt.%(ext)s"}
    ydl_opts.update(kwargs)
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(yt_url, download)

    if download:
        return "yt.mp4"
    else:
        return result["url"]


if __name__ == '__main__':
    yt_video = "https://www.youtube.com/watch?v=9XPBNaLXzPo"
    print("url=%s" % get_yt_vid(yt_video))
