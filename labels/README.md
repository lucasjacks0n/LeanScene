# Hand labels on skateboarding videos

The 1,245 hand labels the bundled model was trained on: 601 cuts, 25 fade frames and 619
confirmed non-transitions, on 466 skateboarding videos from YouTube (78 hours). They were
collected by reviewing the mistakes of earlier models, so they concentrate on hard cases; they
are not a complete annotation of any video.

The videos are not ours to distribute. A label is a YouTube id and a frame index, and you
download the video yourself.

| File | Rows | Columns |
|---|---|---|
| `labels.csv` | 1,245 | `video_id`, `frame`, `time`, `label` |
| `videos.csv` | 466 | `video_id`, `fps`, `frames`, `width`, `height`, `duration`, `codec` |

Licence: [CC BY 4.0](LICENSE). The licence covers the labels, not the videos.

## What a label means

`frame` is 0-based and counts frames in display order, as reading the file frame by frame
does (an OpenCV `read()` loop, or `leanscene`). `time` is `frame / fps` in seconds.

| `label` | Meaning |
|---|---|
| `cut` | This frame is the first frame of a new shot |
| `fade` | This frame is inside a gradual transition (dissolve, fade, wipe) |
| `same` | Checked by hand: no transition at this frame |

Only the labelled frame was checked. In training, each label supervises a few frames around
it and the rest of the video is ignored:

- `cut` or `fade` at frame *f*: *f* takes the label, *f* ± 1 are ignored (a cut may be marked
  one frame off), and *f* ± 2 to *f* ± 8 count as *same*.
- `same` at frame *f*: *f* − 3 to *f* + 3 count as *same*.

Training windows are the 81 frames centred on a label.

## Getting the same frames

A frame index only means something for a particular file. `videos.csv` describes the file each
index was made on, so you can check that yours lines up. Ours were downloaded at up to 720p with:

```sh
yt-dlp -f "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b" --merge-output-format mp4 \
  -o "%(id)s.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"
```

Compare your file with its row in `videos.csv`:

```sh
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,avg_frame_rate -of csv=p=0 VIDEO_ID.mp4
```

If `frames` and `fps` match, the indices apply as they are. If they do not (YouTube offers the
same video at other frame rates, and re-encodes over time), use `time` to find the frame.

## What has and has not been checked

- The counts match the training snapshot: 1,245 labels, 466 videos, no duplicate
  (video, frame) pairs, every frame index inside its video.
- `frames` in `videos.csv` is the container's frame count. On 25 of our source files it
  equalled the count from reading every frame, and `frame / fps` was within 1 ms of the
  frame's real timestamp. The other files were not read in full; 32 have a non-standard
  average frame rate, where `time` may be less exact.
- **Not checked:** whether each video is still on YouTube, and whether a download made today
  gives the same frame count as ours. Expect some of both to fail.

The resolutions are 720p (341 videos), 480p (85) and lower (40); the codecs are H.264 (316),
AV1 (125) and VP9 (25).
