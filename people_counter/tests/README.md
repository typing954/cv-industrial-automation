# Tests

34 automated tests. They need **no GPU, no internet and no cloud account**: the video is synthetic, and the cloud endpoint is replaced
by a fake. The whole suite takes about 15 seconds.

## Running them
```bash
source .venv/bin/activate
pip install pytest                      # once; the rest of the packages come from requirements.txt
python -m pytest tests                  # everything
python -m pytest tests -v               # list every test as it runs
python -m pytest tests/test_alert.py    # one file
python -m pytest tests -k merge         # only tests whose name contains "merge"
```
`test_cloudstream.py::test_server_app_with_stub_model` is skipped (not failed) if `fastapi` or `httpx` is missing
(`pip install fastapi httpx`).

## What each file covers
| File | Tests | Code under test | What it checks |
|---|---|---|---|
| `test_alert.py` | 7 | `peoplecount/alert.py` | The alert rules: alert after the count stays over the limit for `hold` seconds, cleared after `clear` seconds, every violation recorded with its first-exceeded time, peak and duration, short blips and short dips handled, an open violation closed when the video ends, optional cooldown. |
| `test_clips.py` | 5 | `peoplecount/clips.py` | Evidence clips run from 3 s before the violation to 3 s after the alert; overlapping or nearly adjacent alerts merge into one clip; distant alerts stay separate; a clip near the start of the video is cut to the available footage. |
| `test_target_fps.py` | 8 | `peoplecount/video.py` | `--target-fps`: frames are picked evenly in time (including ratios like 25 to 10 FPS), a target above the video's FPS uses every frame, bad values are rejected, `--realtime` paces the video and drops frames when the detector is too slow. |
| `test_cloudstream.py` | 14 | `cloudstream/` | The cloud-endpoint client: wire format, the looping video player, the SageMaker and Vertex transports (with stand-in clients), the inference server's routes (with a stub model), and the full streaming pipeline against a fake endpoint: ordering, same alert behaviour as the local pipeline, one clip for overlapping alerts, dropping frames when the endpoint is slow, surviving errors. |

## How to read a test
- **Docstrings** at the top of each file explain the system in a paragraph, and each test has a one-line docstring saying what behaviour it proves.
- **Timings are spelled out in comments.** In `test_alert.py` a list of counts is played one sample per second, so a number's position in the list is its time in seconds. In `test_clips.py` the comments give each clip's expected window.
- **Tolerances have reasons.** The clip tests allow 0.15 s because frames are 0.1 s apart. The pipeline tests allow a few tenths of a second because the count is smoothed over several frames.

## Test data and fakes (nothing to download)
- **Synthetic videos:** the video tests write a small video file with OpenCV at the start of the run (a fixture shared by all tests in a file). In `test_cloudstream.py`, frame number *i* is a gray image of brightness 2*i*, so the fake endpoint can read the frame number back from the image.
- **`FakeEndpoint`** stands in for a hosted YOLO endpoint. It reports 5 people from 1 s to 3 s of the video, and adds random delays so that replies arrive out of order, like a real network.
- **Stand-in cloud clients** replace boto3 (SageMaker) and the Vertex SDK, and a stub model replaces YOLO in the server test.

## Adding a test
1. Put it in the file for the code it covers, or create `test_<area>.py`.
2. Give it a descriptive name and a one-line docstring that says the behaviour, not the implementation.
3. Comment any number that isn't obvious, such as a timestamp, a tolerance or a count.
4. Reuse the helpers in that file (`run`, `feed`, `run_pipeline`, the `video` fixture) rather than writing new ones.

## Not covered by these tests
- Real detection quality (accuracy of the YOLO models) and real GPU speed.
- The Docker build and the SageMaker and Vertex deploy scripts, and real cloud calls.
- The notebooks and the report charts. These were checked by running them by hand.
