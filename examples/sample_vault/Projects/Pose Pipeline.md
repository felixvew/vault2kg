---
type: Project
status: active
owner: "[[Anna Weber]]"
related:
  - "[[Release Timing Study]]"
  - [[MediaPipe]]
start: 2026-03-01
---
The Pose Pipeline turns archery training videos into joint-angle time series.

## Architecture

The pipeline runs RT-DETR for person detection, then [[MediaPipe]] for keypoints. See ![[pipeline.png]] and [the spec](../Specs/Pipeline%20Spec.md). Background reading: [MediaPipe docs](https://developers.google.com/mediapipe).

```python
# [[NotALink]] #notatag
angles = compute(keypoints)
```

## Open questions

- Does [[Missing Note]] already exist? Compare with the [[#Architecture]] section.
- Use `#inline-code` tags? No.
