# Annotator

This is a simple tool for annotating (short) videos. It is designed to facilitate frame-level annotations, at the expense of having to load the entire video into memory. Future versions may relax this requirement.

## Building

Requirements are:

 * libconfuse (for parsing config file)
 * opencv (for video playback)

Compile by running:

```console
./configure
make
```

## Usage

Available labels are configured in `annotator.conf`. Controls are as follows:

 * Number keys: Make annotation
 * Escape: Quit (doesn't save)
 * Space: Pause
 * Left arrow: Go backward 1 frame
 * Right arrow: Go forward 1 frame
 * Up arrow: Seek backard 1 second
 * Down arrow: Seek forward 1 second
 * u: Undo
 * r: Redo
 * Delete: Remove closest annotation to current frame
 * s: Save (there is no autosave)

