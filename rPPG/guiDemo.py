#!/usr/bin/env python3
import cv2
import numpy as np

def preproc(qAcc, qRaw, videoPath, buffer=10, frameWidth=64, frameHeight=64):
    ''' Read the video and perform necessary preprocessing
    
    Arguments:
      qAcc: Queue of accumulated boxes, signals, and cropped video
      qRaw: Queue of (mostly) raw video frames
      videoPath: Path of video to read
      buffer: Number of seconds of video to accumulate before sending a batch to qAcc
      frameWidth: Width of the cropped frame in pixels
      frameHeight: Height of the cropped frame in pixels
    '''
    import mediapipe
    from rPPG.utils import vidcap, bboxUtils
    from rPPG.bboxes import bboxesMp
    from rPPG.baselines import baselinesPreprocess
    from rPPG.crop import crop
    acc = {'boxes': [], 'signals': [], 'cropped': []}
    with mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_detection:
        with vidcap.Reader(videoPath, 'Frames Read') as cap:
            faceCropper = bboxUtils.FaceCropperPlain(frameWidth, frameHeight)
            qRaw.put(cap.fps)
            mostRecentBox = [cap.time, 100, 100, 200, 200]
            for frame in cap:
                box = bboxesMp.calcBoxSingle(frame, cap.time, face_detection, postprocess=True)
                if box:
                    acc['boxes'].append(box)
                    mostRecentBox = box
                elif mostRecentBox:
                    acc['boxes'].append([cap.time] + [b for b in mostRecentBox[1:]])
                if len(acc['boxes']) > 0:
                    signals = baselinesPreprocess.calcSignalsFrame(frame, acc['boxes'][-1][1:])
                    if not np.any(signals) and len(acc['signals']) > 0:
                        signals = acc['signals'][-1]
                    acc['signals'].append(signals)
                    acc['cropped'].append(crop.cropSingle(frame, acc['boxes'][-1], None, faceCropper))
                    # Mess with raw frame
                    box = acc['boxes'][-1]
                    cv2.rectangle(frame, (box[1], box[2]), (box[3], box[4]), (0, 255, 0), 2)
                    qRaw.put(np.array(frame))
                    if acc['boxes'][-1][0] - acc['boxes'][0][0] >= buffer:
                        acc['signals'] = np.vstack(acc['signals'])
                        qAcc.put(acc)
                        acc = {name: [] for name in acc} # Wipe out acc
    qRaw.put(None)

def inferPulse(qAcc, qInfer, modelPath=None):
    ''' Infer the pulse waveform over the most recent accumulated values

    Arguments:
      qAcc: Queue of accumulated boxes, signals, and cropped video
      qInfer: Queue of outputted inferences
      modelPath: Path to deep learning model with which to infer
    '''
    if modelPath:
        import torch
        from rPPG.utils import modelLoader
        model, _ = modelLoader.load(modelPath)
        # Evaluate over the model
        model.eval()
        device = next(model.parameters()).device
    while True:
        accumulated = qAcc.get()
        # Calculate fps for the accumulated frames
        fps = len(accumulated['boxes'])/(accumulated['boxes'][-1][0]-accumulated['boxes'][0][0]) # 0th index of each box is timestamp
        # Evaluate over the baselines
        from rPPG.baselines import CHROM, POS
        results = {'fps': fps, 'CHROM': CHROM.process_CHROM(accumulated['signals'], fps), 'POS': POS.process_POS(accumulated['signals'], fps)}
        if modelPath:
            from rPPG.utils import augmentations
            clip = augmentations.transformClip(np.array(accumulated['cropped']), model.config.channels())
            clip = augmentations.normalize(clip, normalization=model.config.normalization())
            # Add dim for batch
            clip = clip[np.newaxis,:]
            clip = torch.from_numpy(clip).float().to(device)
            outputs = model(clip)
            from copy import deepcopy
            results[model.config.architecture()] = deepcopy(outputs.cpu().detach().numpy())[0]
            del outputs
        qInfer.put(results)

def postproc(qInfer, qResults):
    ''' Perform postprocessing heart rate calculation
    
    Arguments:
      qInfer: Queue of pulse waveform inferences
      qResults: Queue of outputted pulse waveforms and heart rates
    '''
    inferences = []
    from utils import hr
    while True:
        inf = qInfer.get()
        fps = inf['fps']
        del inf['fps']
        inferences.append(inf)
        inf = {key: val for key, val in inf.items()} # Make a copy
        i = len(inferences)-2
        while len(inf['CHROM'])/fps < 10 and i >= 0:
            for stream in inf:
                inf[stream] = np.concatenate((inferences[i][stream], inf[stream]))
            i -= 1
        window=len(inf['CHROM']*2/fps) # Window is larger than data ensuring single result for each stream
        HRs = hr.calcHR(inf, fps, window=window)
        HRs = {stream: HR[0][0] for stream, HR in HRs.items()}
        results = {'fps': fps,
                   'waves': inferences[-1],
                   'hr': HRs}
        qResults.put(results)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a GUI demo of rPPG. Continues to run until the user presses "q".', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', default=0, help='Path to a video file to process (default uses webcam)')
    parser.add_argument('--buffer', default=5, type=float, help='Number of seconds of video buffer to process in a batch')
    from importlib.resources import files
    parser.add_argument('--model', default=files('models').joinpath('mspm/mspm.model'), help='Deep learning model')
    parser.add_argument('--out', help='Path to write video output')

    args = parser.parse_args()

    import multiprocessing as mp
    import queue
    from utils import hr

    # Make some queues
    qAcc = mp.Queue()
    qRaw = mp.Queue()
    qInfer = mp.Queue()
    qResults = mp.Queue()

    # Start background tasks
    bg = [mp.Process(target=preproc, args=(qAcc, qRaw, args.video, args.buffer)),
          mp.Process(target=inferPulse, args=(qAcc, qInfer, args.model)),
          mp.Process(target=postproc, args=(qInfer, qResults))]
    for b in bg:
        b.start()

    outWriter = None
    accResults = []
    # First item in the raw queue is fps
    fps = qRaw.get() # This is a faux fps as estimated by opencv. Do not use except for writing output!
    # Figure out timing so that we can display an animated plot
    import time
    timeLastResults = None
    # Start pulling frames from the raw queue
    while True:
        frame = qRaw.get()
        if frame is None:
            break
        try:
            accResults.append(qResults.get_nowait())
            timeLastResults = time.time()
        except queue.Empty:
            pass
        if accResults: # Annotate results to the frame
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for i, (stream, HR) in enumerate(sorted(accResults[-1]['hr'].items())):
                ybounds = (50+50*i, 50*i)
                color = colors[i%len(colors)]
                frame = cv2.putText(frame, f'{stream}: {HR:.1f} BPM', (10, ybounds[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                wave = accResults[-1]['waves'][stream]
                fracThrough = min(1, (time.time()-timeLastResults)/args.buffer)
                if len(accResults) == 1: # Trim down wave
                    wave = wave[:max(1, int(len(wave)*fracThrough))]
                else: # Move a sliding across two waves
                    wave = np.append(accResults[-2]['waves'][stream], wave)
                    wave = wave[int(len(wave)*fracThrough/2):int(len(wave)/2+len(wave)*fracThrough/2)]
                # Scale wave to ybounds
                wave = np.array(wave) * (ybounds[1]-ybounds[0]) / (max(wave)-min(wave)+0.001)
                wave = wave - min(wave) + min(ybounds)
                # plot: https://stackoverflow.com/q/35281427/4888839
                x = np.linspace(300, frame.shape[1]-50, num=len(wave))
                pts = np.vstack((x,wave)).astype(np.int32).T
                cv2.polylines(frame, [pts], isClosed=False, color=color)
        cv2.imshow('Results', frame)
        if args.out:
            if outWriter is None:
                outShape = (frame.shape[1], frame.shape[0])
                outWriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'XVID'), fps, outShape)
            outWriter.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    if outWriter is not None:
        outWriter.release()
    cv2.destroyAllWindows()
    for b in bg:
        b.terminate()
    for b in bg:
        b.join()
    print('\nExited\x1b[?25h')
