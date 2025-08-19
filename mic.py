import os
import threading
from collections import deque
from pathlib import Path
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------- Config ----------
SAMPLE_RATE = 37120
CHANNELS = 1
DTYPE = "int16"
CLIP_SECONDS = 20
OUTPUT_DIR = Path("clips")
OUTPUT_DIR_ANALYZE = OUTPUT_DIR / "output"
PROCESSOR_BASE = ["python", "analyze.py"]   # analyzer entry
INPUT_DEVICE = None                         # or device index/name from sd.query_devices()
BLOCKSIZE = 2048                            # frames per audio callback; 0 lets backend choose
PUBLISH_ROLLING_ALIAS = True                # also update clips/clip.wav each segment
PUBLIC_FILE = OUTPUT_DIR / "clip.wav"       # rolling alias (optional)
# ----------------------------

# Shared buffer of audio blocks from the callback (1D np.int16 arrays)
_shared_blocks = deque()
_buf_lock = threading.Lock()
_stop_flag = threading.Event()

def _audio_callback(indata, frames, time_info, status):
    # Keep this super fast to avoid XRuns
    if status:
        # You can log status if needed
        pass
    # Flatten to 1D int16 (mono). If CHANNELS > 1, select or mix as needed.
    block = indata.copy().reshape(-1).astype(np.int16)
    with _buf_lock:
        _shared_blocks.append(block)

def _pop_all_blocks():
    """Move all blocks from the shared deque into a local deque quickly."""
    local = deque()
    with _buf_lock:
        while _shared_blocks:
            local.append(_shared_blocks.popleft())
    return local

def _gather_samples(local_q, carry, needed):
    """
    Pull exactly `needed` samples from carry + local_q.

    Returns:
      out: np.int16 of length `needed` if enough data, else None
      new_carry: leftover samples not used this round
      consumed_any: whether we consumed from local_q (for sleep behavior)
    """
    # Fast path if carry already holds enough
    if carry.size >= needed:
        return carry[:needed], carry[needed:], False

    consumed_any = False
    # Move all currently queued blocks into a list (we'll re-carry leftovers)
    if local_q:
        consumed_any = True
        chunks = [carry]
        while local_q:
            chunks.append(local_q.popleft())
        big = np.concatenate(chunks) if len(chunks) > 1 else carry
    else:
        big = carry

    if big.size < needed:
        # Not enough yet: keep everything in carry
        return None, big, consumed_any

    out = big[:needed]
    new_carry = big[needed:]
    return out, new_carry, consumed_any


def _save_clip_and_launch(clip_samples, clip_index):
    """
    Save the clip to clips/clip.wav (atomically) and launch analyzer on that path.
    Only ONE file is kept; each new segment replaces the previous.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_ANALYZE.mkdir(parents=True, exist_ok=True)

    pub_tmp = OUTPUT_DIR / "clip.tmp.wav"
    # write temp, then atomic swap to avoid partial reads
    sf.write(str(pub_tmp), clip_samples.reshape(-1, 1), SAMPLE_RATE, subtype="PCM_16")
    os.replace(pub_tmp, PUBLIC_FILE)  # atomic on macOS/Linux

    # Run analyzer on the rolling file
    analyze_cmd = (
        PROCESSOR_BASE
        + ["-i", str(PUBLIC_FILE),
           "-o", str(OUTPUT_DIR_ANALYZE),
           "--offset", str(clip_index * CLIP_SECONDS)]
    )
    subprocess.Popen(analyze_cmd)

    print(f"Saved clip.wav (offset {clip_index * CLIP_SECONDS}s)")
    
def _segmenter():
    needed = SAMPLE_RATE * CLIP_SECONDS
    i = 0
    carry = np.empty((0,), dtype=np.int16)
    local_q = deque()

    while not _stop_flag.is_set():
        # Pull any new blocks from shared buffer
        new_blocks = _pop_all_blocks()
        if new_blocks:
            local_q.extend(new_blocks)

        # Try to cut as many full clips as available
        made_clip = False
        while True:
            out, carry, consumed = _gather_samples(local_q, carry, needed)
            if out is None:
                break
            _save_clip_and_launch(out, i)
            i += 1
            made_clip = True

        # Back off briefly if nothing happened to avoid busy waiting
        if not made_clip and not new_blocks:
            sd.sleep(3)  # a few milliseconds

def main():
    print(f"Recording continuously at {SAMPLE_RATE} Hz. Ctrl+C to stop.")
    seg_thread = threading.Thread(target=_segmenter, daemon=True)
    seg_thread.start()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=INPUT_DEVICE,
            blocksize=BLOCKSIZE,
            callback=_audio_callback,
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        _stop_flag.set()
        seg_thread.join()
        print("Done.")

if __name__ == "__main__":
    main()
