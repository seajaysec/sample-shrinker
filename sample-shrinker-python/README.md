
# Sample Shrinker

A Python script to conditionally batch-convert audio samples into minimal `.wav` files, based on target criteria. This script is useful for saving storage space and reducing the I/O stress during simultaneous real-time streaming of multiple `.wav` files on devices like the Dirtywave M8 tracker.

If you have directories full of 24/32-bit stereo `.wav` files or stereo samples with effectively mono content, this script can reclaim wasted storage space and reduce I/O stress on your SD card. It can also detect if the content of a stereo sample is actually mono and convert it automatically!

## Features
- **Conditional Conversion**: Only converts samples that don't meet the target criteria (bit depth, channels, etc.).
- **Auto-Mono**: Automatically convert stereo samples to mono if the content is effectively mono, with a configurable threshold.
- **Backup and Spectrogram Generation**: Converted files are backed up (unless disabled) and spectrograms of old vs. new files are generated.
- **Pre-Normalization**: Optionally normalize samples before downsampling the bit depth to preserve dynamic range.
- **Parallel Processing**: Use the `-j` option to process multiple files in parallel for faster conversions.

## Requirements

- Python 3.10 or later
- `pydub`, `librosa`, `matplotlib`, `soundfile` (install with `pip`)
- `ffmpeg` or `libav` installed for `pydub`

Install dependencies:
```bash
pip install -r requirements.txt
```

You will also need `ffmpeg`:
```bash
# MacOS with Homebrew
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

```bash
python sample-shrinker.py [options] FILE|DIRECTORY ...
```

### Basic Example:
```bash
python sample-shrinker.py directory_of_samples/
```

This will:
- Convert samples in place with a target bit depth of 16 and stereo channels unchanged.
- Back up the original files in a parallel `_backup/` directory.
- Generate `.png` spectrograms comparing old and new files.

### Options:
- `-b BIT_DEPTH`: Set the target bit depth (default: 16). Samples will only be downsampled unless `-B` is set.
- `-B MIN_BIT_DEPTH`: Set a minimum bit depth. This will upsample any samples below the minimum.
- `-c CHANNELS`: Set the target number of output channels (default: 2). For mono, use `-c 1`.
- `-r SAMPLERATE`: Set the target sample rate (default: 44100 Hz).
- `-R MIN_SAMPLERATE`: Set a minimum sample rate. Samples below this will be upsampled.
- `-a`: Automatically convert stereo samples to mono if they are effectively mono.
- `-A DB_THRESHOLD`: Set the auto-mono threshold in dB (default: `-95.5`). This implies `-a`.
- `-p`: Pre-normalize samples before downsampling bit depth.
- `-S`: Skip generating spectrogram files.
- `-d BACKUP_DIR`: Set a directory to store backups. Use `-d -` to disable backups and spectrogram generation.
- `-l`: List files and preview changes without converting.
- `-n`: Dry run—log actions without converting any files.
- `-j JOBS`: Process files in parallel with multiple jobs (default: 1).
- `-v`: Increase verbosity.

## Examples

### Convert a Directory with Default Settings
```bash
python sample-shrinker.py my_samples/
```
- Convert samples to 16-bit with channels left unchanged.
- Back up the original files under `_backup/`.
- Generate spectrogram `.png` files for comparison.

### Convert to Mono Automatically for Effectively Mono Samples
```bash
python sample-shrinker.py -a my_samples/
```
- Automatically convert stereo samples to mono if they are effectively mono (i.e., the difference between the channels is below the threshold).

### Preview Changes Without Modifying Files
```bash
python sample-shrinker.py -l -a -A -80 my_samples/
```
- Lists all files and shows which ones would be changed without actually modifying them. The threshold for auto-mono is set to -80 dB.

### Convert and Skip Backups
```bash
python sample-shrinker.py -d - my_samples/
```
- Converts files but does not create backups or generate spectrograms.

### Pre-Normalize Before Downsampling
```bash
python sample-shrinker.py -p my_samples/
```
- Normalize the audio before downsampling the bit depth to preserve as much dynamic range as possible.

### Process Files in Parallel
```bash
python sample-shrinker.py -j 10 my_samples/
```
- Process up to 10 files at the same time for faster batch conversion.

## Output Example:

```bash
Processing file: /Volumes/Untitled/Samples/wii sports/sound effects/Baseball/Sample_0028.wav
/Volumes/Untitled/Samples/wii sports/sound effects/Baseball/Sample_0028.wav [UNCHANGED]
Processing file: /Volumes/Untitled/Samples/wii sports/sound effects/Boxing/Sample_0029.wav
/Volumes/Untitled/Samples/wii sports/sound effects/Baseball/Sample_0029.wav [CHANGED]: sample rate 48000 -> 44100
Processing file: /Volumes/Untitled/Samples/wii sports/sound effects/Boxing/Sample_0030.wav
/Volumes/Untitled/Samples/wii sports/sound effects/Baseball/Sample_0030.wav[CHANGED]: auto-mono
```

In the updated output format:
- The script logs each file being processed with the `Processing file:` prefix.
- After processing, each file will either be marked as `[UNCHANGED]` or `[CHANGED]` depending on whether any modifications (bit depth, sample rate, or channels) were made.
- If changes are made, the specific adjustments (e.g., `sample rate 48000 -> 44100`) will be displayed.
  
### Additional Details:
- The `[CHANGED]` notation follows files that were modified.
- `[UNCHANGED]` appears for files that meet the target criteria and required no modifications.
- **Changes made**:
  - Sample rate conversions (e.g., `sample rate 48000 -> 44100`).
  - Bit depth reductions (e.g., `bit depth 32 -> 16`).
  - Channel conversions (e.g., stereo to mono).
- Verbose output (`-v`) will print additional information such as ongoing file processing.
