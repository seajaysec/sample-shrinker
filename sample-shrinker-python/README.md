# Sample Shrinker

A Python script to conditionally batch-convert audio samples into minimal `.wav` files and manage duplicate audio files. This script is useful for saving storage space, reducing I/O stress during simultaneous real-time streaming of multiple `.wav` files, and cleaning up duplicate samples across your library.

## Features

### Sample Conversion
- **Conditional Conversion**: Only converts samples that don't meet the target criteria (bit depth, channels, etc.)
- **Auto-Mono**: Automatically convert stereo samples to mono if the content is effectively mono
- **Backup and Spectrogram Generation**: Converted files are backed up with original folder structure preserved
- **Pre-Normalization**: Optionally normalize samples before downsampling bit depth
- **Parallel Processing**: Process multiple files simultaneously for faster conversions

### Duplicate Management
- **Multi-Level Detection**: Finds duplicates at both directory and file levels
- **Intelligent Matching**: Uses file size, content hashes, and audio fingerprinting
- **Audio Fingerprinting**: Uses spectral analysis to detect similar audio content
- **Safe Defaults**: Moves duplicates to backup instead of deleting
- **Directory Structure**: Maintains original folder structure in backup directory

## Requirements

- Python 3.10 or later
- Required Python packages (install with `pip install -r requirements.txt`):
  ```
  librosa==0.10.2.post1
  matplotlib==3.9.2
  numpy
  pydub==0.25.1
  questionary==2.0.1
  soundfile==0.12.1
  scipy>=1.11.0
  ```
- `ffmpeg` or `libav` installed for audio processing

Install system dependencies:
```bash
# MacOS with Homebrew
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

### Interactive Mode (Recommended)
Simply run the script without arguments:
```bash
python sample-shrinker.py
```

The interactive interface will guide you through:
1. Choosing between sample conversion or duplicate removal
2. Selecting directories/files to process (add multiple paths)
3. Configuring operation-specific options
4. Setting advanced parameters

### Command Line Mode
For automation or scripting:
```bash
python sample-shrinker.py [options] FILE|DIRECTORY ...
```

## Sample Conversion Options

### Interactive Configuration
When choosing "Shrink samples", configure:
- Target bit depth (8, 16, or 24 bit)
- Channel count (mono or stereo)
- Sample rate (22050, 44100, or 48000 Hz)
- Advanced options:
  - Auto-mono conversion
  - Pre-normalization
  - Spectrogram generation
  - Parallel processing
  - Minimum sample rate
  - Minimum bit depth
  - Dry run preview

### Command Line Options
- `-b BIT_DEPTH`: Set target bit depth (default: 16)
- `-B MIN_BIT_DEPTH`: Set minimum bit depth
- `-c CHANNELS`: Set target channels (1=mono, 2=stereo)
- `-r SAMPLERATE`: Set target sample rate (default: 44100)
- `-R MIN_SAMPLERATE`: Set minimum sample rate
- `-a`: Enable auto-mono conversion
- `-p`: Enable pre-normalization
- `-j JOBS`: Set number of parallel jobs
- `-n`: Preview changes without converting
- `-d BACKUP_DIR`: Set backup directory (default: _backup)

## Duplicate Removal Options

### Interactive Configuration
When choosing "Remove duplicates", configure:
- Audio matching options:
  - Similarity threshold (80-95%)
  - File length comparison
  - Sample rate comparison
  - Channel count comparison
- Filename handling:
  - Match by name and content
  - Match by content only
- Duplicate handling:
  - Move to backup (safe)
  - Delete immediately
  - Preview only

### Detection Process
1. **Directory Level**:
   - Finds directories with matching names
   - Compares file counts and total sizes
   - Verifies exact content matches
   - Keeps oldest copy, moves others to backup

2. **File Level**:
   - Groups files by size (fast initial filter)
   - Performs quick hash comparison for exact matches
   - Uses audio fingerprinting for similar content detection
   - Maintains original directory structure in backup

### Audio Fingerprinting
- Converts audio to mono for comparison
- Generates spectral fingerprints
- Compares frequency content
- Provides similarity scores as percentages
- Configurable similarity threshold

### Safety Features
- Dry run option to preview changes
- Backup by default instead of deletion
- Verification of file accessibility
- Symlink detection
- Lock checking
- Detailed progress reporting
- Original folder structure preserved in backups

## Examples

### Basic Sample Conversion
```bash
# Interactive mode with guided configuration
python sample-shrinker.py

# Command line with specific options
python sample-shrinker.py -c 1 -b 16 -a samples/
```

### Duplicate Removal
```bash
# Interactive mode (recommended)
python sample-shrinker.py

# Preview duplicate detection
python sample-shrinker.py samples/ -n
```

### Output Example
```
Processing file: samples/drums/kick.wav
samples/drums/kick.wav [CHANGED]: bit depth 24 -> 16, auto-mono

Found duplicate directories named 'drums' with 10 files (1.2MB):
Keeping oldest copy: samples/drums (created: Thu Mar 21 10:00:00 2024)
Moving duplicate: samples/backup/drums (created: Thu Mar 21 11:30:00 2024)

Found similar files: 'snare.wav' (250KB)
Similarity scores:
  snare_old.wav: 92.5% similar
  snare_copy.wav: 95.8% similar
Keeping oldest copy: samples/snare.wav
Moving similar files to backup...
```

## Directory Structure
```
samples/                  # Original directory
  drums/
    kick.wav
    snare.wav
_backup/                 # Backup directory
  samples/               # Original structure preserved
    drums/
      kick.wav.old      # Original files
      kick.wav.old.png  # Spectrograms
      kick.wav.new.png
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
