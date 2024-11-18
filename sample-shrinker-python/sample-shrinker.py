import argparse
import concurrent.futures
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import questionary
import soundfile as sf
from pydub import AudioSegment


def usage_intro():
    return """
Conditionally batch-converts audio samples into minimal .wav files.

Each DIRECTORY is recursively searched for audio files to process, based on their extension (configured with -x). Any FILE specified directly will be processed (regardless of its extension).

If a sample does not already meet the target BIT_DEPTH or CHANNELS, it will be converted in place and the original will be backed up to a parallel directory structure.

Upon conversion, spectrogram .png files are generated alongside the backed-up original file to compare the original vs new audio files (disable with -S).

Examples:
    Recursively convert samples under 'sample_dir/' using the default settings:
        $ sample-shrinker.py sample_dir/
    Convert samples down to 8-bit, mono:
        $ sample-shrinker.py -c 1 -b 8 sample_dir/
    Auto-convert stereo samples to mono:
        $ sample-shrinker.py -a sample_dir/
    """


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch convert audio files.")
    parser.add_argument("files", nargs="+", help="Files or directories to process")
    parser.add_argument(
        "-b", "--bitdepth", type=int, default=16, help="Target bit depth (8, 16, 24)"
    )
    parser.add_argument(
        "-B", "--min_bitdepth", type=int, help="Minimum bit depth (8, 16, 24)"
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=2,
        help="Target number of channels (1=mono, 2=stereo)",
    )
    parser.add_argument(
        "-r", "--samplerate", type=int, default=44100, help="Target sample rate"
    )
    parser.add_argument("-R", "--min_samplerate", type=int, help="Minimum sample rate")
    parser.add_argument(
        "-x",
        "--ext",
        default="wav,mp3",
        help="Comma-separated file extensions to search for (default: wav,mp3)",
    )
    parser.add_argument(
        "-a",
        "--auto_mono",
        action="store_true",
        help="Automatically convert stereo samples to mono",
    )
    parser.add_argument(
        "-A",
        "--auto_mono_threshold",
        type=float,
        default=-95.5,
        help="Auto-mono threshold dB",
    )
    parser.add_argument(
        "-S",
        "--skip_spectrograms",
        action="store_true",
        help="Skip generating spectrogram files",
    )
    parser.add_argument(
        "-d",
        "--backup_dir",
        default="_backup",
        help="Directory to store backups (default: _backup)",
    )
    parser.add_argument(
        "-p",
        "--pre_normalize",
        action="store_true",
        help="Pre-normalize before downsampling bit-depth",
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List files without converting"
    )
    parser.add_argument(
        "-n", "--dry_run", action="store_true", help="Log actions without converting"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )

    return parser.parse_args()


def delete_resource_forks(directory):
    """Recursively find and delete all '._' resource fork files in the directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                file_path = os.path.join(root, file)
                print(f"Deleting resource fork file: {file_path}")
                os.remove(file_path)


def reencode_audio(file_path):
    """Re-encode audio file to PCM 16-bit if it has a different encoding."""
    try:
        with sf.SoundFile(file_path) as f:
            print(
                f"Audio encoding: {f.format}, subtype: {f.subtype}, channels: {f.channels}"
            )
            if f.subtype != "PCM_16":
                # If the file is not PCM 16, re-save it as PCM_16
                data, samplerate = sf.read(file_path)
                temp_output = file_path.replace(
                    os.path.splitext(file_path)[1], "_reencoded.wav"
                )
                sf.write(temp_output, data, samplerate, subtype="PCM_16")
                print(f"File re-encoded to PCM_16: {file_path} -> {temp_output}")
                return temp_output
    except Exception as e:
        print(f"Error re-encoding {file_path}: {e}")
    return None


def process_audio(file_path, args, dry_run=False):
    """Main function to process audio files based on arguments."""
    try:
        print(f"Processing file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        modified = False
        change_reason = []

        # Check if we need to convert the channels
        if audio.channels > args.channels:
            change_reason.append("channels")
            if not dry_run:
                audio = audio.set_channels(args.channels)
            modified = True

        # Auto-mono logic: convert stereo to mono if it is effectively mono
        if args.auto_mono and audio.channels == 2:
            mono_candidate = check_effectively_mono(audio, args.auto_mono_threshold)
            if mono_candidate:
                change_reason.append("auto-mono")
                if not dry_run:
                    audio = audio.set_channels(1)
                modified = True

        # Pre-normalize before downsampling bit depth if necessary
        if args.pre_normalize:
            change_reason.append("pre-normalize")
            if not dry_run:
                audio = audio.apply_gain(-audio.max_dBFS)
            modified = True

        # Check if we need to convert the bit depth
        if audio.sample_width * 8 > args.bitdepth:
            change_reason.append(
                f"bit depth {audio.sample_width * 8} -> {args.bitdepth}"
            )
            if not dry_run:
                audio = audio.set_sample_width(args.bitdepth // 8)
            modified = True

        # Sample rate conversion logic: Downsample only
        if audio.frame_rate > args.samplerate:
            change_reason.append(f"sample rate {audio.frame_rate} -> {args.samplerate}")
            if not dry_run:
                audio = audio.set_frame_rate(args.samplerate)
            modified = True
        elif args.min_samplerate and audio.frame_rate < args.min_samplerate:
            # Only upsample if the user specifies a minimum sample rate
            change_reason.append(
                f"sample rate {audio.frame_rate} -> {args.min_samplerate}"
            )
            if not dry_run:
                audio = audio.set_frame_rate(args.min_samplerate)
            modified = True

        if modified:
            print(f"{file_path} [CHANGED]: {', '.join(change_reason)}")
            if not dry_run:
                # Backup the original file if required
                if args.backup_dir != "-":
                    # Get the relative path from the current working directory
                    rel_path = os.path.relpath(file_path)
                    # Create the backup path maintaining the directory structure
                    backup_path = os.path.join(args.backup_dir, rel_path)
                    # Ensure the directory structure exists
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(file_path, backup_path)  # copy2 preserves metadata

                # Export the converted audio file
                output_file = file_path.replace(os.path.splitext(file_path)[1], ".wav")
                audio.export(output_file, format="wav")

                # Generate spectrogram if enabled
                if not args.skip_spectrograms:
                    generate_spectrogram(
                        file_path, output_file, os.path.dirname(backup_path)
                    )
        else:
            print(f"{file_path} [UNCHANGED]")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

        # Try re-encoding the file if ffmpeg failed
        reencoded_file = reencode_audio(file_path)
        if reencoded_file:
            try:
                # Retry the process with the re-encoded file
                process_audio(reencoded_file, args, dry_run)
            except Exception as retry_error:
                print(
                    f"Failed to process the re-encoded file {reencoded_file}: {retry_error}"
                )


def check_effectively_mono(audio, threshold_dB):
    """Check if a stereo file is effectively mono."""
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1].invert_phase()

    difference = left_channel.overlay(right_channel)
    peak_diff_db = difference.max_dBFS
    return peak_diff_db < threshold_dB


def generate_spectrogram(original_file, new_file, backup_dir):
    """Generate and save spectrograms for the original and new files."""
    y_old, sr_old = librosa.load(original_file, sr=None)
    y_new, sr_new = librosa.load(new_file, sr=None)

    # Spectrogram for original file
    plt.figure(figsize=(10, 4))
    D_old = librosa.amplitude_to_db(np.abs(librosa.stft(y_old)), ref=np.max)
    librosa.display.specshow(D_old, sr=sr_old, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram of {os.path.basename(original_file)}")
    old_spectrogram_path = os.path.join(
        backup_dir, os.path.basename(original_file) + ".old.png"
    )
    os.makedirs(backup_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(old_spectrogram_path)
    plt.close()

    # Spectrogram for new file
    plt.figure(figsize=(10, 4))
    D_new = librosa.amplitude_to_db(np.abs(librosa.stft(y_new)), ref=np.max)
    librosa.display.specshow(D_new, sr=sr_new, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram of {os.path.basename(new_file)}")
    new_spectrogram_path = os.path.join(
        backup_dir, os.path.basename(new_file) + ".new.png"
    )
    plt.savefig(new_spectrogram_path)
    plt.close()


def list_files(args, file_list):
    """Prints file summary and actions without performing them."""
    for file_path in file_list:
        print(f"Previewing: {file_path}")


def collect_files(args):
    """Collect all wav and mp3 files from provided directories and files."""
    file_list = []
    # Split extensions string into a list and clean up whitespace
    valid_extensions = [ext.strip().lower() for ext in args.ext.split(",")]

    for path in args.files:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_lower = file.lower()
                    # Check if file ends with any of the valid extensions
                    if any(
                        file_lower.endswith(f".{ext}") for ext in valid_extensions
                    ) and not file.startswith("._"):
                        file_list.append(os.path.join(root, file))
        elif os.path.isfile(path):
            file_lower = path.lower()
            if any(
                file_lower.endswith(f".{ext}") for ext in valid_extensions
            ) and not os.path.basename(path).startswith("._"):
                file_list.append(path)
    return file_list


def run_in_parallel(file_list, args):
    """Run the audio processing in parallel."""
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_audio, file, args): file for file in file_list
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = (
                        future.result()
                    )  # Get the result of the future (processed file)
                except Exception as exc:
                    file = futures[future]
                    print(f"File {file} generated an exception: {exc}")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, attempting to cancel all threads...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def find_duplicate_directories(paths):
    """Find directories with matching names and file counts."""
    dir_map = defaultdict(list)

    for path in paths:
        path = Path(path)
        if path.is_dir():
            for dir_path in path.rglob("*"):
                if dir_path.is_dir():
                    # Get directory name, file count, and total size
                    dir_name = dir_path.name.lower()  # Case-insensitive comparison
                    files = list(dir_path.glob("*"))
                    file_count = len([f for f in files if f.is_file()])
                    total_size = sum(f.stat().st_size for f in files if f.is_file())

                    dir_map[(dir_name, file_count, total_size)].append(dir_path)

    # Return only directories that have duplicates
    return {k: v for k, v in dir_map.items() if len(v) > 1}


def process_duplicate_directories(duplicates, args):
    """Process duplicate directories, keeping the oldest copy."""
    for (dir_name, file_count, total_size), paths in duplicates.items():
        print(
            f"\nFound duplicate directories named '{dir_name}' with {file_count} files ({total_size} bytes):"
        )

        # Sort paths by creation time
        paths_with_time = [(p, p.stat().st_ctime) for p in paths]
        paths_with_time.sort(key=lambda x: x[1])

        # Keep the oldest directory
        original_dir = paths_with_time[0][0]
        print(
            f"Keeping oldest copy: {original_dir} (created: {time.ctime(paths_with_time[0][1])})"
        )

        # Process newer copies
        for dir_path, ctime in paths_with_time[1:]:
            print(f"Moving duplicate: {dir_path} (created: {time.ctime(ctime)})")
            if not args.dry_run:
                # Create backup path
                rel_path = dir_path.relative_to(dir_path.parent.parent)
                backup_path = Path(args.backup_dir) / rel_path

                # Ensure backup directory exists
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.move(str(dir_path), str(backup_path))
                except Exception as e:
                    print(f"Error moving directory {dir_path}: {e}")


def main():
    args = parse_args()

    # Ensure that at least one file or directory is provided
    if not args.files:
        print(usage_intro())
        return

    # Ask user what they want to do
    action = questionary.select(
        "What would you like to do?",
        choices=[
            "Shrink samples (convert audio files)",
            "Remove duplicate directories",
            "Exit",
        ],
    ).ask()

    if action == "Exit":
        return
    elif action == "Remove duplicate directories":
        # Find and process duplicate directories
        print("\nSearching for duplicate directories...")
        duplicates = find_duplicate_directories(args.files)

        if not duplicates:
            print("No duplicate directories found.")
            return

        if args.dry_run:
            print("\nDRY RUN - No files will be moved")

        process_duplicate_directories(duplicates, args)
        print("\nDuplicate removal complete!")

    else:  # Shrink samples
        # Delete all '._' files before processing anything
        for path in args.files:
            if os.path.isdir(path):
                delete_resource_forks(path)

        # Collect the files to process
        file_list = collect_files(args)

        if args.dry_run or args.list:
            list_files(args, file_list)
            for file in file_list:
                process_audio(file, args, dry_run=True)
        else:
            run_in_parallel(file_list, args)


if __name__ == "__main__":
    main()
