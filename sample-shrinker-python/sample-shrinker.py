import argparse
import concurrent.futures
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import filecmp
import ssdeep  # Add to imports

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


def get_file_hash(file_path, fuzzy=False, chunk_size=1024*1024):
    """Calculate file hash using either SHA-256 or fuzzy hashing."""
    if fuzzy:
        try:
            # Generate fuzzy hash for the file
            return ssdeep.hash_from_file(str(file_path))
        except Exception as e:
            print(f"Error generating fuzzy hash for {file_path}: {e}")
            return None
    else:
        # Standard SHA-256 hash with quick check
        sha256_hash = hashlib.sha256()
        file_size = os.path.getsize(file_path)
        
        with open(file_path, "rb") as f:
            # Read first chunk
            first_chunk = f.read(chunk_size)
            sha256_hash.update(first_chunk)
            
            # If file is large enough, read last chunk
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                last_chunk = f.read(chunk_size)
                sha256_hash.update(last_chunk)
                
            return sha256_hash.hexdigest()

def is_audio_file(file_path):
    """Check if file is an audio file we want to process."""
    return file_path.lower().endswith(('.wav', '.mp3'))

def find_duplicate_files(paths, fuzzy_threshold=90):
    """Find duplicate files using a multi-stage approach with optional fuzzy matching."""
    # Stage 1: Group by size (fast)
    size_groups = defaultdict(list)
    
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and is_audio_file(str(file_path)):
                    size = file_path.stat().st_size
                    size_groups[size].append(file_path)
    
    # Stage 2: For same-size files, group by quick hash
    hash_groups = defaultdict(list)
    fuzzy_groups = []  # Store groups of similar files
    
    for size, file_paths in size_groups.items():
        if len(file_paths) > 1:  # Only process groups with potential duplicates
            # First, try exact matches
            for file_path in file_paths:
                try:
                    file_hash = get_file_hash(file_path, fuzzy=False)
                    hash_groups[file_hash].append(file_path)
                except Exception as e:
                    print(f"Error hashing file {file_path}: {e}")
            
            # Then, try fuzzy matching for files that weren't exact matches
            unmatched_files = [f for f in file_paths if not any(f in group for group in hash_groups.values() if len(group) > 1)]
            if len(unmatched_files) > 1:
                fuzzy_matches = defaultdict(list)
                for file_path in unmatched_files:
                    fuzzy_hash = get_file_hash(file_path, fuzzy=True)
                    if fuzzy_hash:
                        fuzzy_matches[file_path] = fuzzy_hash
                
                # Compare fuzzy hashes
                matched = set()
                for file1, hash1 in fuzzy_matches.items():
                    if file1 in matched:
                        continue
                    similar_files = [file1]
                    for file2, hash2 in fuzzy_matches.items():
                        if file2 != file1 and file2 not in matched:
                            similarity = ssdeep.compare(hash1, hash2)
                            if similarity >= fuzzy_threshold:
                                similar_files.append(file2)
                                matched.add(file2)
                    if len(similar_files) > 1:
                        fuzzy_groups.append(similar_files)
                        matched.add(file1)
    
    # Combine exact and fuzzy matches
    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    duplicates.extend(fuzzy_groups)
    
    return duplicates, fuzzy_groups

def process_duplicate_files(duplicates, fuzzy_groups, args):
    """Process duplicate files with enhanced reporting."""
    for group in duplicates:
        is_fuzzy = group in fuzzy_groups
        match_type = "similar" if is_fuzzy else "identical"
        
        # Get file size for reporting
        file_size = group[0].stat().st_size
        print(f"\nFound {match_type} files: '{group[0].name}' ({file_size} bytes)")
        
        if is_fuzzy:
            # For fuzzy matches, show similarity percentages
            base_hash = get_file_hash(group[0], fuzzy=True)
            print("Similarity scores:")
            for file in group[1:]:
                file_hash = get_file_hash(file, fuzzy=True)
                similarity = ssdeep.compare(base_hash, file_hash)
                print(f"  {file.name}: {similarity}% similar")
        
        # Sort files by creation time
        files_with_time = [(f, f.stat().st_ctime) for f in group]
        files_with_time.sort(key=lambda x: x[1])
        
        # Keep the oldest file
        original_file = files_with_time[0][0]
        print(f"Keeping oldest copy: {original_file} (created: {time.ctime(files_with_time[0][1])})")
        
        # Process newer copies
        for file_path, ctime in files_with_time[1:]:
            print(f"Moving {match_type} file: {file_path} (created: {time.ctime(ctime)})")
            if not args.dry_run:
                try:
                    # Create backup path maintaining directory structure
                    rel_path = file_path.relative_to(file_path.parent.parent)
                    backup_path = Path(args.backup_dir) / rel_path
                    
                    # Ensure backup directory exists
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move the file
                    shutil.move(str(file_path), str(backup_path))
                except Exception as e:
                    print(f"Error moving file {file_path}: {e}")


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


def get_interactive_config():
    """Get configuration through interactive questionary prompts."""
    
    # First, get the action type
    action = questionary.select(
        "What would you like to do?",
        choices=[
            "Shrink samples (convert audio files)",
            "Remove duplicate directories",
            "Exit",
        ],
    ).ask()

    if action == "Exit":
        return None, None

    # Get the directory/files to process
    paths = questionary.path(
        "Select directory or file to process:",
        only_directories=False,
        multiple=True
    ).ask()

    if not paths:
        return None, None

    # Create a namespace object to match argparse structure
    args = argparse.Namespace()
    args.files = paths.split(",") if isinstance(paths, str) else paths
    
    # Set defaults
    args.backup_dir = "_backup"
    args.dry_run = False
    args.skip_spectrograms = False
    args.jobs = 1
    args.verbose = False
    args.ext = "wav,mp3"

    if action == "Remove duplicate directories":
        # For duplicate removal, we only need a few additional options
        args.dry_run = questionary.confirm(
            "Would you like to do a dry run first (preview without making changes)?",
            default=True
        ).ask()
        
        return "duplicates", args

    # For sample shrinking, get all the conversion options
    args.bitdepth = questionary.select(
        "Select target bit depth:",
        choices=["8", "16", "24"],
        default="16"
    ).ask()
    args.bitdepth = int(args.bitdepth)

    args.channels = questionary.select(
        "Select target channels:",
        choices=[
            "1 (mono)",
            "2 (stereo)"
        ],
        default="2 (stereo)"
    ).ask()
    args.channels = 1 if "1" in args.channels else 2

    args.samplerate = questionary.select(
        "Select target sample rate:",
        choices=["22050", "44100", "48000"],
        default="44100"
    ).ask()
    args.samplerate = int(args.samplerate)

    # Advanced options in a checkbox group
    advanced_options = questionary.checkbox(
        "Select additional options:",
        choices=[
            "Auto-convert stereo to mono when possible",
            "Pre-normalize before conversion",
            "Skip generating spectrograms",
            "Preview changes (dry run)",
            "Process files in parallel"
        ]
    ).ask()

    args.auto_mono = "Auto-convert stereo to mono when possible" in advanced_options
    args.pre_normalize = "Pre-normalize before conversion" in advanced_options
    args.skip_spectrograms = "Skip generating spectrograms" in advanced_options
    args.dry_run = "Preview changes (dry run)" in advanced_options
    
    if "Process files in parallel" in advanced_options:
        args.jobs = questionary.select(
            "How many parallel jobs?",
            choices=["2", "4", "8", "16"],
            default="4"
        ).ask()
        args.jobs = int(args.jobs)

    if args.auto_mono:
        args.auto_mono_threshold = float(
            questionary.text(
                "Auto-mono threshold in dB (default: -95.5):",
                default="-95.5"
            ).ask()
        )

    return "shrink", args


def process_duplicates(args):
    """Process both directory and file level duplicates with safety checks."""
    print("\nPhase 1: Searching for duplicate directories...")
    dir_duplicates = find_duplicate_directories(args.files)
    
    if dir_duplicates:
        print(f"\nFound {sum(len(v) - 1 for v in dir_duplicates.values())} duplicate directories")
        
        # Safety check: Verify directory contents match exactly
        verified_duplicates = {}
        for key, paths in dir_duplicates.items():
            dir_name, file_count, total_size = key
            
            # Get file listing for each directory
            dir_contents = defaultdict(list)
            for path in paths:
                files = sorted(f.relative_to(path) for f in path.rglob("*") if f.is_file())
                content_hash = hashlib.sha256(str(files).encode()).hexdigest()
                dir_contents[content_hash].append(path)
            
            # Only keep directories with exactly matching contents
            for content_hash, matching_paths in dir_contents.items():
                if len(matching_paths) > 1:
                    verified_duplicates[key + (content_hash,)] = matching_paths
        
        if args.dry_run:
            print("\nDRY RUN - No directories will be moved")
        process_duplicate_directories(verified_duplicates, args)
    else:
        print("No duplicate directories found.")
    
    print("\nPhase 2: Searching for duplicate files...")
    file_duplicates, fuzzy_groups = find_duplicate_files(args.files)
    
    if file_duplicates:
        total_duplicates = sum(len(group) - 1 for group in file_duplicates)
        print(f"\nFound {total_duplicates} duplicate files")
        
        # Additional safety checks for file processing
        safe_duplicates = []
        for group in file_duplicates:
            # Verify files are not symbolic links
            real_files = [f for f in group if not f.is_symlink()]
            
            # Check if files are in use (on Windows) or locked
            available_files = []
            for file in real_files:
                try:
                    with open(file, 'rb') as f:
                        # Try to get a shared lock
                        pass
                    available_files.append(file)
                except (IOError, OSError):
                    print(f"Warning: File {file} appears to be in use, skipping")
            
            if len(available_files) > 1:
                safe_duplicates.append(available_files)
        
        if args.dry_run:
            print("\nDRY RUN - No files will be moved")
        process_duplicate_files(safe_duplicates, fuzzy_groups, args)
    else:
        print("No duplicate files found.")
    
    print("\nDuplicate removal complete!")


def main():
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        args = parse_args()
        action = "shrink"  # Default to shrink mode for command line
    else:
        # Use interactive mode
        action, args = get_interactive_config()
        
    if not args:
        return

    if action == "duplicates":
        process_duplicates(args)
    else:  # Shrink samples
        # Delete all '._' files before processing anything
        for path in args.files:
            if os.path.isdir(path):
                delete_resource_forks(path)

        # Collect the files to process
        file_list = collect_files(args)

        if args.dry_run:
            list_files(args, file_list)
            for file in file_list:
                process_audio(file, args, dry_run=True)
        else:
            run_in_parallel(file_list, args)


if __name__ == "__main__":
    main()