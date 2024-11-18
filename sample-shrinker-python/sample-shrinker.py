import argparse
import concurrent.futures
import filecmp
import hashlib
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import questionary
import soundfile as sf
import scipy.signal
from scipy.io import wavfile
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Initialize console
console = Console()


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
        output_path = str(Path(file_path).with_suffix('.reencoded.wav'))
        # Use ffmpeg directly for more reliable conversion
        cmd = [
            'ffmpeg', '-y',
            '-i', str(file_path),
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]Successfully re-encoded: {output_path}[/green]")
            return output_path
        else:
            console.print(f"[red]FFmpeg error: {result.stderr}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error re-encoding {file_path}: {str(e)}[/red]")
        return None


def check_ffmpeg():
    """Check if ffmpeg is available and properly installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        console.print("[red]Error: ffmpeg is not installed or not found in PATH[/red]")
        console.print("Please install ffmpeg:")
        console.print("  MacOS: brew install ffmpeg")
        console.print("  Ubuntu/Debian: sudo apt install ffmpeg")
        console.print("  Windows: https://ffmpeg.org/download.html")
        return False


def process_audio(file_path, args, dry_run=False, task_id=None, progress=None):
    """Main function to process audio files based on arguments."""
    try:
        if progress:
            progress.update(task_id, description=f"Processing: {Path(file_path).name}")
        else:
            console.print(f"Processing file: [cyan]{file_path}[/cyan]")
        
        try:
            audio = AudioSegment.from_file(file_path)
        except (IndexError, OSError) as e:
            console.print(f"[red]Error loading {file_path}: {str(e)}[/red]")
            console.print("[yellow]Attempting to re-encode file...[/yellow]")
            reencoded_file = reencode_audio(file_path)
            if reencoded_file:
                try:
                    audio = AudioSegment.from_file(reencoded_file)
                except Exception as re_err:
                    console.print(f"[red]Failed to process re-encoded file: {str(re_err)}[/red]")
                    return
            else:
                return
            
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
            status = Text()
            status.append(f"{file_path} ", style="cyan")
            status.append("[CHANGED]: ", style="yellow")
            status.append(", ".join(change_reason), style="green")
            console.print(status)
            
            if not dry_run:
                # Backup handling
                if args.backup_dir != "-":
                    try:
                        # Convert the file path to a Path object
                        file_path_obj = Path(file_path).resolve()
                        # Get the absolute path to the backup directory
                        backup_dir = Path(args.backup_dir).resolve()
                        
                        # Create the relative path structure
                        rel_path = file_path_obj.relative_to(file_path_obj.parent)
                        backup_path = backup_dir / rel_path.parent.name / rel_path.name
                        
                        # Ensure the backup directory exists
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Add .old extension for the backup
                        backup_path = backup_path.with_suffix(backup_path.suffix + '.old')
                        
                        # Copy the original file with metadata preserved
                        console.print(f"[cyan]Backing up to: {backup_path}[/cyan]")
                        shutil.copy2(file_path, backup_path)
                        
                        # Generate spectrograms if enabled
                        if not args.skip_spectrograms:
                            generate_spectrogram(file_path, file_path, backup_path.parent)
                            
                    except Exception as e:
                        console.print(f"[red]Error creating backup: {str(e)}[/red]")
                        if args.verbose:
                            import traceback
                            console.print(traceback.format_exc())
                        return
                else:
                    console.print("[yellow]No backup created (backups disabled)[/yellow]")

                # Export the converted audio file
                try:
                    output_file = file_path
                    audio.export(output_file, format="wav")
                    console.print(f"[green]Converted file saved: {output_file}[/green]")
                except Exception as e:
                    console.print(f"[red]Error saving converted file: {str(e)}[/red]")
                    if args.verbose:
                        import traceback
                        console.print(traceback.format_exc())
        else:
            status = Text()
            status.append(f"{file_path} ", style="cyan")
            status.append("[UNCHANGED]", style="blue")
            console.print(status)

    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")
        if args.verbose:
            console.print(f"[yellow]Stack trace:[/yellow]")
            import traceback
            console.print(traceback.format_exc())


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
    valid_extensions = [ext.strip().lower() for ext in args.ext.split(",")]
    
    console.print("[cyan]Starting file collection...[/cyan]")
    
    for path in args.files:
        # Expand user and resolve path
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        path = Path(path).resolve()
        
        console.print(f"[cyan]Scanning path: {path}[/cyan]")
        
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(f".{ext}") for ext in valid_extensions) and not file.startswith("._"):
                        full_path = os.path.join(root, file)
                        file_list.append(full_path)
                        if args.verbose:
                            console.print(f"[dim]Found: {full_path}[/dim]")
        elif path.is_file():
            file_lower = str(path).lower()
            if any(file_lower.endswith(f".{ext}") for ext in valid_extensions) and not path.name.startswith("._"):
                file_list.append(str(path))
                if args.verbose:
                    console.print(f"[dim]Found: {path}[/dim]")
    
    console.print(f"[green]Found {len(file_list)} files to process[/green]")
    return file_list


def run_in_parallel(file_list, args):
    """Run the audio processing in parallel with progress bar."""
    if not file_list:
        console.print("[yellow]No files to process![/yellow]")
        return
        
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            total_files = len(file_list)
            console.print(f"[cyan]Starting processing of {total_files} files with {args.jobs} parallel jobs[/cyan]")
            
            task = progress.add_task("Processing files...", total=total_files)
            
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_audio, 
                        file, 
                        args,
                        task_id=task,
                        progress=progress
                    ): file for file in file_list
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    progress.advance(task)
                    try:
                        result = future.result()
                    except Exception as exc:
                        file = futures[future]
                        console.print(f"[red]File {file} generated an exception: {exc}[/red]")
                    
            console.print("[green]Processing complete![/green]")
                        
    except KeyboardInterrupt:
        console.print("[yellow]Received KeyboardInterrupt, attempting to cancel all threads...[/yellow]")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except Exception as e:
        console.print(f"[red]Error in parallel processing: {e}[/red]")
        raise


def get_file_hash(file_path, fuzzy=False, chunk_size=1024 * 1024):
    """Calculate file hash using either SHA-256 or audio fingerprinting."""
    if fuzzy:
        # Use our audio fingerprinting instead of ssdeep
        return get_audio_fingerprint(file_path)
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
    return file_path.lower().endswith((".wav", ".mp3"))


def get_audio_fingerprint(file_path):
    """Generate an audio fingerprint using cross-correlation."""
    try:
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        # Convert to mono for comparison
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Normalize
        samples = samples / np.max(np.abs(samples))
        
        # Get a signature using peaks in frequency domain
        freqs, times, spectrogram = scipy.signal.spectrogram(
            samples,
            audio.frame_rate,
            nperseg=1024,
            noverlap=512
        )
        
        # Get the strongest frequencies
        peaks = np.mean(spectrogram, axis=1)
        # Normalize the peaks
        peaks = peaks / np.max(peaks)
        
        return peaks
    except Exception as e:
        print(f"Error generating audio fingerprint for {file_path}: {e}")
        return None


def compare_audio_similarity(file1_fingerprint, file2_fingerprint):
    """Compare two audio fingerprints and return similarity score."""
    if file1_fingerprint is None or file2_fingerprint is None:
        return 0
    
    # Ensure same length for comparison
    min_len = min(len(file1_fingerprint), len(file2_fingerprint))
    f1 = file1_fingerprint[:min_len]
    f2 = file2_fingerprint[:min_len]
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(f1, f2)[0, 1]
    # Convert to percentage and handle NaN
    similarity = float(max(0, correlation) * 100)
    return similarity if not np.isnan(similarity) else 0


def find_duplicate_files(paths, args):
    """Find duplicate files using a multi-stage approach with audio fingerprinting."""
    print("Scanning for duplicate files...")
    size_groups = defaultdict(list)
    
    # First pass: group by size
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and is_audio_file(str(file_path)):
                    if args.verbose:
                        print(f"Scanning: {file_path}")
                    size = file_path.stat().st_size
                    size_groups[size].append(file_path)
    
    hash_groups = defaultdict(list)
    similar_groups = []
    
    # Second pass: check content
    for size, file_paths in size_groups.items():
        if len(file_paths) > 1:
            if args.verbose:
                print(f"\nChecking {len(file_paths)} files of size {size} bytes...")
            
            # First try exact matches
            for file_path in file_paths:
                try:
                    file_hash = get_file_hash(file_path, fuzzy=False)
                    if args.ignore_names:
                        hash_groups[file_hash].append(file_path)
                    else:
                        name_key = file_path.stem.lower()
                        hash_groups[(name_key, file_hash)].append(file_path)
                except Exception as e:
                    print(f"Error hashing file {file_path}: {e}")
            
            # Then check for similar audio content
            if args.use_fuzzy:
                unmatched = [f for f in file_paths 
                           if not any(f in g for g in hash_groups.values() if len(g) > 1)]
                
                if len(unmatched) > 1:
                    # Generate fingerprints for all unmatched files
                    fingerprints = {}
                    for file_path in unmatched:
                        fingerprint = get_audio_fingerprint(file_path)
                        if fingerprint is not None:
                            fingerprints[file_path] = fingerprint
                    
                    # Compare fingerprints
                    processed = set()
                    for file1 in fingerprints:
                        if file1 in processed:
                            continue
                        
                        similar_files = [file1]
                        for file2 in fingerprints:
                            if file2 != file1 and file2 not in processed:
                                similarity = compare_audio_similarity(
                                    fingerprints[file1],
                                    fingerprints[file2]
                                )
                                if similarity >= args.fuzzy_threshold:
                                    similar_files.append(file2)
                                    processed.add(file2)
                        
                        if len(similar_files) > 1:
                            similar_groups.append(similar_files)
                            processed.add(file1)
    
    # Combine results
    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    if args.use_fuzzy:
        duplicates.extend(similar_groups)
    
    return duplicates, similar_groups


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
            base_fingerprint = get_audio_fingerprint(group[0])
            print("Similarity scores:")
            for file in group[1:]:
                file_fingerprint = get_audio_fingerprint(file)
                similarity = compare_audio_similarity(base_fingerprint, file_fingerprint)
                print(f"  {file.name}: {similarity:.1f}% similar")

        # Sort files by creation time
        files_with_time = [(f, f.stat().st_ctime) for f in group]
        files_with_time.sort(key=lambda x: x[1])

        # Keep the oldest file
        original_file = files_with_time[0][0]
        print(
            f"Keeping oldest copy: {original_file} (created: {time.ctime(files_with_time[0][1])})"
        )

        # Process newer copies
        for file_path, ctime in files_with_time[1:]:
            print(
                f"Moving {match_type} file: {file_path} (created: {time.ctime(ctime)})"
            )
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


def load_saved_config():
    """Load previously saved configuration."""
    config_path = Path.home() / '.sample-shrinker.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                console.print("[dim]Loaded saved configuration[/dim]")
                return config
        except Exception as e:
            console.print(f"[yellow]Error loading saved config: {e}[/yellow]")
    return {}

def save_config(args, action):
    """Save current configuration."""
    config_path = Path.home() / '.sample-shrinker.json'
    try:
        # Convert namespace to dict and handle Path objects
        config = {
            'last_action': action,
            'files': [str(p) for p in args.files],
            'backup_dir': args.backup_dir,
            'bitdepth': args.bitdepth,
            'channels': args.channels,
            'samplerate': args.samplerate,
            'min_samplerate': args.min_samplerate,
            'min_bitdepth': args.min_bitdepth,
            'auto_mono': args.auto_mono,
            'auto_mono_threshold': args.auto_mono_threshold,
            'skip_spectrograms': args.skip_spectrograms,
            'pre_normalize': args.pre_normalize,
            'jobs': args.jobs,
            # Duplicate removal specific settings
            'use_fuzzy': getattr(args, 'use_fuzzy', False),
            'ignore_names': getattr(args, 'ignore_names', False),
            'fuzzy_threshold': getattr(args, 'fuzzy_threshold', 90),
            'fuzzy_options': getattr(args, 'fuzzy_options', []),
            'advanced_options': getattr(args, 'advanced_options', []),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            console.print("[dim]Saved configuration for next time[/dim]")
    except Exception as e:
        console.print(f"[yellow]Error saving config: {e}[/yellow]")

def get_interactive_config():
    """Get configuration through interactive questionary prompts."""
    # Load saved configuration
    saved_config = load_saved_config()
    
    # First, get the action type
    action = questionary.select(
        "What would you like to do?",
        choices=[
            "Shrink samples (convert audio files)",
            "Remove duplicate directories",
            "Exit",
        ],
        default=saved_config.get('last_action', "Shrink samples (convert audio files)")
    ).ask()

    if action == "Exit":
        return None, None

    # Get the directory/files to process
    paths = []
    last_paths = saved_config.get('files', [])
    
    if last_paths:
        use_last = questionary.confirm(
            f"Use last paths?\n" + "\n".join(last_paths),
            default=True
        ).ask()
        if use_last:
            paths = last_paths

    # ... rest of path collection code ...

    # Create a namespace object with saved defaults
    args = argparse.Namespace()
    args.files = paths
    args.backup_dir = saved_config.get('backup_dir', "_backup")
    args.bitdepth = saved_config.get('bitdepth', 16)
    args.channels = saved_config.get('channels', 2)
    args.samplerate = saved_config.get('samplerate', 44100)
    args.min_samplerate = saved_config.get('min_samplerate', None)
    args.min_bitdepth = saved_config.get('min_bitdepth', None)
    args.auto_mono = saved_config.get('auto_mono', False)
    args.auto_mono_threshold = saved_config.get('auto_mono_threshold', -95.5)
    args.skip_spectrograms = saved_config.get('skip_spectrograms', False)
    args.pre_normalize = saved_config.get('pre_normalize', False)
    args.jobs = saved_config.get('jobs', 1)

    if action == "Remove duplicate directories":
        # Use saved defaults for duplicate options
        saved_duplicate_options = []
        if saved_config.get('use_fuzzy', False):
            saved_duplicate_options.append("Use fuzzy matching for similar files")
        if saved_config.get('ignore_names', False):
            saved_duplicate_options.append("Ignore filenames (match by content only)")
        
        duplicate_options = questionary.checkbox(
            "Select duplicate removal options:",
            choices=[
                "Use fuzzy matching for similar files",
                "Ignore filenames (match by content only)",
                "Preview changes (dry run)",
                "Show detailed progress",
            ],
            default=saved_duplicate_options
        ).ask()

        # ... rest of duplicate removal configuration ...

    else:  # Sample shrinking
        # Use saved defaults for advanced options
        saved_advanced = saved_config.get('advanced_options', [])
        advanced_options = questionary.checkbox(
            "Select additional options:",
            choices=[
                "Auto-convert stereo to mono when possible",
                "Pre-normalize before conversion",
                "Skip generating spectrograms",
                "Preview changes (dry run)",
                "Process files in parallel",
                "Set minimum sample rate",
                "Set minimum bit depth",
                "Convert in place (no backups)",
            ],
            default=saved_advanced
        ).ask()
        
        # Store selected options for next time
        args.advanced_options = advanced_options

    # Save the final configuration
    save_config(args, action)

    return "duplicates" if "Remove" in action else "shrink", args


def process_duplicates(args):
    """Process both directory and file level duplicates with visual feedback."""
    with console.status("[bold green]Phase 1: Searching for duplicate directories...") as status:
        dir_duplicates = find_duplicate_directories(args.files)

    if dir_duplicates:
        count = sum(len(v) - 1 for v in dir_duplicates.values())
        console.print(Panel(f"Found [cyan]{count}[/cyan] duplicate directories", 
                          title="Directory Scan Complete"))
        
        if args.dry_run:
            console.print("[yellow]DRY RUN - No directories will be moved[/yellow]")
        process_duplicate_directories(verified_duplicates, args)
    else:
        console.print("[blue]No duplicate directories found.[/blue]")

    with console.status("[bold green]Phase 2: Searching for duplicate files...") as status:
        file_duplicates, fuzzy_groups = find_duplicate_files(args.files, args)

    if file_duplicates:
        total_duplicates = sum(len(group) - 1 for group in file_duplicates)
        console.print(Panel(
            f"Found [cyan]{total_duplicates}[/cyan] duplicate files\n"
            f"Including [cyan]{len(fuzzy_groups)}[/cyan] groups of similar files",
            title="File Scan Complete"
        ))
        
        # Additional safety checks for file processing
        safe_duplicates = []
        for group in file_duplicates:
            # Verify files are not symbolic links
            real_files = [f for f in group if not f.is_symlink()]

            # Check if files are in use (on Windows) or locked
            available_files = []
            for file in real_files:
                try:
                    with open(file, "rb") as f:
                        # Try to get a shared lock
                        pass
                    available_files.append(file)
                except (IOError, OSError):
                    print(f"Warning: File {file} appears to be in use, skipping")

            if len(available_files) > 1:
                safe_duplicates.append(available_files)

        if args.dry_run:
            console.print("[yellow]DRY RUN - No files will be moved[/yellow]")
        process_duplicate_files(safe_duplicates, fuzzy_groups, args)
    else:
        console.print("[blue]No duplicate files found.[/blue]")

    console.print("[green]Duplicate removal complete![/green]")


def main():
    # Check for ffmpeg first
    if not check_ffmpeg():
        return

    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        args = parse_args()
        action = "shrink"  # Default to shrink mode for command line
    else:
        # Use interactive mode with saved configuration
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
