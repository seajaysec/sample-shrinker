import argparse
import concurrent.futures
import filecmp
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import questionary
import scipy.signal
import soundfile as sf
from pydub import AudioSegment
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text
from scipy.io import wavfile

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
        output_path = str(Path(file_path).with_suffix(".reencoded.wav"))

        # First try with ADPCM decoder explicitly
        cmd = [
            "ffmpeg",
            "-y",
            "-c:a",
            "adpcm_ms",  # Try ADPCM first
            "-i",
            str(file_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "wav",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(
                f"[green]Successfully re-encoded with ADPCM decoder: {output_path}[/green]"
            )
            return output_path

        # If ADPCM fails, try with default decoder
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(file_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "wav",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]Successfully re-encoded: {output_path}[/green]")
            return output_path

        # If both attempts fail, try with more aggressive options
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(file_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-af",
            "aresample=resampler=soxr",  # Use high quality resampler
            "-strict",
            "experimental",
            "-f",
            "wav",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(
                f"[green]Successfully re-encoded with resampling: {output_path}[/green]"
            )
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
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        console.print("[red]Error: ffmpeg is not installed or not found in PATH[/red]")
        console.print("Please install ffmpeg:")
        console.print("  MacOS: brew install ffmpeg")
        console.print("  Ubuntu/Debian: sudo apt install ffmpeg")
        console.print("  Windows: https://ffmpeg.org/download.html")
        return False


def get_audio_properties(file_path):
    """Get audio file properties using pydub."""
    try:
        # First try direct loading
        try:
            audio = AudioSegment.from_file(file_path)
            # Fix for incorrect bit depth detection
            actual_bit_depth = audio.sample_width * 8
            # Some 24-bit files might be reported as 32-bit
            if actual_bit_depth == 32:
                # Check if it's actually 24-bit
                max_value = max(
                    abs(min(audio.get_array_of_samples())),
                    abs(max(audio.get_array_of_samples())),
                )
                if max_value <= 0x7FFFFF:  # Max value for 24-bit
                    actual_bit_depth = 24

            return {
                "bit_depth": actual_bit_depth,
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "duration": len(audio),
            }
        except Exception as e:
            # If direct loading fails, try re-encoding first
            reencoded = reencode_audio(file_path)
            if reencoded:
                audio = AudioSegment.from_file(reencoded)
                return {
                    "bit_depth": audio.sample_width * 8,
                    "channels": audio.channels,
                    "sample_rate": audio.frame_rate,
                    "duration": len(audio),
                }
            else:
                raise e
    except Exception as e:
        console.print(
            f"[yellow]Error reading audio properties from {file_path}: {str(e)}[/yellow]"
        )
        return None


def needs_conversion(file_path, args):
    """Check if file needs conversion based on its properties."""
    props = get_audio_properties(file_path)
    if not props:
        return (True, ["unable to read properties"])  # Return tuple with reason

    needs_conv = False
    reasons = []

    if props["bit_depth"] > args.bitdepth:
        needs_conv = True
        reasons.append(f"bit depth {props['bit_depth']} -> {args.bitdepth}")

    if props["channels"] > args.channels:
        needs_conv = True
        reasons.append(f"channels {props['channels']} -> {args.channels}")

    if props["sample_rate"] > args.samplerate:
        needs_conv = True
        reasons.append(f"sample rate {props['sample_rate']} -> {args.samplerate}")

    if args.min_samplerate and props["sample_rate"] < args.min_samplerate:
        needs_conv = True
        reasons.append(f"sample rate {props['sample_rate']} -> {args.min_samplerate}")

    return (needs_conv, reasons)  # Always return a tuple


def process_audio(file_path, args, dry_run=False, task_id=None, progress=None):
    """Main function to process audio files based on arguments."""
    try:
        if progress:
            progress.update(task_id, description=f"Processing: {Path(file_path).name}")
        else:
            console.print(f"Processing file: [cyan]{file_path}[/cyan]")

        # Load the audio file first
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            console.print(f"[yellow]Error loading {file_path}: {str(e)}[/yellow]")
            console.print("[yellow]Attempting to re-encode file...[/yellow]")
            reencoded_file = reencode_audio(file_path)
            if reencoded_file:
                try:
                    audio = AudioSegment.from_file(reencoded_file)
                except Exception as re_err:
                    console.print(
                        f"[red]Failed to process re-encoded file: {str(re_err)}[/red]"
                    )
                    return
            else:
                return

        # Check if file needs processing
        needs_conv, reasons = needs_conversion(file_path, args)
        if not needs_conv:
            console.print(
                f"[blue]Skipping {file_path} (already meets requirements)[/blue]"
            )
            return

        modified = False
        change_reason = reasons.copy()  # Use the reasons from needs_conversion

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
                        # Convert paths to Path objects
                        file_path_obj = Path(file_path).resolve()
                        backup_base = Path(args.backup_dir).resolve()

                        # Get the relative structure from the file path
                        path_parts = file_path_obj.parts[-3:]  # Adjust number as needed
                        backup_path = backup_base.joinpath(*path_parts)

                        # Check if backup already exists
                        if backup_path.exists():
                            console.print(
                                f"[blue]Backup already exists: {backup_path}[/blue]"
                            )
                        else:
                            # Ensure the backup directory exists
                            backup_path.parent.mkdir(parents=True, exist_ok=True)

                            # Copy the original file with metadata preserved
                            console.print(f"[cyan]Backing up to: {backup_path}[/cyan]")
                            shutil.copy2(file_path, backup_path)

                            # Generate spectrograms if enabled
                            if not args.skip_spectrograms:
                                try:
                                    generate_spectrogram(
                                        file_path,
                                        file_path,
                                        backup_path.parent,
                                        verbose=args.verbose,
                                    )
                                except Exception as spec_err:
                                    console.print(
                                        f"[yellow]Warning: Could not generate spectrograms: {spec_err}[/yellow]"
                                    )
                                    if args.verbose:
                                        import traceback

                                        console.print(traceback.format_exc())

                    except Exception as e:
                        console.print(f"[red]Error creating backup: {str(e)}[/red]")
                        if args.verbose:
                            import traceback

                            console.print(traceback.format_exc())
                        return
                else:
                    console.print(
                        "[yellow]No backup created (backups disabled)[/yellow]"
                    )

                # Export the converted audio file
                try:
                    output_file = file_path
                    audio.export(output_file, format="wav")
                    console.print(f"[green]Converted file saved: {output_file}[/green]")
                except Exception as e:
                    console.print(f"[red]Error saving converted file: {str(e)}[/red]")
                    if args.verbose:
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


def generate_spectrogram(original_file, new_file, backup_dir, verbose=False):
    """Generate and save spectrograms for the original and new files."""
    try:
        y_old, sr_old = librosa.load(original_file, sr=None)
        y_new, sr_new = librosa.load(new_file, sr=None)

        # Ensure the backup directory exists
        os.makedirs(backup_dir, exist_ok=True)

        # Set a reasonable n_fft based on signal length
        n_fft = min(2048, len(y_old))
        if n_fft % 2 != 0:  # Ensure n_fft is even
            n_fft -= 1

        # Generate spectrogram for original file
        with plt.ioff():  # Turn off interactive mode
            fig = plt.figure(figsize=(10, 4))
            D_old = librosa.amplitude_to_db(
                np.abs(librosa.stft(y_old, n_fft=n_fft)), ref=np.max
            )
            librosa.display.specshow(D_old, sr=sr_old, x_axis="time", y_axis="log")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Spectrogram of {os.path.basename(original_file)}")
            old_spectrogram_path = os.path.join(
                backup_dir, os.path.basename(original_file) + ".old.png"
            )
            plt.savefig(old_spectrogram_path)
            plt.close(fig)

            # Generate spectrogram for new file
            fig = plt.figure(figsize=(10, 4))
            D_new = librosa.amplitude_to_db(
                np.abs(librosa.stft(y_new, n_fft=n_fft)), ref=np.max
            )
            librosa.display.specshow(D_new, sr=sr_new, x_axis="time", y_axis="log")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Spectrogram of {os.path.basename(new_file)}")
            new_spectrogram_path = os.path.join(
                backup_dir, os.path.basename(new_file) + ".new.png"
            )
            plt.savefig(new_spectrogram_path)
            plt.close(fig)

    except Exception as e:
        console.print(f"[red]Error generating spectrograms: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())


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
                    if any(
                        file_lower.endswith(f".{ext}") for ext in valid_extensions
                    ) and not file.startswith("._"):
                        full_path = os.path.join(root, file)
                        file_list.append(full_path)
                        if args.verbose:
                            console.print(f"[dim]Found: {full_path}[/dim]")
        elif path.is_file():
            file_lower = str(path).lower()
            if any(
                file_lower.endswith(f".{ext}") for ext in valid_extensions
            ) and not path.name.startswith("._"):
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
            console.print(
                f"[cyan]Starting processing of {total_files} files with {args.jobs} parallel jobs[/cyan]"
            )

            task = progress.add_task("Processing files...", total=total_files)

            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_audio, file, args, task_id=task, progress=progress
                    ): file
                    for file in file_list
                }

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    progress.advance(task)
                    try:
                        result = future.result()
                    except Exception as exc:
                        file = futures[future]
                        console.print(
                            f"[red]File {file} generated an exception: {exc}[/red]"
                        )

            console.print("[green]Processing complete![/green]")

    except KeyboardInterrupt:
        console.print(
            "[yellow]Received KeyboardInterrupt, attempting to cancel all threads...[/yellow]"
        )
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
        # Adjust nperseg and noverlap based on sample length
        nperseg = min(1024, len(samples))
        if nperseg % 2 != 0:  # Make sure nperseg is even
            nperseg -= 1
        noverlap = nperseg // 2  # Set noverlap to half of nperseg

        freqs, times, spectrogram = scipy.signal.spectrogram(
            samples, audio.frame_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Get the strongest frequencies
        peaks = np.mean(spectrogram, axis=1)
        # Normalize the peaks
        peaks = peaks / np.max(peaks)

        return peaks
    except Exception as e:
        console.print(
            f"[yellow]Error generating audio fingerprint for {file_path}: {e}[/yellow]"
        )
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


def find_duplicate_files(paths, args, progress, task_id):
    """Find duplicate files using a multi-stage approach with audio fingerprinting."""
    size_groups = defaultdict(list)
    scanned = 0

    # First pass: group by size
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and is_audio_file(str(file_path)):
                    # Update progress
                    scanned += 1
                    progress.update(task_id, completed=scanned)

                    if args.verbose:
                        console.print(f"Scanning: {file_path}")
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
                unmatched = [
                    f
                    for f in file_paths
                    if not any(f in g for g in hash_groups.values() if len(g) > 1)
                ]

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
                                    fingerprints[file1], fingerprints[file2]
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
                similarity = compare_audio_similarity(
                    base_fingerprint, file_fingerprint
                )
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


def find_duplicate_directories(paths, progress, task_id):
    """Find directories with matching names and file counts."""
    dir_map = defaultdict(list)
    scanned = 0

    def get_directory_signature(dir_path):
        """Generate a signature for a directory based on its contents."""
        try:
            # Get all files and subdirectories recursively
            all_items = list(dir_path.rglob("*"))

            # Count files and directories
            files = [f for f in all_items if f.is_file()]
            subdirs = [d for d in all_items if d.is_dir()]

            # Calculate total size of all files
            total_size = sum(f.stat().st_size for f in files)

            # Get relative paths of all items for structure comparison
            rel_paths = sorted(str(item.relative_to(dir_path)) for item in all_items)

            # Get file sizes in a deterministic order
            file_sizes = sorted(f.stat().st_size for f in files)

            return {
                "file_count": len(files),
                "subdir_count": len(subdirs),
                "total_size": total_size,
                "structure": rel_paths,
                "file_sizes": file_sizes,
            }
        except Exception as e:
            console.print(f"[yellow]Error analyzing directory {dir_path}: {e}[/yellow]")
            return None

    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            for dir_path in path.rglob("*"):
                if dir_path.is_dir():
                    # Update progress
                    scanned += 1
                    progress.update(task_id, completed=scanned)

                    # Get directory signature
                    signature = get_directory_signature(dir_path)
                    if signature:
                        # Create a unique key combining name and content signature
                        dir_name = dir_path.name.lower()  # Case-insensitive comparison
                        key = (
                            dir_name,
                            signature["file_count"],
                            signature["subdir_count"],
                            signature["total_size"],
                            tuple(signature["file_sizes"]),  # Make hashable
                            tuple(signature["structure"]),  # Make hashable
                        )
                        dir_map[key].append(dir_path)

    # Return only directories that have duplicates
    duplicates = {k: v for k, v in dir_map.items() if len(v) > 1}

    if duplicates:
        # Log detailed information about matches
        for (
            name,
            file_count,
            subdir_count,
            total_size,
            sizes,
            structure,
        ), paths in duplicates.items():
            console.print(
                f"\n[cyan]Found potential duplicates:[/cyan]\n"
                f"Directory name: [yellow]{name}[/yellow]\n"
                f"File count: {file_count}\n"
                f"Subdirectory count: {subdir_count}\n"
                f"Total size: {total_size} bytes\n"
                f"Structure match: {len(structure)} items"
            )
            if args.verbose:
                console.print("Directory structure:")
                for item in structure[:10]:  # Show first 10 items
                    console.print(f"  {item}")
                if len(structure) > 10:
                    console.print("  ...")

    return duplicates


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
            "Remove duplicate files and directories",
            "Restore from backup",
            "Exit",
        ],
    ).ask()

    if action == "Exit":
        return None, None

    # Get the directory/files to process
    paths = []
    while True:
        path = questionary.path(
            "Select directory or file to process (press Enter with empty path when done):",
            only_directories=False,
        ).ask()

        if not path:  # Empty input
            if paths:  # If we have at least one path, break
                break
            else:  # If no paths yet, ask again
                print("Please select at least one directory or file.")
                continue

        paths.append(path)

        if not questionary.confirm("Add another path?", default=False).ask():
            break

    if not paths:
        return None, None

    # Create a namespace object to match argparse structure
    args = argparse.Namespace()
    args.files = paths

    # Set default values that all modes need
    args.dry_run = False
    args.verbose = False
    args.jobs = 1

    if action == "Remove duplicate files and directories":
        # For duplicate removal, get configuration options
        duplicate_options = questionary.checkbox(
            "Select duplicate removal options:",
            choices=[
                "Use fuzzy matching for similar files",
                "Ignore filenames (match by content only)",
                "Preview changes (dry run)",
                "Show detailed progress",
                "Process files in parallel",
            ],
        ).ask()

        args.use_fuzzy = "Use fuzzy matching for similar files" in duplicate_options
        args.ignore_names = (
            "Ignore filenames (match by content only)" in duplicate_options
        )
        args.dry_run = "Preview changes (dry run)" in duplicate_options
        args.verbose = "Show detailed progress" in duplicate_options

        if "Process files in parallel" in duplicate_options:
            args.jobs = questionary.select(
                "How many parallel jobs?",
                choices=["2", "4", "8", "16", "24", "32", "48", "64"],
                default="4",
            ).ask()
            args.jobs = int(args.jobs)

        # Get backup options
        args.backup_dir = questionary.text(
            "Backup directory path (where duplicates will be moved):",
            default="_backup",
        ).ask()

        if args.backup_dir.strip():  # If not empty
            args.backup_dir = args.backup_dir.strip()
        else:
            args.backup_dir = "_backup"  # Fallback to default

        if args.use_fuzzy:
            threshold_choice = questionary.select(
                "Select fuzzy matching threshold (higher = more strict):",
                choices=[
                    "95 - Nearly identical",
                    "90 - Very similar",
                    "85 - Similar",
                    "80 - Somewhat similar",
                ],
                default="90 - Very similar",
            ).ask()
            args.fuzzy_threshold = int(threshold_choice.split()[0])

        return "duplicates", args

    elif action == "Restore from backup":
        # Get backup directory
        args.backup_dir = questionary.path(
            "Select backup directory to restore from:",
            only_directories=True,
            default="_backup",
        ).ask()

        # Get file extensions to restore
        args.restore_ext = questionary.text(
            "Enter file extensions to restore (comma-separated, e.g., wav,mp3):",
            default="wav,mp3",
        ).ask()

        # Get restore options
        restore_options = questionary.checkbox(
            "Select restore options:",
            choices=[
                "Preview changes (dry run)",
                "Show detailed progress",
                "Process files in parallel",
                "Skip existing files",
                "Overwrite existing files",
            ],
        ).ask()

        args.dry_run = "Preview changes (dry run)" in restore_options
        args.verbose = "Show detailed progress" in restore_options
        args.skip_existing = "Skip existing files" in restore_options
        args.overwrite = "Overwrite existing files" in restore_options

        if "Process files in parallel" in restore_options:
            args.jobs = questionary.select(
                "How many parallel jobs?",
                choices=["2", "4", "8", "16", "24", "32", "48", "64"],
                default="4",
            ).ask()
            args.jobs = int(args.jobs)

        return "restore", args

    elif action == "Shrink samples (convert audio files)":
        # For sample shrinking, get all the conversion options
        args.bitdepth = questionary.select(
            "Select target bit depth:", choices=["8", "16", "24"], default="16"
        ).ask()
        args.bitdepth = int(args.bitdepth)

        args.channels = questionary.select(
            "Select target channels:",
            choices=["1 (mono)", "2 (stereo)"],
            default="2 (stereo)",
        ).ask()
        args.channels = 1 if "1" in args.channels else 2

        args.samplerate = questionary.select(
            "Select target sample rate:",
            choices=["22050", "44100", "48000"],
            default="44100",
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
                "Process files in parallel",
                "Set minimum sample rate",
                "Set minimum bit depth",
                "Convert in place (no backups)",
            ],
        ).ask()

        args.auto_mono = "Auto-convert stereo to mono when possible" in advanced_options
        args.pre_normalize = "Pre-normalize before conversion" in advanced_options
        args.skip_spectrograms = "Skip generating spectrograms" in advanced_options
        args.dry_run = "Preview changes (dry run)" in advanced_options
        convert_in_place = "Convert in place (no backups)" in advanced_options

        # Configure backup settings if not converting in place
        if not convert_in_place:
            args.backup_dir = questionary.text(
                "Backup directory path:",
                default="_backup",
            ).ask()
            if args.backup_dir.strip():  # If not empty
                args.backup_dir = args.backup_dir.strip()
                # Only ask about spectrograms if they weren't explicitly skipped in advanced options
                if not args.skip_spectrograms:
                    args.skip_spectrograms = not questionary.confirm(
                        "Generate spectrograms for backup comparison?", default=False
                    ).ask()
            else:
                args.backup_dir = "-"
                args.skip_spectrograms = True

        if "Process files in parallel" in advanced_options:
            args.jobs = questionary.select(
                "How many parallel jobs? (higher values may improve speed but use more memory)",
                choices=["2", "4", "8", "16", "24", "32", "48", "64"],
                default="4",
            ).ask()
            args.jobs = int(args.jobs)

        if "Set minimum sample rate" in advanced_options:
            args.min_samplerate = questionary.select(
                "Select minimum sample rate:",
                choices=["22050", "44100", "48000"],
                default="22050",
            ).ask()
            args.min_samplerate = int(args.min_samplerate)

        if "Set minimum bit depth" in advanced_options:
            args.min_bitdepth = questionary.select(
                "Select minimum bit depth:", choices=["8", "16", "24"], default="16"
            ).ask()
            args.min_bitdepth = int(args.min_bitdepth)

        if args.auto_mono:
            args.auto_mono_threshold = float(
                questionary.text(
                    "Auto-mono threshold in dB (default: -95.5):", default="-95.5"
                ).ask()
            )

        return "shrink", args

    return action.split()[0].lower(), args  # 'shrink', 'duplicates', or 'restore'


def process_duplicates(args):
    """Process both directory and file level duplicates with visual feedback."""
    # Phase 1: Directory scan - Compare directory contents
    console.print("\n[cyan]Phase 1: Directory Structure Analysis[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} directories"),
        console=console,
    ) as progress:
        # First count total directories for progress
        total_dirs = sum(
            1
            for path_str in args.files
            for _ in Path(path_str).rglob("*")
            if Path(path_str).is_dir()
        )
        scan_task = progress.add_task(
            "[magenta]Scanning for duplicate directory structures...[/magenta]",
            total=total_dirs,
        )

        # Modify find_duplicate_directories to update progress
        dir_duplicates = find_duplicate_directories(args.files, progress, scan_task)
        progress.update(scan_task, completed=total_dirs)

    if dir_duplicates:
        count = sum(len(v) - 1 for v in dir_duplicates.values())
        console.print(
            Panel(
                f"Found [cyan]{count}[/cyan] directories with identical contents",
                title="Directory Structure Analysis Complete",
            )
        )
        if not args.dry_run:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.completed}/{task.total} duplicates"),
                console=console,
            ) as progress:
                dir_task = progress.add_task(
                    "[green]Processing directories...", total=count
                )
                with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                    futures = []
                    for (
                        dir_name,
                        file_count,
                        total_size,
                    ), paths in dir_duplicates.items():
                        future = executor.submit(
                            process_directory_group,
                            dir_name,
                            file_count,
                            total_size,
                            paths,
                            args,
                            progress,
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            future.result()
                            progress.advance(dir_task)
                        except Exception as e:
                            console.print(f"[red]Error processing directory: {e}[/red]")

    # Phase 2: File scan - Compare individual files
    console.print("\n[cyan]Phase 2: Individual File Analysis[/cyan]")

    # Step 1: Initial file scanning
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        total_files = sum(
            1
            for path_str in args.files
            for _ in Path(path_str).rglob("*")
            if Path(_).is_file()
        )
        scan_task = progress.add_task(
            "[magenta]Scanning filesystem for files...[/magenta]",
            total=total_files,
        )

        # First pass: collect files and group by size
        size_groups = defaultdict(list)
        scanned = 0
        for path_str in args.files:
            path = Path(path_str)
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and is_audio_file(str(file_path)):
                        scanned += 1
                        progress.update(scan_task, completed=scanned)
                        size_groups[file_path.stat().st_size].append(file_path)

    # Step 2: Similarity analysis
    potential_duplicates = {
        size: files for size, files in size_groups.items() if len(files) > 1
    }
    total_to_check = sum(len(files) for files in potential_duplicates.values())

    file_duplicates = []
    fuzzy_groups = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        check_task = progress.add_task(
            "[magenta]Analyzing files for duplicates...[/magenta]",
            total=total_to_check,
        )

        checked = 0
        for size, file_paths in potential_duplicates.items():
            if args.verbose:
                console.print(
                    f"\nChecking {len(file_paths)} files of size {size} bytes..."
                )

            # Group files by hash first
            hash_groups = defaultdict(list)
            for file_path in file_paths:
                try:
                    file_hash = get_file_hash(file_path, fuzzy=False)
                    if args.ignore_names:
                        hash_groups[file_hash].append(file_path)
                    else:
                        name_key = file_path.stem.lower()
                        hash_groups[(name_key, file_hash)].append(file_path)
                    checked += 1
                    progress.update(check_task, completed=checked)
                except Exception as e:
                    console.print(f"[red]Error hashing file {file_path}: {e}[/red]")

            # Add exact matches to results
            for group in hash_groups.values():
                if len(group) > 1:
                    file_duplicates.append(group)

            # Check for similar audio content if enabled
            if args.use_fuzzy:
                # Get unmatched files (not in any exact match group)
                unmatched = [
                    f for f in file_paths if not any(f in g for g in file_duplicates)
                ]

                if len(unmatched) > 1:
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
                                    fingerprints[file1], fingerprints[file2]
                                )
                                if similarity >= args.fuzzy_threshold:
                                    similar_files.append(file2)
                                    processed.add(file2)

                        if len(similar_files) > 1:
                            fuzzy_groups.append(similar_files)
                            file_duplicates.append(similar_files)
                            processed.add(file1)

    # Report results and process duplicates
    if file_duplicates:
        total_duplicates = sum(len(group) - 1 for group in file_duplicates)
        console.print(
            Panel(
                f"Found [cyan]{total_duplicates}[/cyan] duplicate files\n"
                f"Including [cyan]{len(fuzzy_groups)}[/cyan] groups of similar files",
                title="File Analysis Complete",
            )
        )

        # Step 3: Process duplicates if not in dry run mode
        if not args.dry_run:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.completed}/{task.total} duplicates"),
                console=console,
            ) as progress:
                process_task = progress.add_task(
                    "[green]Processing duplicate files...", total=total_duplicates
                )

                with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                    futures = []
                    for group in file_duplicates:
                        future = executor.submit(
                            process_file_group,
                            group,
                            fuzzy_groups,
                            args,
                            progress,
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            future.result()
                            progress.advance(process_task)
                        except Exception as e:
                            console.print(
                                f"[red]Error processing file group: {e}[/red]"
                            )

    console.print("[green]Duplicate analysis and removal complete![/green]")


def process_directory_group(dir_name, file_count, total_size, paths, args, progress):
    """Process a group of duplicate directories."""
    try:
        console.print(
            f"\nFound duplicate directories named '[cyan]{dir_name}[/cyan]' "
            f"with {file_count} files ({total_size} bytes):"
        )

        # Sort paths by creation time
        valid_paths = []
        for path in paths:
            try:
                # Wait briefly for cloud storage to download if needed
                retries = 3
                while retries > 0:
                    if path.exists():
                        stat = path.stat()
                        valid_paths.append((path, stat.st_ctime))
                        break
                    retries -= 1
                    if retries > 0:
                        time.sleep(1)  # Wait a second before retry

                if retries == 0:
                    console.print(
                        f"[yellow]Warning: Directory not available after retries: {path}[/yellow]"
                    )
            except (FileNotFoundError, OSError) as e:
                console.print(
                    f"[yellow]Warning: Cannot access directory {path}: {e}[/yellow]"
                )
                continue

        if not valid_paths:
            console.print("[red]No valid paths found in group[/red]")
            return

        valid_paths.sort(key=lambda x: x[1])

        # Keep the oldest directory
        original_dir = valid_paths[0][0]
        console.print(
            f"Keeping oldest copy: [green]{original_dir}[/green] "
            f"(created: {time.ctime(valid_paths[0][1])})"
        )

        # Process newer copies
        for dir_path, ctime in valid_paths[1:]:
            try:
                # First verify source exists
                if not dir_path.exists():
                    console.print(
                        f"[yellow]Skipping unavailable directory: {dir_path}[/yellow]"
                    )
                    continue

                console.print(
                    f"Moving duplicate: [yellow]{dir_path}[/yellow] "
                    f"(created: {time.ctime(ctime)})"
                )

                if not args.dry_run:
                    try:
                        # Create backup path
                        rel_path = dir_path.relative_to(dir_path.parent.parent)
                        backup_path = Path(args.backup_dir) / rel_path

                        # IMPORTANT: Create ALL parent directories first
                        backup_path.parent.mkdir(parents=True, exist_ok=True)

                        # Verify the backup path is valid before attempting move
                        if not backup_path.parent.exists():
                            console.print(
                                f"[red]Error: Backup directory could not be created: {backup_path.parent}[/red]"
                            )
                            continue

                        # Check if destination already exists
                        if backup_path.exists():
                            console.print(
                                f"[yellow]Warning: Backup path already exists: {backup_path}[/yellow]"
                            )
                            # Create a unique name by appending a number
                            counter = 1
                            while backup_path.exists():
                                new_name = f"{backup_path.name}_{counter}"
                                backup_path = backup_path.parent / new_name
                                counter += 1
                            console.print(
                                f"[blue]Using alternate path: {backup_path}[/blue]"
                            )

                        # Do the move
                        try:
                            shutil.move(str(dir_path), str(backup_path))
                        except Exception as move_error:
                            console.print(
                                f"[red]Error moving {dir_path} to {backup_path}: {move_error}[/red]"
                            )
                            # Try to provide more context about the error
                            if not dir_path.exists():
                                console.print(
                                    "[red]Source directory no longer exists[/red]"
                                )
                            if not backup_path.parent.exists():
                                console.print(
                                    "[red]Destination directory does not exist[/red]"
                                )

                    except Exception as e:
                        console.print(
                            f"[red]Error setting up backup path for {dir_path}: {e}[/red]"
                        )

            except Exception as e:
                console.print(f"[red]Error processing directory {dir_path}: {e}[/red]")
                continue

    except Exception as e:
        console.print(f"[red]Error processing directory group {dir_name}: {e}[/red]")
        raise


def process_file_group(group, fuzzy_groups, args, progress):
    """Process a group of duplicate files."""
    try:
        # Get file size for reporting
        file_size = group[0].stat().st_size
        console.print(
            f"\nProcessing duplicate group for '[cyan]{group[0].name}[/cyan]' ({file_size} bytes)"
        )

        # For fuzzy matches, show similarity percentages
        if group in fuzzy_groups:
            base_fingerprint = get_audio_fingerprint(group[0])
            console.print("[cyan]Similarity scores:[/cyan]")
            for file in group[1:]:
                file_fingerprint = get_audio_fingerprint(file)
                similarity = compare_audio_similarity(
                    base_fingerprint, file_fingerprint
                )
                console.print(
                    f"  {file.name}: [yellow]{similarity:.1f}%[/yellow] similar"
                )

        # Sort files by creation time
        files_with_time = []
        for file_path in group:
            try:
                stat = file_path.stat()
                files_with_time.append((file_path, stat.st_ctime))
            except FileNotFoundError:
                console.print(f"[yellow]Warning: File not found: {file_path}[/yellow]")
                continue

        if not files_with_time:
            console.print("[red]No valid files found in group[/red]")
            return

        files_with_time.sort(key=lambda x: x[1])

        # Keep the oldest file
        original_file = files_with_time[0][0]
        console.print(
            f"Keeping oldest copy: [green]{original_file}[/green] "
            f"(created: {time.ctime(files_with_time[0][1])})"
        )

        # Process newer copies
        for file_path, ctime in files_with_time[1:]:
            console.print(
                f"Processing duplicate: [yellow]{file_path}[/yellow] "
                f"(created: {time.ctime(ctime)})"
            )

            if not args.dry_run:
                try:
                    if args.delete_duplicates:
                        console.print(f"[red]Deleting: {file_path}[/red]")
                        file_path.unlink()
                    else:
                        # Create backup path maintaining directory structure
                        rel_path = file_path.relative_to(file_path.parent.parent)
                        backup_path = Path(args.backup_dir) / rel_path

                        # Ensure backup directory exists
                        backup_path.parent.mkdir(parents=True, exist_ok=True)

                        # Move the file
                        console.print(f"Moving to: [blue]{backup_path}[/blue]")
                        shutil.move(str(file_path), str(backup_path))

                except Exception as e:
                    console.print(f"[red]Error processing file {file_path}: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Error processing file group: {e}[/red]")
        raise


def restore_from_backup(args):
    """Restore files from backup to their original locations."""
    console.print("\n[cyan]Starting Backup Restore Process[/cyan]")

    backup_path = Path(args.backup_dir)
    if not backup_path.exists():
        console.print(f"[red]Error: Backup directory {backup_path} not found[/red]")
        return

    # Get list of extensions to restore
    extensions = [ext.strip().lower() for ext in args.restore_ext.split(",")]

    # Step 1: Scan backup directory
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        scan_task = progress.add_task(
            "[magenta]Scanning backup directory...[/magenta]", total=None
        )

        # Collect all files to restore
        restore_files = []
        for ext in extensions:
            for file_path in backup_path.rglob(f"*.{ext}"):
                try:
                    # Calculate original path
                    rel_path = file_path.relative_to(backup_path)
                    target_path = Path(args.files[0]) / rel_path
                    restore_files.append((file_path, target_path))
                except Exception as e:
                    console.print(f"[yellow]Error processing {file_path}: {e}[/yellow]")

        progress.update(
            scan_task, total=len(restore_files), completed=len(restore_files)
        )

    # Report findings
    console.print(
        Panel(
            f"Found [cyan]{len(restore_files)}[/cyan] files to restore",
            title="Backup Scan Complete",
        )
    )

    if not restore_files:
        return

    # Step 2: Restore files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        restore_task = progress.add_task(
            "[green]Restoring files...[/green]", total=len(restore_files)
        )

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for backup_file, target_path in restore_files:
                future = executor.submit(
                    restore_single_file,
                    backup_file,
                    target_path,
                    args,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                    progress.advance(restore_task)
                except Exception as e:
                    console.print(f"[red]Error during restore: {e}[/red]")

    console.print("[green]Restore process complete![/green]")


def restore_single_file(backup_file, target_path, args):
    """Restore a single file from backup to its original location."""
    try:
        if args.verbose:
            console.print(f"Processing: {backup_file} -> {target_path}")

        if target_path.exists():
            if args.skip_existing:
                if args.verbose:
                    console.print(
                        f"[yellow]Skipping existing file: {target_path}[/yellow]"
                    )
                return
            elif not args.overwrite:
                console.print(
                    f"[yellow]Target exists (skipping): {target_path}[/yellow]"
                )
                return

        if not args.dry_run:
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file with metadata preserved
            shutil.copy2(backup_file, target_path)

            if args.verbose:
                console.print(f"[green]Restored: {target_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error restoring {backup_file}: {e}[/red]")
        raise


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

    if action == "restore":
        restore_from_backup(args)
    elif action == "duplicates":
        process_duplicates(args)
    elif action == "shrink":
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
