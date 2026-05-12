"""
Augment human preference data by rotating game states 90/180/270 degrees.

Gomoku is symmetric under board rotation, so each preference session
can be used 4x by rotating clockwisely. This script reads all sessions
under data/human_preferences and generates rotated copies.

Usage:
    python augment_preferences.py

Output:
    For each session_*.json, creates:
        session_*_rot90.json
        session_*_rot180.json
        session_*_rot270.json
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path

BOARD_SIZE = 15
PREFERENCE_DIR = 'data/human_preferences'


def rotate_action(action: int, board_size: int = BOARD_SIZE, times: int = 1) -> int:
    """
    Rotate a board action clockwise.

    Args:
        action: flat action index (row * board_size + col)
        board_size: board dimension (15 for Gomoku)
        times: number of 90-degree clockwise rotations (1, 2, or 3)

    Returns:
        New action index after rotation.
    """
    if action is None:
        return None

    row = action // board_size
    col = action % board_size

    for _ in range(times):
        # 90° clockwise: (row, col) -> (col, board_size - 1 - row)
        row, col = col, board_size - 1 - row

    return row * board_size + col


def rotate_state(state_list, board_size: int = BOARD_SIZE, times: int = 1):
    """
    Rotate a game state (3, board_size, board_size) clockwisely.

    Args:
        state_list: nested list from JSON, shape (3, board_size, board_size)
        board_size: board dimension
        times: number of 90-degree clockwise rotations

    Returns:
        Nested list of the rotated state.
    """
    state = np.array(state_list, dtype=np.float32)
    # np.rot90(..., k=1) is counter-clockwise, so k=-1 is clockwise
    rotated = np.rot90(state, k=-times, axes=(1, 2))
    return rotated.tolist()


def process_session_file(filepath: str, dry_run: bool = False, force: bool = False) -> int:
    """
    Read a session file and generate 3 rotated copies.

    Returns:
        Number of new preference entries written.
    """
    dir_path = os.path.dirname(filepath)
    base_name = Path(filepath).stem  # e.g., "session_abc123"

    # Check if already processed (any rotated file exists)
    rotated_names = [
        f"{base_name}_rot90.json",
        f"{base_name}_rot180.json",
        f"{base_name}_rot270.json",
    ]
    existing_rotated = [n for n in rotated_names if os.path.exists(os.path.join(dir_path, n))]

    if existing_rotated and not force:
        print(f"  SKIPPED (already has {len(existing_rotated)} rotated file(s): {', '.join(existing_rotated)})")
        return 0

    with open(filepath, 'r') as f:
        data = json.load(f)

    original_prefs = data.get('preferences', [])
    if not original_prefs:
        return 0

    total_written = 0

    for rotation in (1, 2, 3):  # 90°, 180°, 270°
        rotated_prefs = []
        for entry in original_prefs:
            rotated_entry = dict(entry)
            rotated_entry['state'] = rotate_state(
                entry['state'], times=rotation
            )
            rotated_entry['preferred_action'] = rotate_action(
                entry.get('preferred_action'), times=rotation
            )
            rotated_entry['rejected_action'] = rotate_action(
                entry.get('rejected_action'), times=rotation
            )
            rotated_entry['rotation'] = rotation * 90
            rotated_prefs.append(rotated_entry)

        out_data = {
            'session_id': f"{data.get('session_id', 'unknown')}_rot{rotation * 90}",
            'source_session': data.get('session_id', 'unknown'),
            'rotation_degrees': rotation * 90,
            'preferences': rotated_prefs,
        }

        out_name = f"{base_name}_rot{rotation * 90}.json"
        out_path = os.path.join(dir_path, out_name)

        if not dry_run:
            with open(out_path, 'w') as f:
                json.dump(out_data, f, indent=2)

        total_written += len(rotated_prefs)
        print(f"  {'Would write' if dry_run else 'Wrote'}: {out_name} ({len(rotated_prefs)} prefs)")

    return total_written


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Augment human preferences by rotation')
    parser.add_argument('--dir', type=str, default=PREFERENCE_DIR,
                        help='Directory containing preference JSON files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing files')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing rotated files')
    args = parser.parse_args()

    print("=" * 60)
    print("Augment Human Preferences by Rotation")
    print("=" * 60)
    print(f"Directory: {args.dir}")
    print(f"Mode: {'DRY RUN (no files written)' if args.dry_run else 'LIVE'}")
    print(f"Force overwrite: {args.force}")
    print("=" * 60)

    pattern = os.path.join(args.dir, '**', 'session_*.json')
    files = glob.glob(pattern, recursive=True)

    # Exclude already-rotated files and files whose rotated versions exist (unless --force)
    original_files = []
    skipped_files = []
    for f in files:
        basename = os.path.basename(f)
        if '_rot' in basename:
            continue  # skip derived files
        dir_path = os.path.dirname(f)
        base_name = Path(f).stem
        rotated_exist = any(
            os.path.exists(os.path.join(dir_path, f"{base_name}_rot{deg}.json"))
            for deg in (90, 180, 270)
        )
        if rotated_exist and not args.force:
            skipped_files.append(f)
        else:
            original_files.append(f)

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} already-processed session(s):")
        for f in skipped_files:
            print(f"  - {f}")

    if not original_files:
        print(f"\nNo unprocessed original session files found in {args.dir}")
        return

    print(f"\nFound {len(original_files)} original session file(s) to process")
    total_original = 0
    total_new = 0

    for filepath in sorted(original_files):
        with open(filepath, 'r') as f:
            data = json.load(f)
        n_prefs = len(data.get('preferences', []))
        total_original += n_prefs

        print(f"\nProcessing: {filepath} ({n_prefs} preferences)")
        new_prefs = process_session_file(filepath, dry_run=args.dry_run, force=args.force)
        total_new += new_prefs

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original preferences: {total_original}")
    print(f"New preferences generated: {total_new}")
    print(f"Total after augmentation: {total_original + total_new}")
    print(f"Multiplier: {(total_original + total_new) / max(total_original, 1):.1f}x")

    if args.dry_run:
        print("\nThis was a dry run. No files were written.")
        print("Run without --dry-run to actually create the files.")


if __name__ == '__main__':
    main()
