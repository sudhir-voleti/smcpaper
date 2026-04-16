#!/usr/bin/env python3
"""
think_PKL.py - Surgical PKL slimming tool
Removes alpha_filtered from large SMC result PKLs to reclaim disk space
"""

import argparse
import pickle
import os
import glob
import sys
from pathlib import Path

def check_and_slim_pkl(pkl_path, dry_run=True, verbose=False, size_threshold_mb=500):
    """
    Check if PKL has alpha_filtered and remove it if present
    Returns: (success, bytes_saved, message)
    """
    try:
        original_size = os.path.getsize(pkl_path)
        original_size_mb = original_size / (1024**2)
        
        # Skip small files (DGP, etc.)
        if original_size_mb < size_threshold_mb:
            return (False, 0, f"SKIP: {original_size_mb:.1f}MB < {size_threshold_mb}MB threshold")
        
        # Load PKL
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check for alpha_filtered
        if 'alpha_filtered' not in data:
            return (False, 0, "SKIP: no alpha_filtered key")
        
        # Calculate alpha size
        alpha_data = data['alpha_filtered']
        alpha_bytes = len(pickle.dumps(alpha_data))
        alpha_mb = alpha_bytes / (1024**2)
        
        if dry_run:
            return (True, alpha_bytes, f"DRY-RUN: would save {alpha_mb:.1f}MB ({100*alpha_mb/original_size_mb:.1f}%)")
        
        # Surgical removal
        del data['alpha_filtered']
        
        # Write to temp file first (atomic operation)
        temp_path = pkl_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Replace original
        os.replace(temp_path, pkl_path)
        
        new_size = os.path.getsize(pkl_path)
        actual_saved = original_size - new_size
        
        return (True, actual_saved, f"SAVED: {actual_saved/(1024**2):.1f}MB (expected {alpha_mb:.1f}MB)")
        
    except Exception as e:
        return (False, 0, f"ERROR: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='Surgically remove alpha_filtered from SMC result PKLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python think_PKL.py --base_dir ./results --dry_run
  python think_PKL.py --base_dir ./results --execute --verbose
  python think_PKL.py --base_dir ./results --threshold 1000 --execute
        """
    )
    
    parser.add_argument('--base_dir', required=True, 
                        help='Base directory containing PKL files (recursive search)')
    parser.add_argument('--dry_run', action='store_true', default=True,
                        help='Preview what would be done (default)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform deletions (use with caution)')
    parser.add_argument('--threshold', type=float, default=500,
                        help='Size threshold in MB to consider a PKL "large" (default: 500)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print details for every file processed')
    
    args = parser.parse_args()
    
    # Safety: dry_run is default unless --execute is explicitly passed
    dry_run = not args.execute
    
    if not dry_run:
        print("⚠️  EXECUTE MODE: Will actually modify PKL files!")
        print("Press Ctrl+C within 3 seconds to cancel...")
        import time
        time.sleep(3)
        print("Proceeding with execution...\n")
    else:
        print("🔍 DRY-RUN MODE: No files will be modified\n")
    
    # Find all PKLs
    base_path = Path(args.base_dir)
    pkl_pattern = str(base_path / '**' / '*.pkl')
    all_pkls = glob.glob(pkl_pattern, recursive=True)
    
    print(f"Found {len(all_pkls)} PKL files in {args.base_dir}")
    print(f"Size threshold: {args.threshold} MB")
    print(f"Mode: {'EXECUTE' if not dry_run else 'DRY-RUN'}\n")
    
    # Process
    results = {
        'processed': 0,
        'skipped_small': 0,
        'skipped_no_alpha': 0,
        'success': 0,
        'error': 0,
        'total_bytes_saved': 0
    }
    
    for pkl_path in sorted(all_pkls):
        success, bytes_saved, msg = check_and_slim_pkl(
            pkl_path, 
            dry_run=dry_run, 
            verbose=args.verbose,
            size_threshold_mb=args.threshold
        )
        
        results['processed'] += 1
        
        if 'SKIP: no alpha' in msg:
            results['skipped_no_alpha'] += 1
        elif 'SKIP:' in msg and 'threshold' in msg:
            results['skipped_small'] += 1
        elif 'ERROR' in msg:
            results['error'] += 1
        elif success:
            results['success'] += 1
            results['total_bytes_saved'] += bytes_saved
        
        if args.verbose or success or 'ERROR' in msg:
            print(f"{os.path.basename(pkl_path)}: {msg}")
        
        # Progress every 100 files
        if results['processed'] % 100 == 0:
            print(f"  ... processed {results['processed']}/{len(all_pkls)} files")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total PKLs found:     {len(all_pkls)}")
    print(f"Processed:            {results['processed']}")
    print(f"Skipped (small):      {results['skipped_small']}")
    print(f"Skipped (no alpha):   {results['skipped_no_alpha']}")
    print(f"Successfully slimmed: {results['success']}")
    print(f"Errors:               {results['error']}")
    print(f"\nTotal space {'would be' if dry_run else ''} saved: {results['total_bytes_saved']/(1024**3):.2f} GB")
    
    if dry_run and results['success'] > 0:
        print(f"\nRun with --execute to actually perform deletions")

if __name__ == '__main__':
    main()
