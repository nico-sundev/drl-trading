"""Helper script for profiling components of the DRL trading system."""

import cProfile
import pstats
from collections.abc import Callable
from pathlib import Path
from typing import Any


def profile_function(
    func: Callable,
    output_file: str = "profile_output.prof",
    sort_by: str = "cumulative",
    top_n: int = 20,
) -> Any:
    """
    Profile a function and save/display results.

    Args:
        func: The function to profile (should be a callable with no args)
        output_file: Path to save the profile data
        sort_by: Sorting criterion ('cumulative', 'time', 'calls')
        top_n: Number of top results to display

    Returns:
        The result of the function call
    """
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = func()
    finally:
        profiler.disable()

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(output_path))

    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(sort_by)

    print(f"\n{'=' * 80}")
    print(f"Top {top_n} functions by {sort_by} time")
    print(f"{'=' * 80}\n")
    stats.print_stats(top_n)

    print(f"\nProfile data saved to: {output_path.absolute()}")
    print(f"To visualize: uv run snakeviz {output_path}")

    return result


def profile_preprocessing_pipeline() -> None:
    """Example: Profile the preprocessing pipeline."""
    from drl_trading_preprocess.main import main

    def run_preprocess() -> None:
        # Your preprocessing logic here
        main()

    profile_function(
        run_preprocess,
        output_file="logs/profiles/preprocessing_profile.prof",
        sort_by="cumulative",
        top_n=30,
    )


def profile_data_ingestion() -> None:
    """Example: Profile data ingestion."""
    # Import your ingest main or specific components
    pass


def compare_profiles(profile1: str, profile2: str) -> None:
    """
    Compare two profile outputs to see improvements/regressions.

    Args:
        profile1: Path to first profile
        profile2: Path to second profile
    """
    print(f"\n{'=' * 80}")
    print("Profile 1:")
    print(f"{'=' * 80}\n")
    stats1 = pstats.Stats(profile1)
    stats1.strip_dirs()
    stats1.sort_stats("cumulative")
    stats1.print_stats(20)

    print(f"\n{'=' * 80}")
    print("Profile 2:")
    print(f"{'=' * 80}\n")
    stats2 = pstats.Stats(profile2)
    stats2.strip_dirs()
    stats2.sort_stats("cumulative")
    stats2.print_stats(20)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "preprocessing":
            profile_preprocessing_pipeline()
        elif sys.argv[1] == "compare" and len(sys.argv) == 4:
            compare_profiles(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python profile_helper.py preprocessing")
        print("  python profile_helper.py compare profile1.prof profile2.prof")
