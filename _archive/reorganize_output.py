#!/usr/bin/env python3
"""
Standalone script to reorganize output structure
Can be run independently without full pipeline execution
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.reorganization_utils import reorganize_output

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for standalone reorganization"""
    import argparse

    parser = argparse.ArgumentParser(description='Reorganize clustering output into clean hybrid structure')
    parser.add_argument('--market', type=str, required=True,
                        help='Market name (e.g., germany, usa)')
    parser.add_argument('--keep-details', action='store_true',
                        help='Keep DETAILS folder with original outputs')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup creation (NOT RECOMMENDED)')
    parser.add_argument('--base-dir', type=str, default='output',
                        help='Base output directory (default: output)')

    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("🔄 OUTPUT REORGANIZATION TOOL")
    logger.info("=" * 80)
    logger.info(f"Market: {args.market}")
    logger.info(f"Keep details: {args.keep_details}")
    logger.info(f"Create backup: {not args.no_backup}")
    logger.info("=" * 80)

    try:
        stats = reorganize_output(
            market=args.market,
            base_output_dir=args.base_dir,
            keep_details=args.keep_details,
            create_backup=not args.no_backup
        )

        if stats['errors']:
            logger.warning(f"\n⚠️  Completed with {len(stats['errors'])} errors")
            for error in stats['errors']:
                logger.warning(f"  - {error}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"\n❌ Reorganization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
