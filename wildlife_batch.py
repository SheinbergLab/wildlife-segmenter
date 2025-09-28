#!/usr/bin/env python3
"""
Wildlife Documentary Batch Processor
Batch downloads, segments, and analyzes multiple wildlife documentaries
Packages results for transfer to experiment systems
"""

import os
import sys
import json
import argparse
import logging
import sqlite3
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

# Import the main WildlifeDownloader class
from wildlife_segmenter import WildlifeDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WildlifeBatchProcessor:
    """Batch processor for wildlife documentaries"""
    
    def __init__(self, output_dir: str = "wildlife_clips", parallel_jobs: int = 2,
                 clip_duration: int = 30, enable_analysis: bool = True,
                 package_results: bool = True):
        self.output_dir = Path(output_dir)
        self.parallel_jobs = parallel_jobs
        self.clip_duration = clip_duration
        self.enable_analysis = enable_analysis
        self.package_results = package_results
        
        # Setup directories
        self.log_dir = self.output_dir / "batch_logs"
        self.status_dir = self.output_dir / ".batch_status"
        self.package_name = f"wildlife_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._setup_directories()
        self._setup_logging()
        
        # Initialize downloader
        self.downloader = WildlifeDownloader(
            output_dir=str(self.output_dir),
            enable_analysis=enable_analysis,
            show_progress=True
        )
        
        # Default wildlife collections to process
        self.default_items = [
            "Wildlife_Specials",
            "AfricaDocumentary", 
            "ArcticWildlife",
            "OceanDocumentaries",
            "BirdDocumentaries",
            "time-life-nature-video-library"
        ]
        
        logger.info(f"Batch processor initialized - output: {self.output_dir}")
        logger.info(f"Parallel jobs: {parallel_jobs}, Analysis: {enable_analysis}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.status_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup file logging for batch operations"""
        log_file = self.log_dir / "batch_process.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def list_available_documentaries(self) -> List[Dict]:
        """List available wildlife documentaries"""
        logger.info("Fetching available wildlife documentaries...")
        
        # Try multiple collections
        all_items = []
        collections = ["WildlifeDocumentaries", "Wildlife_Specials", "nature"]
        
        for collection in collections:
            try:
                items = self.downloader.get_collection_items(collection)
                all_items.extend(items)
                logger.info(f"Found {len(items)} items in {collection}")
            except Exception as e:
                logger.warning(f"Could not access collection {collection}: {e}")
        
        # Also try known individual items
        known_items = self.downloader._try_known_wildlife_items()
        all_items.extend(known_items)
        
        # Remove duplicates
        seen_ids = set()
        unique_items = []
        for item in all_items:
            item_id = item.get('identifier', '')
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)
        
        logger.info(f"Total unique items found: {len(unique_items)}")
        return unique_items
    
    def explore_item(self, item_id: str) -> Dict:
        """Explore a specific item to see its contents"""
        logger.info(f"Exploring item: {item_id}")
        
        try:
            video_files = self.downloader.get_video_files(item_id)
            
            exploration_data = {
                'item_id': item_id,
                'video_count': len(video_files),
                'total_size_mb': sum(int(vf.get('size', 0)) for vf in video_files) / (1024 * 1024),
                'files': video_files[:10],  # Show first 10
                'has_more': len(video_files) > 10
            }
            
            logger.info(f"Item {item_id}: {len(video_files)} videos, "
                       f"{exploration_data['total_size_mb']:.1f} MB total")
            
            return exploration_data
            
        except Exception as e:
            logger.error(f"Failed to explore {item_id}: {e}")
            return {'item_id': item_id, 'error': str(e)}
    
    def is_item_completed(self, item_id: str) -> bool:
        """Check if an item has been completed previously"""
        status_file = self.status_dir / f"{item_id}.completed"
        return status_file.exists()
    
    def mark_item_completed(self, item_id: str, stats: Dict):
        """Mark an item as completed with statistics"""
        status_file = self.status_dir / f"{item_id}.completed"
        
        completion_data = {
            'completed_at': datetime.now().isoformat(),
            'stats': stats
        }
        
        with open(status_file, 'w') as f:
            json.dump(completion_data, f, indent=2)
    
    def process_single_item(self, item_id: str) -> Tuple[bool, Dict]:
        """Process a single documentary item"""
        logger.info(f"Starting processing: {item_id}")
        
        start_time = time.time()
        stats = {
            'item_id': item_id,
            'start_time': datetime.now().isoformat(),
            'clips_created': 0,
            'clips_analyzed': 0,
            'success': False,
            'error': None
        }
        
        # Setup item-specific logging
        item_log_file = self.log_dir / f"{item_id}.log"
        item_logger = logging.getLogger(f"batch.{item_id}")
        item_handler = logging.FileHandler(item_log_file)
        item_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        item_logger.addHandler(item_handler)
        item_logger.setLevel(logging.INFO)
        
        try:
            # Process the documentary
            item_logger.info(f"Processing documentary: {item_id}")
            
            # Get video files first to estimate scope
            video_files = self.downloader.get_video_files(item_id)
            if not video_files:
                raise Exception(f"No video files found for {item_id}")
            
            item_logger.info(f"Found {len(video_files)} video files")
            
            # Process the documentary
            self.downloader.process_documentary(
                item_id=item_id,
                download=True,
                segment=True,
                parallel=True,
                clip_duration=self.clip_duration,
                specific_file=None
            )
            
            # Count created clips
            clips_dir = self.output_dir / "clips"
            item_clips = []
            for video_dir in clips_dir.iterdir():
                if video_dir.is_dir() and item_id.lower() in video_dir.name.lower():
                    item_clips.extend(list(video_dir.glob("*.mp4")))
            
            stats['clips_created'] = len(item_clips)
            stats['processing_time'] = time.time() - start_time
            stats['success'] = True
            
            item_logger.info(f"Successfully processed {item_id}: {stats['clips_created']} clips created")
            logger.info(f"✓ Completed {item_id}: {stats['clips_created']} clips in {stats['processing_time']:.1f}s")
            
            return True, stats
            
        except Exception as e:
            stats['error'] = str(e)
            stats['processing_time'] = time.time() - start_time
            
            item_logger.error(f"Failed to process {item_id}: {e}")
            logger.error(f"✗ Failed {item_id}: {e}")
            
            return False, stats
        
        finally:
            item_logger.removeHandler(item_handler)
            item_handler.close()
    
    def process_items_parallel(self, item_ids: List[str], resume: bool = False) -> Dict:
        """Process multiple items in parallel"""
        logger.info(f"Processing {len(item_ids)} items with {self.parallel_jobs} parallel jobs")
        
        # Filter out completed items if resuming
        if resume:
            remaining_items = [item for item in item_ids if not self.is_item_completed(item)]
            if len(remaining_items) < len(item_ids):
                skipped = len(item_ids) - len(remaining_items)
                logger.info(f"Resuming: skipping {skipped} completed items")
            item_ids = remaining_items
        
        if not item_ids:
            logger.info("No items to process")
            return {'completed': [], 'failed': [], 'skipped': 0}
        
        completed_items = []
        failed_items = []
        
        # Process items in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all jobs
            future_to_item = {
                executor.submit(self.process_single_item, item_id): item_id 
                for item_id in item_ids
            }
            
            # Process completed jobs
            for future in as_completed(future_to_item):
                item_id = future_to_item[future]
                try:
                    success, stats = future.result()
                    
                    if success:
                        completed_items.append(item_id)
                        self.mark_item_completed(item_id, stats)
                    else:
                        failed_items.append(item_id)
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {item_id}: {e}")
                    failed_items.append(item_id)
        
        results = {
            'completed': completed_items,
            'failed': failed_items,
            'skipped': len(item_ids) - len(completed_items) - len(failed_items) if resume else 0
        }
        
        logger.info(f"Batch processing completed: {len(completed_items)} success, {len(failed_items)} failed")
        return results
    
    def run_analysis(self) -> Dict:
        """Run computer vision analysis on all clips"""
        if not self.enable_analysis:
            logger.info("Analysis disabled, skipping")
            return {'analyzed': 0, 'success': True}
        
        logger.info("Starting computer vision analysis on all clips...")
        
        try:
            # Use the existing scan functionality
            analyzed_count = self.downloader.scan_existing_clips(analyze=True)
            
            logger.info(f"Analysis completed: {analyzed_count} clips analyzed")
            return {'analyzed': analyzed_count, 'success': True}
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'analyzed': 0, 'success': False, 'error': str(e)}
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report_data = {
            'generation_time': datetime.now().isoformat(),
            'processing_summary': {},
            'database_stats': {},
            'top_detections': [],
            'detection_summary': {},
            'confidence_distribution': {}
        }
        
        # Database statistics
        db_path = self.output_dir / "clips_database.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Basic stats
                cursor.execute("SELECT COUNT(*) FROM clips")
                total_clips = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT source_video) FROM clips")
                total_videos = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM clips WHERE analysis_method = 'resnet50_cv_analysis'")
                analyzed_clips = cursor.fetchone()[0]
                
                report_data['database_stats'] = {
                    'total_clips': total_clips,
                    'total_source_videos': total_videos,
                    'analyzed_clips': analyzed_clips,
                    'analysis_coverage': analyzed_clips / total_clips if total_clips > 0 else 0
                }
                
                # Top detections by confidence
                cursor.execute('''
                    SELECT filename, detected_objects, 
                           JSON_EXTRACT(analysis_confidence, '$.max_confidence') as conf
                    FROM clips 
                    WHERE analysis_method = 'resnet50_cv_analysis'
                    ORDER BY conf DESC 
                    LIMIT 20
                ''')
                
                report_data['top_detections'] = [
                    {'filename': row[0], 'objects': row[1], 'confidence': row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Detection type summary
                cursor.execute('''
                    SELECT detected_objects, COUNT(*) as count
                    FROM clips 
                    WHERE detected_objects IS NOT NULL
                    GROUP BY detected_objects 
                    ORDER BY count DESC
                ''')
                
                report_data['detection_summary'] = {
                    row[0]: row[1] for row in cursor.fetchall()
                }
                
                # Confidence distribution
                cursor.execute('''
                    SELECT 
                        CASE 
                            WHEN JSON_EXTRACT(analysis_confidence, '$.max_confidence') >= 0.8 THEN 'very_high'
                            WHEN JSON_EXTRACT(analysis_confidence, '$.max_confidence') >= 0.6 THEN 'high'
                            WHEN JSON_EXTRACT(analysis_confidence, '$.max_confidence') >= 0.4 THEN 'medium'
                            WHEN JSON_EXTRACT(analysis_confidence, '$.max_confidence') >= 0.2 THEN 'low'
                            ELSE 'very_low'
                        END as confidence_level,
                        COUNT(*) as count
                    FROM clips 
                    WHERE analysis_method = 'resnet50_cv_analysis'
                    GROUP BY confidence_level
                ''')
                
                report_data['confidence_distribution'] = {
                    row[0]: row[1] for row in cursor.fetchall()
                }
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error generating database stats: {e}")
                report_data['database_error'] = str(e)
        
        return report_data
    
    def create_package(self, report_data: Dict) -> str:
        """Create tar.gz package with all results"""
        if not self.package_results:
            logger.info("Packaging disabled, skipping")
            return ""
        
        logger.info("Creating results package...")
        
        package_dir = Path(self.package_name)
        package_dir.mkdir(exist_ok=True)
        
        try:
            # Copy clips directory
            clips_src = self.output_dir / "clips"
            if clips_src.exists():
                shutil.copytree(clips_src, package_dir / "clips", dirs_exist_ok=True)
            
            # Copy database
            db_src = self.output_dir / "clips_database.db"
            if db_src.exists():
                shutil.copy2(db_src, package_dir / "clips_database.db")
            
            # Copy CSV export
            csv_src = self.output_dir / "clips_summary.csv"
            if csv_src.exists():
                shutil.copy2(csv_src, package_dir / "clips_summary.csv")
            
            # Copy logs
            if self.log_dir.exists():
                shutil.copytree(self.log_dir, package_dir / "logs", dirs_exist_ok=True)
            
            # Create summary report
            report_file = package_dir / "analysis_summary.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Create human-readable summary
            self._create_readable_summary(package_dir, report_data)
            
            # Create tar.gz
            package_path = f"{self.package_name}.tar.gz"
            with tarfile.open(package_path, "w:gz") as tar:
                tar.add(package_dir, arcname=self.package_name)
            
            # Cleanup temp directory
            shutil.rmtree(package_dir)
            
            # Get package info
            package_size_mb = Path(package_path).stat().st_size / (1024 * 1024)
            
            logger.info(f"Package created: {package_path} ({package_size_mb:.1f} MB)")
            
            return package_path
            
        except Exception as e:
            logger.error(f"Failed to create package: {e}")
            # Cleanup on failure
            if package_dir.exists():
                shutil.rmtree(package_dir)
            return ""
    
    def _create_readable_summary(self, package_dir: Path, report_data: Dict):
        """Create human-readable summary file"""
        summary_file = package_dir / "README.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Wildlife Documentary Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report_data['generation_time']}\n\n")
            
            # Database stats
            if 'database_stats' in report_data and report_data['database_stats']:
                stats = report_data['database_stats']
                f.write("Processing Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total clips: {stats.get('total_clips', 0)}\n")
                f.write(f"Source videos: {stats.get('total_source_videos', 0)}\n")
                f.write(f"Analyzed clips: {stats.get('analyzed_clips', 0)}\n")
                f.write(f"Analysis coverage: {stats.get('analysis_coverage', 0):.1%}\n\n")
            
            # Top detections
            if report_data.get('top_detections'):
                f.write("Top 10 Highest Confidence Detections:\n")
                f.write("-" * 40 + "\n")
                for i, detection in enumerate(report_data['top_detections'][:10], 1):
                    f.write(f"{i:2d}. {detection['filename']} | {detection['objects']} | "
                           f"conf: {detection['confidence']:.3f}\n")
                f.write("\n")
            
            # Detection summary
            if report_data.get('detection_summary'):
                f.write("Detection Type Summary:\n")
                f.write("-" * 25 + "\n")
                for obj_type, count in report_data['detection_summary'].items():
                    f.write(f"{obj_type}: {count}\n")
                f.write("\n")
            
            # Confidence distribution
            if report_data.get('confidence_distribution'):
                f.write("Confidence Distribution:\n")
                f.write("-" * 25 + "\n")
                for level, count in report_data['confidence_distribution'].items():
                    f.write(f"{level}: {count}\n")
                f.write("\n")
            
            f.write("Files included:\n")
            f.write("-" * 15 + "\n")
            f.write("- clips/: All video clips organized by source\n")
            f.write("- clips_database.db: SQLite database with analysis results\n")
            f.write("- clips_summary.csv: CSV export for data analysis\n")
            f.write("- analysis_summary.json: Detailed statistics in JSON format\n")
            f.write("- logs/: Processing logs for debugging\n")
    
    def run_batch_processing(self, item_ids: Optional[List[str]] = None, 
                           resume: bool = False) -> Dict:
        """Run complete batch processing pipeline"""
        logger.info("Starting wildlife documentary batch processing")
        
        start_time = time.time()
        
        # Use default items if none specified
        if not item_ids:
            item_ids = self.default_items
            logger.info("Using default wildlife collections")
        
        logger.info(f"Processing {len(item_ids)} items: {item_ids}")
        
        # Process items
        processing_results = self.process_items_parallel(item_ids, resume=resume)
        
        # Run analysis
        analysis_results = self.run_analysis()
        
        # Export database
        try:
            self.downloader.export_database_for_tcl()
            logger.info("Database exported successfully")
        except Exception as e:
            logger.error(f"Database export failed: {e}")
        
        # Generate report
        report_data = self.generate_summary_report()
        report_data['processing_results'] = processing_results
        report_data['analysis_results'] = analysis_results
        
        # Create package
        package_path = self.create_package(report_data)
        
        # Final summary
        total_time = time.time() - start_time
        
        summary = {
            'success': len(processing_results['failed']) == 0 and analysis_results['success'],
            'total_time': total_time,
            'items_processed': len(processing_results['completed']),
            'items_failed': len(processing_results['failed']),
            'clips_analyzed': analysis_results.get('analyzed', 0),
            'package_path': package_path,
            'report_data': report_data
        }
        
        logger.info(f"Batch processing completed in {total_time:.1f}s")
        logger.info(f"Success: {summary['items_processed']}, Failed: {summary['items_failed']}")
        if package_path:
            logger.info(f"Results packaged: {package_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Wildlife Documentary Batch Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process default collections
  %(prog)s Wildlife_Specials ArcticWildlife  # Process specific items
  %(prog)s --no-analyze Wildlife_Specials    # Skip computer vision analysis
  %(prog)s --list                            # List available documentaries
  %(prog)s --explore Wildlife_Specials       # Explore specific item
  %(prog)s --resume                          # Resume previous batch job
        """
    )
    
    parser.add_argument('items', nargs='*', help='Documentary item IDs to process')
    parser.add_argument('-d', '--duration', type=int, default=30, 
                       help='Clip duration in seconds (default: 30)')
    parser.add_argument('-o', '--output', default='wildlife_clips',
                       help='Output directory (default: wildlife_clips)')
    parser.add_argument('-j', '--jobs', type=int, default=2,
                       help='Number of parallel jobs (default: 2)')
    parser.add_argument('--no-analyze', action='store_true',
                       help='Skip computer vision analysis')
    parser.add_argument('--no-package', action='store_true',
                       help='Skip creating tar.gz package')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--list', action='store_true',
                       help='List available documentaries')
    parser.add_argument('--explore', metavar='ITEM_ID',
                       help='Explore specific item contents')
    
    args = parser.parse_args()
    
    # Create processor
    processor = WildlifeBatchProcessor(
        output_dir=args.output,
        parallel_jobs=args.jobs,
        clip_duration=args.duration,
        enable_analysis=not args.no_analyze,
        package_results=not args.no_package
    )
    
    try:
        # Handle special commands
        if args.list:
            documentaries = processor.list_available_documentaries()
            print(f"\nFound {len(documentaries)} wildlife documentaries:")
            print("-" * 60)
            for i, doc in enumerate(documentaries[:20], 1):
                title = doc.get('title', 'Unknown Title')
                item_id = doc.get('identifier', '')
                print(f"{i:2d}. {title}")
                print(f"    ID: {item_id}")
                if doc.get('description'):
                    desc = doc['description'][:100] + "..." if len(doc['description']) > 100 else doc['description']
                    print(f"    Description: {desc}")
                print()
            
            if len(documentaries) > 20:
                print(f"... and {len(documentaries) - 20} more")
            return
        
        if args.explore:
            exploration = processor.explore_item(args.explore)
            if 'error' in exploration:
                print(f"Error exploring {args.explore}: {exploration['error']}")
            else:
                print(f"\nExploring: {exploration['item_id']}")
                print("-" * 40)
                print(f"Video files: {exploration['video_count']}")
                print(f"Total size: {exploration['total_size_mb']:.1f} MB")
                print("\nVideo files:")
                for i, vf in enumerate(exploration['files'], 1):
                    size_mb = int(vf.get('size', 0)) / (1024 * 1024)
                    print(f"  {i:2d}. {vf['name']} ({size_mb:.1f} MB)")
                if exploration['has_more']:
                    print(f"  ... and {exploration['video_count'] - len(exploration['files'])} more")
            return
        
        # Run batch processing
        results = processor.run_batch_processing(
            item_ids=args.items if args.items else None,
            resume=args.resume
        )
        
        # Print final summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        
        if results['success']:
            print("✓ All processing completed successfully!")
        else:
            print("⚠ Some operations failed - check logs for details")
        
        print(f"Total time: {results['total_time']:.1f} seconds")
        print(f"Items processed: {results['items_processed']}")
        print(f"Items failed: {results['items_failed']}")
        print(f"Clips analyzed: {results['clips_analyzed']}")
        
        if results['package_path']:
            print(f"Results package: {results['package_path']}")
        
        # Show top detections if available
        report = results.get('report_data', {})
        if report.get('top_detections'):
            print("\nTop 5 Detections:")
            for i, detection in enumerate(report['top_detections'][:5], 1):
                print(f"  {i}. {detection['filename']} - {detection['objects']} "
                     f"(conf: {detection['confidence']:.3f})")
        
        print("\nNext steps:")
        print("1. Review the analysis summary and logs")
        print("2. Transfer the .tar.gz file to your experiment systems")
        print("3. Use the SQLite database for detailed analysis")
        print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        print("\nProcessing interrupted - partial results may be available")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
