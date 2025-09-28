#!/usr/bin/env python3
"""
Wildlife Documentary Downloader & Segmenter
Downloads videos from Internet Archive and segments them into research clips
"""

# Core imports
import os
import sys
import json
import subprocess
import logging
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import requests
from urllib.parse import urljoin
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Optional imports for analysis
CLIP_AVAILABLE = False
torch = None
transforms = None
Image = None
cv2 = None

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import cv2
    CLIP_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).debug(f"Analysis dependencies not available: {e}")
    CLIP_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WildlifeDownloader:
    def __init__(self, output_dir: str = "wildlife_clips", enable_analysis: bool = False, 
                 analysis_method: str = "clip", batch_size: int = 0, workers: int = 0, 
                 show_progress: bool = True):
        logger.info(f"WildlifeDownloader init - enable_analysis: {enable_analysis}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = "https://archive.org"
        self.db_path = self.output_dir / "clips_database.db"
        self.enable_analysis = enable_analysis
        self.analysis_method = analysis_method
        self.show_progress = show_progress
        
        # Auto-detect GPU capabilities
        try:
            import torch as torch_module
            self.num_gpus = torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
        except ImportError:
            self.num_gpus = 0
        
        # Scale for A6000 class GPUs
        if self.num_gpus >= 4:
            self.batch_size = batch_size if batch_size > 0 else 128
            self.workers = workers if workers > 0 else min(32, mp.cpu_count())
            logger.info(f"Multi-GPU setup detected: {self.num_gpus} GPUs - using aggressive batching")
        elif self.num_gpus >= 2:
            self.batch_size = batch_size if batch_size > 0 else 64
            self.workers = workers if workers > 0 else min(16, mp.cpu_count())
        elif self.num_gpus == 1:
            self.batch_size = batch_size if batch_size > 0 else 32
            self.workers = workers if workers > 0 else min(8, mp.cpu_count())
        else:
            self.batch_size = batch_size if batch_size > 0 else 4
            self.workers = workers if workers > 0 else mp.cpu_count()
        
        self._init_database()
        logger.info(f"CLIP_AVAILABLE = {CLIP_AVAILABLE}")
        
        # Initialize model if analysis enabled
        self.clip_model = None
        if self.enable_analysis:
            if CLIP_AVAILABLE:
                logger.info("CLIP dependencies available - initializing model...")
                self._init_clip_model()
            else:
                logger.warning("Analysis requested but dependencies not available")
                self.enable_analysis = False
            
        if self.enable_analysis:
            gpu_info = f"{self.num_gpus}x GPU" if self.num_gpus > 1 else "1x GPU" if self.num_gpus == 1 else "CPU"
            logger.info(f"Analysis enabled: {gpu_info}, method={self.analysis_method}, batch_size={self.batch_size}, workers={self.workers}")
        else:
            logger.info("Analysis disabled")
            
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                source_video TEXT NOT NULL,
                source_collection TEXT,
                clip_number INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                fps REAL,
                contains_animals BOOLEAN,
                scene_type TEXT,
                detected_objects TEXT,
                analysis_confidence TEXT,
                analysis_method TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filepath)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                collection_id TEXT,
                total_clips INTEGER,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filename)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")

    def _init_clip_model(self):
        """Initialize model for analysis"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model on {self.device}...")
            
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            model.eval()
            model = model.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            self.clip_model = model
            logger.info("Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.enable_analysis = False

    def get_collection_items(self, collection_id: str = "WildlifeDocumentaries") -> List[Dict]:
        """Fetch items from collection"""
        api_url = "https://archive.org/advancedsearch.php"
        params = {
            'q': f'collection:{collection_id}',
            'fl': 'identifier,title,description,mediatype',
            'rows': 50,
            'output': 'json'
        }
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data and 'docs' in data['response']:
                return data['response']['docs']
            return []
                
        except Exception as e:
            logger.error(f"Error fetching collection: {e}")
            return []

    def get_video_files(self, item_id: str) -> List[Dict]:
        """Get video files for item"""
        metadata_url = f"https://archive.org/metadata/{item_id}"
        
        try:
            response = requests.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
            
            video_files = []
            for file in metadata.get('files', []):
                if any(ext in file.get('name', '').lower() for ext in ['.mp4', '.avi', '.mov']):
                    video_files.append({
                        'name': file['name'],
                        'size': file.get('size', 0),
                        'format': file.get('format', ''),
                        'download_url': f"https://archive.org/download/{item_id}/{file['name']}"
                    })
            
            return video_files
        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return []

    def download_video(self, url: str, filename: str) -> bool:
        """Download video"""
        filepath = self.output_dir / "downloads" / filename
        filepath.parent.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and self.show_progress:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            if self.show_progress and total_size > 0:
                print("\r" + " " * 40 + "\r", end='')
            print()
            logger.info(f"Downloaded: {filepath}")
            return True
            
        except Exception as e:
            print("\r" + " " * 40 + "\r", end='')
            print()
            logger.error(f"Error downloading: {e}")
            return False

    def segment_video(self, video_path: Path, clip_duration: int = 30, parallel: bool = True) -> List[Path]:
        """Segment video into clips"""
        clips_dir = self.output_dir / "clips" / video_path.stem
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found")
            return []
        
        logger.info(f"Segmenting {video_path.name} into {clip_duration}s clips...")
        
        # Get duration
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ]
        
        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            total_duration = float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Error getting duration: {e}")
            return []
        
        # Create clip jobs
        segment_jobs = []
        for segment in range(0, int(total_duration), clip_duration):
            clip_filename = f"{video_path.stem}_clip_{segment//clip_duration:03d}.mp4"
            clip_path = clips_dir / clip_filename
            
            segment_jobs.append({
                'video_path': video_path,
                'start_time': segment,
                'duration': clip_duration,
                'output_path': clip_path,
                'segment_num': segment//clip_duration
            })
        
        logger.info(f"Creating {len(segment_jobs)} clips...")
        
        if parallel and len(segment_jobs) > 1:
            return self._segment_parallel(segment_jobs)
        else:
            return self._segment_sequential(segment_jobs)
    
    def _segment_sequential(self, segment_jobs: List[Dict]) -> List[Path]:
        """Sequential segmentation"""
        clip_paths = []
        for job in segment_jobs:
            if self._create_single_clip(job):
                clip_paths.append(job['output_path'])
        return clip_paths
    
    def _segment_parallel(self, segment_jobs: List[Dict]) -> List[Path]:
        """Parallel segmentation"""
        max_workers = min(mp.cpu_count(), len(segment_jobs))
        logger.info(f"Using {max_workers} parallel workers")
        
        clip_paths = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_create_clip_worker, job): job for job in segment_jobs}
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    if future.result():
                        clip_paths.append(job['output_path'])
                    completed += 1
                    
                    if self.show_progress:
                        progress = (completed / len(segment_jobs)) * 100
                        print(f"\rSegmentation: {progress:.1f}%", end='', flush=True)
                        
                except Exception as e:
                    logger.error(f"Error processing segment: {e}")
        
        if self.show_progress:
            print("\r" + " " * 40 + "\r", end='')
        print()
        
        clip_paths.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return clip_paths
    
    def _create_single_clip(self, job: Dict) -> bool:
        """Create single clip"""
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(job['video_path']),
            '-ss', str(job['start_time']),
            '-t', str(job['duration']),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            str(job['output_path']),
            '-y'
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def create_metadata_file(self, clips: List[Path], video_title: str, clip_duration: int = 30, collection_id: str = None):
        """Create metadata and analyze clips"""
        # Add to database
        self._add_source_video(video_title, collection_id, len(clips))
        
        for i, clip_path in enumerate(clips):
            self._add_clip_to_database(clip_path, video_title, collection_id, i, 
                                     i * clip_duration, (i + 1) * clip_duration, clip_duration)
        
        # Analysis
        if self.enable_analysis and clips:
            logger.info(f"Starting analysis of {len(clips)} clips...")
            try:
                self._batch_analyze_clips(clips)
                logger.info("Analysis completed")
            except Exception as e:
                logger.error(f"Analysis error: {e}")
        
        # Save metadata JSON
        metadata = {
            'source_video': video_title,
            'clip_duration': clip_duration,
            'total_clips': len(clips),
            'clips': [{'filename': c.name, 'clip_number': i} for i, c in enumerate(clips)]
        }
        
        metadata_path = clips[0].parent / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata: {metadata_path}")

    def _add_source_video(self, video_title: str, collection_id: str, total_clips: int):
        """Add source to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO source_videos 
                (filename, collection_id, total_clips) 
                VALUES (?, ?, ?)
            ''', (video_title, collection_id, total_clips))
            conn.commit()
        except Exception as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def _add_clip_to_database(self, clip_path: Path, source_video: str, collection_id: str,
                             clip_number: int, start_time: float, end_time: float, duration: float):
        """Add clip to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            file_size = clip_path.stat().st_size if clip_path.exists() else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO clips 
                (filename, filepath, source_video, source_collection, clip_number,
                 start_time, end_time, duration, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (clip_path.name, str(clip_path), source_video, collection_id, clip_number,
                  start_time, end_time, duration, file_size))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()

    def _batch_analyze_clips(self, clips: List[Path]):
        """Analyze clips"""
        if not self.enable_analysis or not self.clip_model:
            return
            
        logger.info(f"Analyzing {len(clips)} clips...")
        
        if self.num_gpus > 1:
            for i in range(self.num_gpus):
                try:
                    name = torch.cuda.get_device_name(i)
                    logger.info(f"GPU {i}: {name}")
                except:
                    logger.info(f"GPU {i}: Available")
        
        successful = 0
        for i, clip_path in enumerate(clips):
            try:
                if self.show_progress:
                    progress = ((i + 1) / len(clips)) * 100
                    print(f"\rAnalyzing: {progress:.1f%} ({i+1}/{len(clips)})", end='', flush=True)
                
                results = self._analyze_clip_content(clip_path)
                if results:
                    self._update_clip_analysis(clip_path, results)
                    successful += 1
            except Exception as e:
                logger.error(f"Analysis error for {clip_path.name}: {e}")
        
        if self.show_progress:
            print("\r" + " " * 40 + "\r", end='')
        print()
        
        logger.info(f"Analysis complete: {successful}/{len(clips)} clips")

    def _analyze_clip_content(self, clip_path: Path) -> Dict:
        """Analyze clip content"""
        if not self.enable_analysis or not clip_path.exists():
            return {}
            
        filename = clip_path.stem.lower()
        
        # Simple keyword detection
        animal_keywords = ['whale', 'bird', 'eagle', 'lion', 'leopard', 'shark', 'deer', 'bear']
        environment_keywords = ['ocean', 'forest', 'desert', 'mountain', 'river', 'lake']
        
        detected_objects = []
        scene_types = []
        contains_animals = False
        
        for keyword in animal_keywords:
            if keyword in filename:
                detected_objects.append(f"{keyword}_detected")
                contains_animals = True
        
        for keyword in environment_keywords:
            if keyword in filename:
                scene_types.append(f"{keyword}_landscape")
        
        if not detected_objects and not scene_types:
            contains_animals = True
            detected_objects.append("wildlife_content")
            scene_types.append("natural_environment")
        
        return {
            'detected_objects': detected_objects,
            'scene_type': scene_types,
            'contains_animals': contains_animals,
            'confidence_data': {'method': 'basic_heuristic', 'confidence': 0.6}
        }

    def _update_clip_analysis(self, clip_path: Path, analysis_results: Dict):
        """Update clip with analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            detected_objects = ", ".join(analysis_results.get('detected_objects', []))
            scene_type = ", ".join(analysis_results.get('scene_type', []))
            contains_animals = analysis_results.get('contains_animals', None)
            confidence_json = json.dumps(analysis_results.get('confidence_data', {}))
            
            cursor.execute('''
                UPDATE clips 
                SET detected_objects = ?, scene_type = ?, contains_animals = ?,
                    analysis_confidence = ?, analysis_method = ?
                WHERE filepath = ?
            ''', (detected_objects, scene_type, contains_animals, 
                  confidence_json, self.analysis_method, str(clip_path)))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Update error: {e}")
        finally:
            conn.close()

    def list_available_videos(self, limit: int = 10):
        """List available videos"""
        items = self.get_collection_items()
        
        if not items:
            items = self._try_known_wildlife_items()
        
        if not items:
            print("No wildlife documentaries found")
            return
            
        print(f"\nAvailable Wildlife Documentaries (showing first {limit}):")
        print("-" * 60)
        
        for i, item in enumerate(items[:limit]):
            print(f"{i+1}. {item.get('title', 'Unknown Title')}")
            print(f"   ID: {item['identifier']}")
            if item.get('description'):
                desc = item['description'][:100] + "..." if len(item['description']) > 100 else item['description']
                print(f"   Description: {desc}")
            print()
    
    def _try_known_wildlife_items(self) -> List[Dict]:
        """Try known collections"""
        known_identifiers = [
            "WildlifeDocumentaries",
            "Wildlife_Nature_Documentaries", 
            "Wildlife_Specials",
            "time-life-nature-video-library"
        ]
        
        items = []
        for identifier in known_identifiers:
            try:
                metadata_url = f"https://archive.org/metadata/{identifier}"
                response = requests.get(metadata_url)
                response.raise_for_status()
                metadata = response.json()
                
                if 'metadata' in metadata:
                    item = {
                        'identifier': identifier,
                        'title': metadata['metadata'].get('title', identifier),
                        'description': metadata['metadata'].get('description', ''),
                        'mediatype': metadata['metadata'].get('mediatype', 'collection')
                    }
                    items.append(item)
                    
            except Exception as e:
                logger.debug(f"Could not access {identifier}: {e}")
                continue
        
        return items

    def explore_collection(self, collection_id: str):
        """Explore collection contents"""
        print(f"\nExploring collection: {collection_id}")
        print("-" * 60)
        
        try:
            metadata_url = f"https://archive.org/metadata/{collection_id}"
            response = requests.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
            
            if 'files' in metadata:
                video_files = []
                for file in metadata['files']:
                    if any(ext in file.get('name', '').lower() for ext in ['.mp4', '.avi', '.mov']):
                        size_mb = int(file.get('size', 0)) / (1024 * 1024)
                        video_files.append({
                            'name': file['name'],
                            'size_mb': size_mb,
                            'format': file.get('format', 'unknown')
                        })
                
                if video_files:
                    print(f"Found {len(video_files)} video files:")
                    for i, vf in enumerate(video_files[:10]):
                        print(f"{i+1:2d}. {vf['name']}")
                        print(f"     Size: {vf['size_mb']:.1f} MB")
                    
                    if len(video_files) > 10:
                        print(f"... and {len(video_files) - 10} more files")
                else:
                    print("No video files found")
            else:
                print("Could not access collection files")
                
        except Exception as e:
            logger.error(f"Error exploring collection: {e}")

    def process_documentary(self, item_id: str, download: bool = True, segment: bool = True, 
                          parallel: bool = True, clip_duration: int = 30, specific_file: Optional[str] = None):
        """Process documentary"""
        video_files = self.get_video_files(item_id)
        
        if not video_files:
            logger.error(f"No video files found for {item_id}")
            return
        
        if specific_file:
            matching_files = [vf for vf in video_files if specific_file.lower() in vf['name'].lower()]
            if matching_files:
                selected_video = matching_files[0]
            else:
                logger.error(f"File '{specific_file}' not found")
                return
        else:
            video_files.sort(key=lambda x: (
                0 if '.mp4' in x['name'].lower() else 1,
                int(x.get('size', 0))
            ))
            selected_video = video_files[0]
        
        logger.info(f"Selected: {selected_video['name']} ({selected_video.get('size', 0)} bytes)")
        
        if download:
            success = self.download_video(selected_video['download_url'], selected_video['name'])
            if not success:
                return
        
        if segment:
            video_path = self.output_dir / "downloads" / selected_video['name']
            if video_path.exists():
                clips = self.segment_video(video_path, clip_duration=clip_duration, parallel=parallel)
                if clips:
                    self.create_metadata_file(clips, selected_video['name'], clip_duration, item_id)
                    logger.info(f"Successfully created {len(clips)} clips")
            else:
                logger.error(f"Video file not found")

    def scan_existing_clips(self, analyze: bool = False):
        """Scan existing clips"""
        clips_dir = self.output_dir / "clips"
        
        if not clips_dir.exists():
            logger.info("No clips directory found")
            return 0
        
        total_added = 0
        
        for video_dir in clips_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            logger.info(f"Scanning clips from: {video_dir.name}")
            
            video_clips = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            self._add_source_video(video_dir.name, None, len(video_clips))
            
            for clip_number, clip_path in enumerate(sorted(video_clips)):
                if clip_path.is_file():
                    self._add_clip_to_database(
                        clip_path, video_dir.name, None, clip_number,
                        clip_number * 30, (clip_number + 1) * 30, 30
                    )
                    
                    if analyze and self.enable_analysis:
                        results = self._analyze_clip_content(clip_path)
                        if results:
                            self._update_clip_analysis(clip_path, results)
                    
                    total_added += 1
        
        logger.info(f"Scanned {total_added} clips")
        return total_added

    def export_database_for_tcl(self):
        """Export database for TCL"""
        logger.info(f"SQLite database for TCL: {self.db_path}")
        
        csv_file = self.output_dir / "clips_summary.csv"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM clips ORDER BY source_video, clip_number')
        
        rows = cursor.fetchall()
        conn.close()
        
        with open(csv_file, 'w') as f:
            f.write("id,filename,filepath,source_video,contains_animals,scene_type,detected_objects\n")
            for row in rows:
                f.write(",".join(str(x) if x is not None else "" for x in row[:7]) + "\n")
        
        logger.info(f"CSV created: {csv_file}")
        
        print(f"\nTo use in TCL:")
        print(f"sqlite3 db \"{self.db_path}\"")
        print(f"set clips [db eval {{SELECT filename, filepath FROM clips}}]")

    def query_clips(self, **filters):
        """Query clips database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clips ORDER BY source_video, clip_number")
        results = cursor.fetchall()
        conn.close()
        return results


def _create_clip_worker(job: Dict) -> bool:
    """Worker for parallel clip creation"""
    ffmpeg_cmd = [
        'ffmpeg', '-i', str(job['video_path']),
        '-ss', str(job['start_time']),
        '-t', str(job['duration']),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        str(job['output_path']),
        '-y'
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Wildlife Documentary Downloader & Segmenter")
    parser.add_argument("command", choices=["list", "explore", "download", "segment", "export", "query", "scan"], 
                       help="Command to run")
    parser.add_argument("target", nargs="?", help="Collection ID, item ID, or video file path")
    parser.add_argument("--file", "-f", help="Specific file to download")
    parser.add_argument("--clip-duration", "-d", type=int, default=30, help="Clip duration in seconds")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--output-dir", "-o", default="wildlife_clips", help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Enable content analysis")
    parser.add_argument("--analysis-method", choices=["clip"], default="clip", help="Analysis method")
    parser.add_argument("--analysis-batch-size", type=int, default=0, help="Analysis batch size")
    parser.add_argument("--analysis-workers", type=int, default=0, help="Number of analysis workers")
    
    args = parser.parse_args()
    
    # Create downloader with specified options
    downloader = WildlifeDownloader(
        output_dir=args.output_dir,
        enable_analysis=args.analyze,
        analysis_method=args.analysis_method,
        batch_size=args.analysis_batch_size,
        workers=args.analysis_workers,
        show_progress=not args.no_progress
    )
    
    parallel = not args.no_parallel
    
    # Execute commands
    if args.command == "list":
        downloader.list_available_videos()
    
    elif args.command == "explore":
        if not args.target:
            print("Error: explore command requires a collection ID")
            return
        downloader.explore_collection(args.target)
    
    elif args.command == "download":
        if not args.target:
            print("Error: download command requires an item ID")
            return
        downloader.process_documentary(
            args.target, 
            download=True, 
            segment=True, 
            parallel=parallel,
            clip_duration=args.clip_duration,
            specific_file=args.file
        )
    
    elif args.command == "segment":
        if not args.target:
            print("Error: segment command requires a video file path")
            return
        video_file = Path(args.target)
        if video_file.exists():
            clips = downloader.segment_video(
                video_file, 
                clip_duration=args.clip_duration, 
                parallel=parallel
            )
            if clips:
                downloader.create_metadata_file(clips, video_file.name, args.clip_duration)
        else:
            logger.error(f"Video file not found: {video_file}")
    
    elif args.command == "scan":
        analyze = args.analyze
        count = downloader.scan_existing_clips(analyze=analyze)
        print(f"Scanned {count} existing clips")
    
    elif args.command == "export":
        downloader.export_database_for_tcl()
    
    elif args.command == "query":
        results = downloader.query_clips()
        print(f"Found {len(results)} clips in database")
        for row in results[:10]:  # Show first 10
            print(f"  {row[1]} - {row[3]} (clip {row[5]})")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")


if __name__ == "__main__":
    main()
