#!/usr/bin/env python3
"""
Wildlife Documentary Downloader & Segmenter
Downloads videos from Internet Archive and segments them into research clips
"""

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

# Optional imports for analysis - import at module level
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
    def __init__(self, output_dir: str = "wildlife_clips", enable_analysis: bool = False, analysis_method: str = "clip", batch_size: int = 0, workers: int = 0, show_progress: bool = True):
        # Debug: show what parameters we received
        logger.info(f"WildlifeDownloader init - enable_analysis: {enable_analysis}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = "https://archive.org"
        self.db_path = self.output_dir / "clips_database.db"
        self.enable_analysis = enable_analysis
        self.analysis_method = analysis_method
        self.show_progress = show_progress
        
        # Auto-detect GPU capabilities and scale accordingly  
        try:
            import torch as torch_module
            self.num_gpus = torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
        except ImportError:
            self.num_gpus = 0
        
        # Aggressive scaling for A6000 class GPUs (48GB each)
        if self.num_gpus >= 4:
            # 4x A6000s - can handle massive batches
            self.batch_size = batch_size if batch_size > 0 else 128
            self.workers = workers if workers > 0 else min(32, mp.cpu_count())
            logger.info(f"Multi-GPU setup detected: {self.num_gpus} GPUs - using aggressive batching")
        elif self.num_gpus >= 2:
            # 2+ GPUs
            self.batch_size = batch_size if batch_size > 0 else 64
            self.workers = workers if workers > 0 else min(16, mp.cpu_count())
        elif self.num_gpus == 1:
            # Single GPU - moderate batching
            self.batch_size = batch_size if batch_size > 0 else 32
            self.workers = workers if workers > 0 else min(8, mp.cpu_count())
        else:
            # CPU only
            self.batch_size = batch_size if batch_size > 0 else 4
            self.workers = workers if workers > 0 else mp.cpu_count()
        
        # Initialize CLIP model if analysis is enabled
        self.clip_model = None
        self.clip_preprocess = None
        if self.enable_analysis:
            if CLIP_AVAILABLE:
                logger.info("CLIP dependencies available - initializing model...")
                self._init_clip_model()
            else:
                logger.warning("Analysis requested but some dependencies not available")
                logger.warning("Checking individual imports...")
                
                # Check each import individually
                missing_deps = []
                try:
                    import torch as torch_check
                    logger.info(f"✓ PyTorch {torch_check.__version__} available")
                except ImportError:
                    missing_deps.append("torch")
                    
                try:
                    import cv2 as cv2_check
                    logger.info("✓ OpenCV available")
                except ImportError:
                    missing_deps.append("opencv-python")
                    
                try:
                    from PIL import Image as pil_check
                    logger.info("✓ PIL available")
                except ImportError:
                    missing_deps.append("pillow")
                    
                try:
                    import numpy as np
                    logger.info(f"✓ NumPy {np.__version__} available")
                except ImportError:
                    missing_deps.append("numpy")
                
                if missing_deps:
                    logger.error(f"Missing dependencies: {missing_deps}")
                    logger.error("Install with: uv sync --group analysis")
                    self.enable_analysis = False
                else:
                    logger.info("All dependencies available - proceeding with analysis")
                    # Force enable and try to load model
                    try:
                        self._init_clip_model()
                    except Exception as e:
                        logger.error(f"Failed to initialize model: {e}")
                        self.enable_analysis = False
            
        self._init_database()
        
        # Debug: show what CLIP_AVAILABLE is set to
        logger.info(f"CLIP_AVAILABLE = {CLIP_AVAILABLE}")
        
        if self.enable_analysis:
            gpu_info = f"{self.num_gpus}x GPU" if self.num_gpus > 1 else "1x GPU" if self.num_gpus == 1 else "CPU"
            logger.info(f"Analysis enabled: {gpu_info}, method={self.analysis_method}, batch_size={self.batch_size}, workers={self.workers}")
        else:
            logger.info("Analysis disabled")
        
    def _init_clip_model(self):
        """Initialize CLIP model for content analysis"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP model on {self.device}...")
            
            # Download CLIP model directly from GitHub (more reliable)
            model_url = "https://github.com/openai/CLIP/raw/main/clip/clip.py"
            
            # Use a simpler approach with torch hub or direct implementation
            # For now, we'll implement a lightweight version
            self.clip_model = self._load_simple_clip_model()
            
            if self.clip_model:
                logger.info("CLIP model loaded successfully")
            else:
                raise Exception("Failed to load model")
                
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            logger.info("Falling back to basic computer vision analysis")
            self.enable_analysis = False
    
    def _load_simple_clip_model(self):
        """Load a simple alternative to CLIP for basic analysis"""
        try:
            # Use a pre-trained ResNet for basic image classification
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            model.eval()
            model = model.to(self.device)
            
            # Define transform for preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            return None
        
    def _init_database(self):
        """Initialize SQLite database for clips metadata"""
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
                dominant_colors TEXT,
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
                title TEXT,
                description TEXT,
                total_duration REAL,
                total_clips INTEGER,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filename)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")
        
    def get_collection_items(self, collection_id: str = "WildlifeDocumentaries") -> List[Dict]:
        """Fetch list of items in the wildlife collection"""
        api_url = f"https://archive.org/advancedsearch.php"
        params = {
            'q': f'collection:{collection_id}',
            'fl': 'identifier,title,description,mediatype',
            'rows': 50,
            'output': 'json'
        }
        
        try:
            logger.info(f"Fetching collection: {collection_id}")
            logger.info(f"API URL: {api_url}")
            logger.info(f"Params: {params}")
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"API response status: {response.status_code}")
            logger.info(f"Response keys: {list(data.keys())}")
            
            if 'response' in data and 'docs' in data['response']:
                docs = data['response']['docs']
                logger.info(f"Found {len(docs)} items")
                return docs
            else:
                logger.warning(f"Unexpected response structure: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching collection: {e}")
            return []

    def get_video_files(self, item_id: str) -> List[Dict]:
        """Get downloadable video files for a specific item"""
        metadata_url = f"https://archive.org/metadata/{item_id}"
        
        try:
            response = requests.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
            
            video_files = []
            for file in metadata.get('files', []):
                # Look for video files (mp4, avi, mov)
                if any(ext in file.get('name', '').lower() for ext in ['.mp4', '.avi', '.mov']):
                    video_files.append({
                        'name': file['name'],
                        'size': file.get('size', 0),
                        'format': file.get('format', ''),
                        'download_url': f"https://archive.org/download/{item_id}/{file['name']}"
                    })
            
            return video_files
        except Exception as e:
            logger.error(f"Error fetching metadata for {item_id}: {e}")
            return []

    def download_video(self, url: str, filename: str) -> bool:
        """Download a video file"""
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
                            # Use proper progress display with carriage return cleanup
                            progress_msg = f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)"
                            print(progress_msg, end='', flush=True)
            
            # Important: Clear the progress line and add newline
            if self.show_progress and total_size > 0:
                print("\r" + " " * 80 + "\r", end='')  # Clear the line
            print()  # Add proper newline
            logger.info(f"Downloaded: {filepath}")
            return True
            
        except Exception as e:
            # Make sure we clear any partial progress line on error
            print("\r" + " " * 80 + "\r", end='')
            print()
            logger.error(f"Error downloading {filename}: {e}")
            return False

    def segment_video(self, video_path: Path, clip_duration: int = 30, parallel: bool = True, max_workers: Optional[int] = None) -> List[Path]:
        """Segment video into clips using FFmpeg with optional parallel processing"""
        clips_dir = self.output_dir / "clips" / video_path.stem
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found. Please install FFmpeg to segment videos.")
            return []
        
        logger.info(f"Segmenting {video_path.name} into {clip_duration}s clips...")
        
        # Get video duration
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ]
        
        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            total_duration = float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return []
        
        # Prepare segment jobs
        segment_jobs = []
        for segment in range(0, int(total_duration), clip_duration):
            start_time = segment
            clip_filename = f"{video_path.stem}_clip_{segment//clip_duration:03d}.mp4"
            clip_path = clips_dir / clip_filename
            
            segment_jobs.append({
                'video_path': video_path,
                'start_time': start_time,
                'duration': clip_duration,
                'output_path': clip_path,
                'segment_num': segment//clip_duration
            })
        
        logger.info(f"Creating {len(segment_jobs)} clips...")
        
        if parallel and len(segment_jobs) > 1:
            return self._segment_parallel(segment_jobs, max_workers)
        else:
            return self._segment_sequential(segment_jobs)
    
    def _segment_sequential(self, segment_jobs: List[Dict]) -> List[Path]:
        """Sequential segmentation (original method)"""
        clip_paths = []
        
        for job in segment_jobs:
            success = self._create_single_clip(job)
            if success:
                clip_paths.append(job['output_path'])
        
        return clip_paths
    
    def _segment_parallel(self, segment_jobs: List[Dict], max_workers: Optional[int] = None) -> List[Path]:
        """Parallel segmentation using multiprocessing"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(segment_jobs))
        
        logger.info(f"Using {max_workers} parallel workers for segmentation")
        
        clip_paths = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(_create_clip_worker, job): job for job in segment_jobs}
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    success = future.result()
                    if success:
                        clip_paths.append(job['output_path'])
                    completed += 1
                    
                    # Progress update with proper terminal cleanup
                    progress = (completed / len(segment_jobs)) * 100
                    progress_msg = f"\rSegmentation progress: {progress:.1f}% ({completed}/{len(segment_jobs)})"
                    print(progress_msg, end='', flush=True)
                    
                except Exception as e:
                    logger.error(f"Error processing segment {job['segment_num']}: {e}")
        
        # Clear progress line and add proper newline
        print("\r" + " " * 80 + "\r", end='')
        print()
        
        # Sort clips by segment number to maintain order
        clip_paths.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        return clip_paths
    
    def _create_single_clip(self, job: Dict) -> bool:
        """Create a single clip from job parameters"""
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(job['video_path']),
            '-ss', str(job['start_time']),
            '-t', str(job['duration']),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            str(job['output_path']),
            '-y'  # Overwrite output files
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating clip {job['segment_num']}: {e}")
            return False

    def create_metadata_file(self, clips: List[Path], video_title: str, clip_duration: int = 30, collection_id: str = None):
        """Create metadata file for the clips and update database"""
        metadata = {
            'source_video': video_title,
            'clip_duration': clip_duration,
            'total_clips': len(clips),
            'clips': []
        }
        
        # Add to source_videos table
        self._add_source_video(video_title, collection_id, len(clips))
        
        for i, clip_path in enumerate(clips):
            clip_info = {
                'filename': clip_path.name,
                'clip_number': i,
                'start_time': i * clip_duration,
                'end_time': (i + 1) * clip_duration,
                'contains_animals': None,  # To be filled by researcher
                'scene_type': None,  # To be filled by researcher
                'notes': ""
            }
            metadata['clips'].append(clip_info)
            
            # Add clip to database (without analysis for now)
            self._add_clip_to_database(clip_path, video_title, collection_id, i, 
                                     i * clip_duration, (i + 1) * clip_duration, clip_duration)
        
        # Batch analysis after all clips are created
        if self.enable_analysis and clips:
            logger.info(f"Starting batch analysis of {len(clips)} clips...")
            try:
                self._batch_analyze_clips(clips)
                logger.info(f"Completed analysis of {len(clips)} clips")
            except Exception as e:
                logger.error(f"Error during batch analysis: {e}")
                # Fallback to individual analysis
                logger.info("Falling back to individual clip analysis...")
                for clip_path in clips:
                    try:
                        analysis_results = self._analyze_clip_content(clip_path)
                        if analysis_results:
                            self._update_clip_analysis(clip_path, analysis_results)
                    except Exception as e2:
                        logger.error(f"Error analyzing {clip_path.name}: {e2}")
        elif self.enable_analysis:
            logger.warning("Analysis enabled but no clips to analyze")
        else:
            logger.info("Analysis disabled - skipping content analysis")
        
        metadata_path = clips[0].parent / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata file: {metadata_path}")
        if self.enable_analysis:
            logger.info(f"Completed analysis of {len(clips)} clips")

    def list_available_videos(self, limit: int = 10):
        """List available videos in the collection"""
        items = self.get_collection_items()
        
        if not items:
            print("No items found with default collection. Trying alternative searches...")
            
            # Try accessing known collections directly
            known_items = self._try_known_wildlife_items()
            if known_items:
                items = known_items
            else:
                # Try alternative collection IDs
                alternative_collections = [
                    "Wildlife_Nature_Documentaries",
                    "Wildlife_Specials", 
                    "WildlifeOnOneEp95OasisDavidAttenborough23Feb1988"
                ]
                
                for alt_collection in alternative_collections:
                    print(f"\nTrying collection: {alt_collection}")
                    items = self.get_collection_items(alt_collection)
                    if items:
                        break
                
                # Try direct search for wildlife videos
                if not items:
                    print("\nTrying direct search for wildlife videos...")
                    items = self._search_wildlife_videos()
        
        if not items:
            print("No wildlife documentaries found. Please check your internet connection or try again later.")
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
    
    def _search_wildlife_videos(self) -> List[Dict]:
        """Alternative search method for wildlife videos"""
        # Try multiple search strategies
        search_queries = [
            'title:(wildlife OR nature OR animal OR documentary) AND mediatype:movies',
            'wildlife documentary AND mediatype:movies',
            'nature documentary AND mediatype:movies', 
            'animal documentary AND mediatype:movies',
            'david attenborough AND mediatype:movies',
            'BBC nature AND mediatype:movies'
        ]
        
        for query in search_queries:
            api_url = "https://archive.org/advancedsearch.php"
            params = {
                'q': query,
                'fl': 'identifier,title,description,mediatype',
                'rows': 20,
                'output': 'json'
            }
            
            try:
                logger.info(f"Trying search: {query}")
                response = requests.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'response' in data and 'docs' in data['response']:
                    docs = data['response']['docs']
                    # Filter for actual wildlife/nature content
                    wildlife_docs = []
                    for doc in docs:
                        title = doc.get('title', '').lower()
                        desc = doc.get('description', '').lower()
                        
                        # Look for wildlife/nature keywords
                        wildlife_keywords = ['wildlife', 'nature', 'animal', 'bird', 'mammal', 
                                           'ocean', 'forest', 'safari', 'attenborough', 'bbc',
                                           'planet', 'species', 'habitat', 'conservation']
                        
                        if any(keyword in title or keyword in desc for keyword in wildlife_keywords):
                            wildlife_docs.append(doc)
                    
                    if wildlife_docs:
                        logger.info(f"Found {len(wildlife_docs)} wildlife documentaries with query: {query}")
                        return wildlife_docs
                
            except Exception as e:
                logger.error(f"Error in search '{query}': {e}")
                continue
        
        return []
    
    def _try_known_wildlife_items(self) -> List[Dict]:
        """Try to access known wildlife documentary identifiers directly"""
        known_identifiers = [
            "WildlifeDocumentaries",
            "Wildlife_Nature_Documentaries", 
            "Wildlife_Specials",
            "time-life-nature-video-library",
            "duck-academy",  # Nature Documentary Films
            "Natural_History_Wildlife"
        ]
        
        items = []
        for identifier in known_identifiers:
            try:
                # Try to get metadata directly
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
                    logger.info(f"Found direct item: {identifier}")
                    
            except Exception as e:
                logger.debug(f"Could not access {identifier}: {e}")
                continue
        
        return items
    
    def _add_source_video(self, video_title: str, collection_id: str, total_clips: int):
        """Add source video to database"""
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
            logger.error(f"Error adding source video to database: {e}")
        finally:
            conn.close()
    
    def _add_clip_to_database(self, clip_path: Path, source_video: str, collection_id: str,
                             clip_number: int, start_time: float, end_time: float, duration: float):
        """Add individual clip to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get file stats
            file_size = clip_path.stat().st_size if clip_path.exists() else None
            
            # Get video metadata using ffprobe
            width, height, fps = self._get_video_metadata(clip_path)
            
            cursor.execute('''
                INSERT OR REPLACE INTO clips 
                (filename, filepath, source_video, source_collection, clip_number,
                 start_time, end_time, duration, file_size, width, height, fps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (clip_path.name, str(clip_path), source_video, collection_id, clip_number,
                  start_time, end_time, duration, file_size, width, height, fps))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error adding clip to database: {e}")
        finally:
            conn.close()
    
    def _get_video_metadata(self, video_path: Path) -> tuple[Optional[int], Optional[int], Optional[float]]:
        """Get video dimensions and frame rate using ffprobe"""
        if not video_path.exists():
            return None, None, None
            
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    width = stream.get('width')
                    height = stream.get('height')
                    fps_str = stream.get('r_frame_rate', '0/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den) if float(den) != 0 else None
                    else:
                        fps = float(fps_str) if fps_str else None
                    return width, height, fps
                    
        except Exception as e:
            logger.debug(f"Could not get video metadata for {video_path}: {e}")
            
        return None, None, None
    
    def export_database_for_tcl(self, output_file: str = None):
        """Export database information for TCL usage"""
        # The main database file is already available for TCL
        logger.info(f"SQLite database for TCL: {self.db_path}")
        
        # Create SQL dump for backup/portability (optional)
        if output_file is None:
            output_file = str(self.output_dir / "clips_backup.sql")
            
        conn = sqlite3.connect(self.db_path)
        
        with open(output_file, 'w') as f:
            for line in conn.iterdump():
                f.write(f"{line}\n")
        
        logger.info(f"SQL backup created: {output_file}")
        
        # Create CSV for easy inspection
        csv_file = str(self.output_dir / "clips_summary.csv")
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, source_video, clip_number, start_time, end_time,
                   duration, width, height, fps, contains_animals, scene_type, 
                   detected_objects, filepath
            FROM clips ORDER BY source_video, clip_number
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        with open(csv_file, 'w') as f:
            f.write("filename,source_video,clip_number,start_time,end_time,duration,width,height,fps,contains_animals,scene_type,detected_objects,filepath\n")
            for row in rows:
                f.write(",".join(str(x) if x is not None else "" for x in row) + "\n")
        
        logger.info(f"CSV summary created: {csv_file}")
        
        # Show sample TCL usage
        print(f"\nTo use in TCL:")
        print(f"package require sqlite3")
        print(f"sqlite3 db \"{self.db_path}\"")
        print(f"set clips [db eval {{SELECT filename, filepath, contains_animals FROM clips}}]")
    
    def query_clips(self, **filters):
        """Query clips database with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        where_clauses = []
        params = []
        
        if 'source_video' in filters:
            where_clauses.append("source_video LIKE ?")
            params.append(f"%{filters['source_video']}%")
        
        if 'min_duration' in filters:
            where_clauses.append("duration >= ?")
            params.append(filters['min_duration'])
            
        if 'max_duration' in filters:
            where_clauses.append("duration <= ?")
            params.append(filters['max_duration'])
            
        if 'contains_animals' in filters:
            where_clauses.append("contains_animals = ?")
            params.append(filters['contains_animals'])
            
        if 'scene_type' in filters:
            where_clauses.append("scene_type = ?")
            params.append(filters['scene_type'])
        
        query = "SELECT * FROM clips"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY source_video, clip_number"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
    def scan_existing_clips(self, analyze: bool = False):
        """Scan clips directory and add existing clips to database"""
        clips_dir = self.output_dir / "clips"
        
        if not clips_dir.exists():
            logger.info("No clips directory found")
            return
        
        total_added = 0
        
        # Enable analysis for scanning if requested
        if analyze and not self.enable_analysis:
            self.enable_analysis = analyze
            if CLIP_AVAILABLE and not self.clip_model:
                self._init_clip_model()
        
        # Scan all subdirectories in clips/
        for video_dir in clips_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            logger.info(f"Scanning clips from: {video_dir.name}")
            
            # Look for metadata.json to get source info
            metadata_file = video_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                source_video = metadata.get('source_video', video_dir.name)
                clip_duration = metadata.get('clip_duration', 30)
            else:
                source_video = video_dir.name
                clip_duration = 30  # Default assumption
                logger.warning(f"No metadata.json found for {video_dir.name}, using defaults")
            
            # Add source video to database
            video_clips = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            self._add_source_video(source_video, None, len(video_clips))
            
            # Scan all video clips in this directory
            clip_number = 0
            for clip_path in sorted(video_clips):
                if clip_path.is_file():
                    # Extract timing from filename or use sequence
                    start_time = clip_number * clip_duration
                    end_time = start_time + clip_duration
                    
                    # Try to extract clip number from filename
                    filename = clip_path.stem
                    if '_clip_' in filename:
                        try:
                            clip_num_str = filename.split('_clip_')[-1]
                            extracted_clip_num = int(clip_num_str)
                            clip_number = extracted_clip_num
                            start_time = clip_number * clip_duration
                            end_time = start_time + clip_duration
                        except ValueError:
                            pass
                    
                    # Add to database
                    self._add_clip_to_database(
                        clip_path, source_video, None, clip_number,
                        start_time, end_time, clip_duration
                    )
                    
                    # Analyze if requested
                    if analyze and self.enable_analysis:
                        analysis_results = self._analyze_clip_content(clip_path)
                        if analysis_results:
                            self._update_clip_analysis(clip_path, analysis_results)
                    
                    clip_number += 1
                    total_added += 1
        
        logger.info(f"Scanned and added {total_added} existing clips to database")
        if analyze and self.enable_analysis:
            logger.info("Content analysis completed for all clips")
        return total_added

    def explore_collection(self, collection_id: str):
        """Explore the contents of a specific collection"""
        print(f"\nExploring collection: {collection_id}")
        print("-" * 60)
        
        try:
            # Get collection metadata
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
                    for i, vf in enumerate(video_files[:20]):  # Show first 20
                        print(f"{i+1:2d}. {vf['name']}")
                        print(f"     Size: {vf['size_mb']:.1f} MB, Format: {vf['format']}")
                    
                    if len(video_files) > 20:
                        print(f"... and {len(video_files) - 20} more files")
                        
                    print(f"\nTo download this collection:")
                    print(f"uv run wildlife_segmenter.py download {collection_id}")
                else:
                    print("No video files found in this collection.")
            else:
                print("Could not access collection files.")
                
        except Exception as e:
            logger.error(f"Error exploring collection {collection_id}: {e}")

    def process_documentary(self, item_id: str, download: bool = True, segment: bool = True, parallel: bool = True, clip_duration: int = 30, specific_file: Optional[str] = None):
        """Download and process a single documentary"""
        video_files = self.get_video_files(item_id)
        
        if not video_files:
            logger.error(f"No video files found for {item_id}")
            return
        
        # If specific file requested, find it
        if specific_file:
            matching_files = [vf for vf in video_files if specific_file.lower() in vf['name'].lower()]
            if matching_files:
                selected_video = matching_files[0]
                logger.info(f"Found specific file: {selected_video['name']}")
            else:
                logger.error(f"File containing '{specific_file}' not found. Available files:")
                for vf in video_files[:10]:
                    logger.error(f"  - {vf['name']}")
                return
        else:
            # Prefer MP4 files, then smaller files
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
                logger.error(f"Video file not found: {video_path}")


def _create_clip_worker(job: Dict) -> bool:
    """Worker function for parallel clip creation (must be at module level for multiprocessing)"""
    ffmpeg_cmd = [
        'ffmpeg', '-i', str(job['video_path']),
        '-ss', str(job['start_time']),
        '-t', str(job['duration']),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        str(job['output_path']),
        '-y'  # Overwrite output files
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
    parser.add_argument("--file", "-f", help="Specific file to download from collection")
    parser.add_argument("--clip-duration", "-d", type=int, default=30, 
                       help="Duration of each clip in seconds (default: 30)")
    parser.add_argument("--no-parallel", action="store_true", 
                       help="Disable parallel processing")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bars (useful for remote servers)")
    parser.add_argument("--output-dir", "-o", default="wildlife_clips",
                       help="Output directory (default: wildlife_clips)")
    parser.add_argument("--analyze", action="store_true",
                       help="Automatically analyze clips for content keywords")
    parser.add_argument("--analysis-method", choices=["clip", "yolo", "google"], default="clip",
                       help="Method for automatic analysis (default: clip)")
    
    args = parser.parse_args()
    
    downloader = WildlifeDownloader(args.output_dir)
    parallel = not args.no_parallel
    
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
        downloader.process_documentary(args.target, parallel=parallel, clip_duration=args.clip_duration, specific_file=args.file)
    
    elif args.command == "segment":
        if not args.target:
            print("Error: segment command requires a video file path")
            return
        video_file = Path(args.target)
        if video_file.exists():
            clips = downloader.segment_video(video_file, clip_duration=args.clip_duration, parallel=parallel)
            if clips:
                downloader.create_metadata_file(clips, video_file.name, args.clip_duration)
        else:
            logger.error(f"Video file not found: {video_file}")
    
    elif args.command == "export":
        downloader.export_database_for_tcl()
        
    elif args.command == "query":
        # Example query - you can extend this
        results = downloader.query_clips()
        print(f"\nFound {len(results)} clips in database:")
        for row in results[:10]:  # Show first 10
            print(f"  {row[1]} - Clip {row[5]} ({row[6]:.1f}s-{row[7]:.1f}s)")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more clips")
            
    elif args.command == "scan":
        count = downloader.scan_existing_clips()
        print(f"Added {count} existing clips to database")


def main_old():
    downloader = WildlifeDownloader()
    
    if len(sys.argv) == 1:
        print("Wildlife Documentary Downloader & Segmenter")
        print("Usage:")
        print("  python wildlife_segmenter.py list                    # List available documentaries")
        print("  python wildlife_segmenter.py explore <collection_id>   # Explore videos in a collection")
        print("  python wildlife_segmenter.py download <item_id>      # Download and segment a documentary")
        print("  python wildlife_segmenter.py download <item_id> --no-parallel  # Use sequential processing")
        print("  python wildlife_segmenter.py segment <video_file>    # Segment existing video file")
        print("  python wildlife_segmenter.py segment <video_file> --no-parallel  # Sequential segmentation")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        downloader.list_available_videos()
    
    elif command == "explore" and len(sys.argv) > 2:
        collection_id = sys.argv[2]
        downloader.explore_collection(collection_id)
    
    elif command == "download" and len(sys.argv) > 2:
        item_id = sys.argv[2]
        parallel = "--no-parallel" not in sys.argv
        downloader.process_documentary(item_id, parallel=parallel)
    
    elif command == "segment" and len(sys.argv) > 2:
        video_file = Path(sys.argv[2])
        parallel = "--no-parallel" not in sys.argv
        if video_file.exists():
            clips = downloader.segment_video(video_file, parallel=parallel)
            if clips:
                downloader.create_metadata_file(clips, video_file.name)
        else:
            logger.error(f"Video file not found: {video_file}")
    
    else:
        print("Invalid command. Use 'list', 'download <item_id>', or 'segment <video_file>'")

if __name__ == "__main__":
    main()
