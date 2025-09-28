# Wildlife Documentary Segmenter

A Python tool for downloading wildlife documentaries from Internet Archive and automatically segmenting them into research-ready clips with SQLite database management and AI-powered content analysis.

## Features

- **🎬 Automated Downloads**: Download wildlife documentaries from Internet Archive collections
- **⚡ Parallel Segmentation**: Multi-core video processing using FFmpeg
- **🤖 AI Content Analysis**: GPU-accelerated computer vision for automatic content detection
- **🗄️ Database Integration**: SQLite database for clip metadata and research organization
- **🔧 Flexible Configuration**: Customizable clip durations and output directories
- **📊 TCL Integration**: Direct SQLite access for experimental trial configuration
- **🔍 Smart Discovery**: Multiple fallback methods for finding wildlife content
- **🚀 High Performance**: Batch processing optimized for GPU and multi-core systems
- **📦 Batch Processing**: Process multiple collections with intelligent deduplication
- **⏱️ Duration Filtering**: Automatically filter videos by minimum duration
- **📋 Research Packaging**: Create timestamped archives for easy transfer

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (for video processing)
- uv (recommended) or pip

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/](https://ffmpeg.org/)

### Install Dependencies

**Core Dependencies:**
```bash
# Basic functionality (downloading and segmenting)
uv sync

# With AI analysis capabilities (recommended)
uv sync --group analysis
```

**Optional: Manual PyTorch Installation**
For optimal GPU performance, install PyTorch directly:
```bash
# For CUDA-capable systems
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only systems
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Single Collection Processing

```bash
# List available wildlife documentary collections
uv run wildlife_segmenter.py list

# Explore specific collection contents
uv run wildlife_segmenter.py explore Wildlife_Specials

# Download and process single collection
uv run wildlife_segmenter.py download Wildlife_Specials --analyze
```

### Batch Processing (Recommended for Research)

```bash
# Process all videos in Wildlife_Specials with deduplication
uv run wildlife_batch.py Wildlife_Specials

# Process only full-length documentaries (40+ minutes)
uv run wildlife_batch.py --min-duration 40 Wildlife_Specials

# Process multiple collections with custom settings
uv run wildlife_batch.py --min-duration 30 --jobs 4 Wildlife_Specials time-life-nature-video-library

# Explore collections before processing
uv run wildlife_batch.py --explore Wildlife_Specials
```

## Command Reference

### Single Collection Commands (wildlife_segmenter.py)

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show available documentary collections | `uv run wildlife_segmenter.py list` |
| `explore <collection>` | View contents of a specific collection | `uv run wildlife_segmenter.py explore Wildlife_Specials` |
| `download <collection>` | Download and segment videos | `uv run wildlife_segmenter.py download Wildlife_Specials` |
| `segment <video_file>` | Segment existing video file | `uv run wildlife_segmenter.py segment my_video.mp4` |
| `scan` | Index existing clips into database | `uv run wildlife_segmenter.py scan` |
| `query` | Show database contents | `uv run wildlife_segmenter.py query` |
| `export` | Export database for analysis | `uv run wildlife_segmenter.py export` |

### Batch Processing Commands (wildlife_batch.py)

| Command | Description | Example |
|---------|-------------|---------|
| `<collection>` | Process entire collection | `uv run wildlife_batch.py Wildlife_Specials` |
| `--list` | Show available collections | `uv run wildlife_batch.py --list` |
| `--explore <collection>` | View collection contents | `uv run wildlife_batch.py --explore Wildlife_Specials` |
| `--resume` | Resume previous batch job | `uv run wildlife_batch.py --resume` |

### Options

#### Single Collection Options

| Option | Short | Description | Example |
|--------|--------|-------------|---------|
| `--clip-duration` | `-d` | Clip length in seconds (default: 30) | `-d 15` |
| `--file` | `-f` | Specific file to download | `-f Eagle` |
| `--output-dir` | `-o` | Output directory | `-o /path/to/clips` |
| `--no-parallel` | | Disable parallel processing | `--no-parallel` |
| `--analyze` | | Enable AI content analysis | `--analyze` |

#### Batch Processing Options

| Option | Short | Description | Example |
|--------|--------|-------------|---------|
| `--min-duration` | | Minimum video duration in minutes | `--min-duration 40` |
| `--duration` | `-d` | Clip length in seconds (default: 30) | `-d 45` |
| `--jobs` | `-j` | Number of parallel jobs (default: 2) | `-j 4` |
| `--output` | `-o` | Output directory | `-o /data/wildlife` |
| `--no-analyze` | | Skip AI content analysis | `--no-analyze` |
| `--no-package` | | Skip creating tar.gz package | `--no-package` |

## Batch Processing Workflows

The batch processor (`wildlife_batch.py`) is designed for large-scale research data collection and includes several intelligent features:

### Key Features

- **Intelligent Deduplication**: Automatically detects and removes duplicate videos in different formats (e.g., keeps `Eagle.mp4` and removes `Eagle.avi`)
- **Duration Filtering**: Filter videos by minimum duration to focus on full documentaries
- **Parallel Collection Processing**: Process multiple collections simultaneously
- **Research Packaging**: Creates timestamped tar.gz archives for easy data transfer
- **Resume Capability**: Continue interrupted batch jobs
- **Comprehensive Logging**: Detailed logs for debugging and progress tracking

### Research-Ready Examples

**Process all major wildlife collections:**
```bash
# Process all default collections (will take several hours)
uv run wildlife_batch.py

# Process specific collections with 45-minute minimum duration
uv run wildlife_batch.py --min-duration 45 Wildlife_Specials time-life-nature-video-library

# High-performance setup for compute servers (4 parallel jobs)
uv run wildlife_batch.py --jobs 4 --min-duration 30 Wildlife_Specials
```

**Content-specific workflows:**
```bash
# Full-length documentaries only (removes short clips/previews)
uv run wildlife_batch.py --min-duration 40 Wildlife_Specials

# Quick processing without AI analysis (faster)
uv run wildlife_batch.py --no-analyze Wildlife_Specials

# Instagram Reel length clips (15 seconds)
uv run wildlife_batch.py --duration 15 Wildlife_Specials
```

**Research packaging:**
```bash
# Process with automatic packaging for transfer
uv run wildlife_batch.py --min-duration 30 Wildlife_Specials

# Result: wildlife_analysis_20241028_143022.tar.gz
# Contains: clips/, database, logs, and analysis summary
```

### Batch Processing Output

The batch processor creates organized output with research-ready packaging:

```
wildlife_analysis_20241028_143022.tar.gz
├── clips/                          # All video clips organized by source
│   ├── Eagle/
│   │   ├── Eagle_clip_000.mp4
│   │   └── ...
│   ├── Crocodile/
│   └── ...
├── clips_database.db              # Complete SQLite database
├── clips_summary.csv              # Human-readable summary
├── analysis_summary.json          # Detailed statistics
├── README.txt                     # Human-readable summary
└── logs/                          # Processing logs
    ├── batch_process.log
    ├── Wildlife_Specials.log
    └── ...
```

### Smart Deduplication Example

Many Internet Archive collections contain the same video in multiple formats:

**Before deduplication:**
```
Eagle.avi (700MB)
Eagle.mp4 (286MB)
Crocodile.avi (698MB)
Crocodile.mp4 (285MB)
```

**After deduplication:**
```
Eagle.mp4 (286MB) - kept (better format)
Crocodile.mp4 (285MB) - kept (better format)
```

The system intelligently chooses the best version based on:
- Format preference: MP4 > AVI > MOV > MKV
- File size optimization for documentaries (200-800MB range preferred)

## Usage Examples

### Research Workflows

**Social media research (TikTok/Instagram):**
```bash
uv run wildlife_batch.py --duration 30 --min-duration 40 Wildlife_Specials
```

**Micro-moment analysis:**
```bash
uv run wildlife_batch.py --duration 5 --min-duration 20 Wildlife_Specials
```

**Cross-platform comparison:**
```bash
# Create different clip lengths for platform comparison
uv run wildlife_batch.py --duration 15 --output wildlife_15s Wildlife_Specials
uv run wildlife_batch.py --duration 30 --output wildlife_30s Wildlife_Specials
uv run wildlife_batch.py --duration 60 --output wildlife_60s Wildlife_Specials
```

### High-Performance Compute Setup

**Cloud/HPC batch processing:**
```bash
# Process multiple collections simultaneously
uv run wildlife_batch.py --jobs 8 --min-duration 30 \
  Wildlife_Specials time-life-nature-video-library &

# Monitor with nvidia-smi for GPU utilization during AI analysis
```

**Resume interrupted jobs:**
```bash
# If processing was interrupted, resume from where it left off
uv run wildlife_batch.py --resume Wildlife_Specials
```

## AI Content Analysis

The tool includes GPU-accelerated computer vision analysis to automatically detect and categorize wildlife content.

### Analysis Capabilities

- **🦅 Animal Detection**: Birds, mammals, fish, reptiles with behavioral context
- **🌲 Environment Classification**: Forest, ocean, grassland, mountain, desert landscapes  
- **🎬 Scene Analysis**: Close-ups, aerial views, underwater shots, landscape scenes
- **📊 Confidence Scoring**: Statistical confidence for each detection
- **⚡ GPU Acceleration**: Batch processing for high-speed analysis

### Analysis Categories

**Wildlife Types:**
- Birds (flying, perched, feeding)
- Mammals (running, grazing, hunting)
- Marine life (swimming, schooling)
- Reptiles and insects

**Environments:**
- Aquatic (ocean, rivers, coral reefs)
- Terrestrial (forest, grassland, mountains)
- Aerial (sky, cloud formations)
- Desert and polar landscapes

**Behaviors:**
- Hunting and feeding
- Migration and movement
- Social interactions
- Territorial displays

### Performance Metrics

**Analysis Speed:**
- **With GPU**: ~2-8 seconds per 30-second clip
- **CPU only**: ~15-30 seconds per 30-second clip
- **Batch processing**: 8-16 clips processed simultaneously

**Accuracy:**
- Content detection optimized for wildlife documentaries
- Multi-frame validation reduces false positives
- Confidence thresholds tunable for research requirements

## Database Schema

The tool creates a comprehensive SQLite database (`clips_database.db`) with rich metadata:

### Enhanced Clips Table
```sql
CREATE TABLE clips (
    id INTEGER PRIMARY KEY,
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
    
    -- AI Analysis Results
    contains_animals BOOLEAN,        -- Quick animal presence filter
    detected_objects TEXT,           -- Comma-separated detected animals/behaviors
    scene_type TEXT,                -- Environment and scene classification  
    analysis_confidence TEXT,       -- JSON confidence scores
    analysis_method TEXT,           -- Analysis method used
    
    -- Manual Annotation Fields  
    notes TEXT,                     -- Researcher notes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Source Videos Table
```sql
CREATE TABLE source_videos (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    collection_id TEXT,
    total_clips INTEGER,
    download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Advanced Database Queries

### Research-Focused Queries

**Filter by detected animals:**
```sql
-- Get all clips with birds
SELECT filename, filepath, detected_objects, analysis_confidence 
FROM clips 
WHERE detected_objects LIKE '%bird%' AND contains_animals = 1;

-- Predator behavior analysis
SELECT * FROM clips 
WHERE detected_objects LIKE '%hunting%' OR detected_objects LIKE '%predator%';
```

**Environment-based filtering:**
```sql
-- Ocean scenes without animals (pure landscape)
SELECT filename, scene_type 
FROM clips 
WHERE scene_type LIKE '%ocean%' AND contains_animals = 0;

-- Forest scenes with wildlife
SELECT filename, detected_objects, scene_type
FROM clips 
WHERE scene_type LIKE '%forest%' AND contains_animals = 1;
```

**Content analysis statistics:**
```sql
-- Most common animals detected
SELECT detected_objects, COUNT(*) as frequency
FROM clips 
WHERE detected_objects != '' 
GROUP BY detected_objects 
ORDER BY frequency DESC;

-- Environment diversity
SELECT scene_type, COUNT(*) as count, 
       AVG(CAST(contains_animals AS INTEGER)) as animal_percentage
FROM clips 
WHERE scene_type != ''
GROUP BY scene_type;
```

**High-confidence detections:**
```sql
-- Clips with reliable animal detections
SELECT filename, detected_objects, 
       json_extract(analysis_confidence, '$.avg_confidence') as confidence
FROM clips 
WHERE json_extract(analysis_confidence, '$.avg_confidence') > 0.7
  AND contains_animals = 1;
```

## TCL Integration

The SQLite database provides seamless integration with TCL for experimental design:

### Basic TCL Usage
```tcl
package require sqlite3
sqlite3 db "wildlife_clips/clips_database.db"

# Get clips with specific content for trials
set bird_clips [db eval {
    SELECT filename, filepath, start_time, detected_objects
    FROM clips 
    WHERE detected_objects LIKE '%bird%' 
    AND contains_animals = 1
    ORDER BY RANDOM() 
    LIMIT 20
}]

db close
```

### Advanced Experimental Design
```tcl
# Content-based trial configuration
proc setup_behavioral_trials {} {
    sqlite3 db "wildlife_clips/clips_database.db"
    
    # Get different behavior types
    set feeding_clips [db eval {
        SELECT filepath, detected_objects FROM clips 
        WHERE detected_objects LIKE '%feeding%'
        LIMIT 25
    }]
    
    set hunting_clips [db eval {
        SELECT filepath, detected_objects FROM clips 
        WHERE detected_objects LIKE '%hunting%' OR detected_objects LIKE '%predator%'
        LIMIT 25  
    }]
    
    set landscape_clips [db eval {
        SELECT filepath, scene_type FROM clips 
        WHERE contains_animals = 0 AND scene_type != ''
        LIMIT 25
    }]
    
    # Randomize trial order
    set all_trials [concat $feeding_clips $hunting_clips $landscape_clips]
    set randomized_trials [lshuffle $all_trials]
    
    db close
    return $randomized_trials
}

# Configure trials based on content
foreach {filepath content_info} [setup_behavioral_trials] {
    if {[string match "*feeding*" $content_info]} {
        configure_feeding_trial $filepath
    } elseif {[string match "*hunting*" $content_info]} {
        configure_predation_trial $filepath  
    } else {
        configure_control_trial $filepath
    }
}
```

### Cross-Platform Content Studies
```tcl
# Compare attention across different clip durations
proc setup_duration_comparison {} {
    sqlite3 db "wildlife_clips/clips_database.db"
    
    set short_clips [db eval {
        SELECT filepath FROM clips 
        WHERE duration = 15 AND contains_animals = 1 
        LIMIT 30
    }]
    
    set medium_clips [db eval {
        SELECT filepath FROM clips 
        WHERE duration = 30 AND contains_animals = 1
        LIMIT 30
    }]
    
    set long_clips [db eval {
        SELECT filepath FROM clips 
        WHERE duration = 60 AND contains_animals = 1
        LIMIT 30
    }]
    
    db close
    
    # Create balanced trial blocks
    return [list $short_clips $medium_clips $long_clips]
}
```

## File Organization

```
wildlife_clips/
├── clips_database.db          # SQLite database
├── clips_summary.csv          # Human-readable summary
├── downloads/                 # Original downloaded videos
│   ├── Eagle.avi
│   └── Leopard.avi
└── clips/                     # Segmented clips
    ├── Eagle/
    │   ├── Eagle_clip_000.mp4
    │   ├── Eagle_clip_001.mp4
    │   ├── ...
    │   └── metadata.json
    └── Leopard/
        ├── Leopard_clip_000.mp4
        ├── Leopard_clip_001.mp4
        ├── ...
        └── metadata.json
```

## Performance Notes

### Parallel Processing
- Automatically uses all available CPU cores for segmentation
- **Example**: 60-minute documentary → 120 clips in ~2 minutes (8-core machine)
- Use `--no-parallel` for debugging or resource-constrained environments

### AI Analysis Performance  
- **GPU acceleration**: Batch processing of 8-16 clips simultaneously
- **Memory efficient**: Dynamic batch sizing based on available GPU memory
- **Multi-core frame extraction**: Parallel video decoding using all CPU cores

**Analysis Speed Examples:**
- **RTX 4080**: ~3-5 seconds per 30-second clip
- **RTX 3070**: ~5-8 seconds per 30-second clip  
- **CPU only**: ~15-30 seconds per 30-second clip
- **60-min documentary**: 5-15 minutes total analysis time (GPU)

### Cloud Computing
Optimized for spot compute instances:
- **24-48 cores**: Ideal for batch processing
- **Cloud bandwidth**: Faster Internet Archive downloads
- **Cost efficient**: Process dozens of documentaries in minutes

### Recommended Spot Instance Setup
```bash
# Lightning-fast setup on cloud instances
git clone your-repo && cd wildlife-segmenter && uv sync --group analysis

# Batch download multiple collections with GPU analysis
uv run wildlife_batch.py --jobs 4 --min-duration 30 Wildlife_Specials time-life-nature-video-library
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Internet Archive for providing free access to wildlife documentaries
- FFmpeg team for robust video processing tools
- Wildlife filmmakers and researchers who created the original content

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Install FFmpeg (see installation section above)
ffmpeg -version  # Verify installation
```

**Empty database after scanning:**
```bash
# Ensure clips exist in the expected structure
ls wildlife_clips/clips/
# Run scan again
uv run wildlife_segmenter.py scan
```

**Download failures:**
- Check network connection
- Verify collection ID with `explore` command
- Some collections may be temporarily unavailable

**AI analysis not working:**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install analysis dependencies
uv sync --group analysis

# Test with CPU-only analysis
CUDA_VISIBLE_DEVICES="" uv run wildlife_segmenter.py download Wildlife_Specials --analyze
```

**Performance optimization:**
```bash
# Monitor GPU usage during analysis
nvidia-smi -l 1

# Adjust batch size for your GPU memory
uv run wildlife_segmenter.py download Wildlife_Specials --analyze --analysis-batch-size 8

# Use more CPU workers for frame extraction
uv run wildlife_segmenter.py download Wildlife_Specials --analyze --analysis-workers 16
```

### Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check the command help: `uv run wildlife_segmenter.py --help` or `uv run wildlife_batch.py --help`
- Review the database with: `uv run wildlife_segmenter.py query`