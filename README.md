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

### 1. Discover Available Content

```bash
# List available wildlife documentary collections
uv run wildlife_segmenter.py list

# Explore specific collection contents
uv run wildlife_segmenter.py explore Wildlife_Specials
```

### 2. Download and Process Videos

```bash
# Download and segment with 30-second clips (default)
uv run wildlife_segmenter.py download Wildlife_Specials

# With AI content analysis (recommended for research)
uv run wildlife_segmenter.py download Wildlife_Specials --analyze

# Custom clip duration (15 seconds for Instagram Reels)
uv run wildlife_segmenter.py download Wildlife_Specials --clip-duration 15 --analyze

# Download specific video from collection
uv run wildlife_segmenter.py download Wildlife_Specials --file Eagle --analyze
```

### 3. Database Management & Analysis

```bash
# Scan existing clips into database
uv run wildlife_segmenter.py scan

# Scan with AI content analysis
uv run wildlife_segmenter.py scan --analyze

# Query database contents
uv run wildlife_segmenter.py query

# Export for analysis
uv run wildlife_segmenter.py export
```

## Command Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show available documentary collections | `uv run wildlife_segmenter.py list` |
| `explore <collection>` | View contents of a specific collection | `uv run wildlife_segmenter.py explore Wildlife_Specials` |
| `download <collection>` | Download and segment videos | `uv run wildlife_segmenter.py download Wildlife_Specials` |
| `segment <video_file>` | Segment existing video file | `uv run wildlife_segmenter.py segment my_video.mp4` |
| `scan` | Index existing clips into database | `uv run wildlife_segmenter.py scan` |
| `query` | Show database contents | `uv run wildlife_segmenter.py query` |
| `export` | Export database for analysis | `uv run wildlife_segmenter.py export` |

### Options

| Option | Short | Description | Example |
|--------|--------|-------------|---------|
| `--clip-duration` | `-d` | Clip length in seconds (default: 30) | `-d 15` |
| `--file` | `-f` | Specific file to download | `-f Eagle` |
| `--output-dir` | `-o` | Output directory | `-o /path/to/clips` |
| `--no-parallel` | | Disable parallel processing | `--no-parallel` |
| `--analyze` | | Enable AI content analysis | `--analyze` |
| `--analysis-batch-size` | | Batch size for GPU processing (0=auto) | `--analysis-batch-size 16` |
| `--analysis-workers` | | Number of CPU workers (0=auto) | `--analysis-workers 8` |

## Usage Examples

### Research Workflows

**TikTok-length clips with AI analysis:**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 30 --analyze
```

**Instagram Reels with content detection:**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 15 --analyze
```

**YouTube Shorts with GPU acceleration:**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 60 --analyze --analysis-batch-size 16
```

**Micro-moment analysis:**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 5 --analyze
```

### Batch Processing

```bash
# Download multiple collections with AI analysis
uv run wildlife_segmenter.py download WildlifeDocumentaries -d 30 --analyze
uv run wildlife_segmenter.py download Wildlife_Nature_Documentaries -d 30 --analyze
uv run wildlife_segmenter.py download time-life-nature-video-library -d 30 --analyze

# Index everything
uv run wildlife_segmenter.py scan --analyze
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
for collection in WildlifeDocumentaries Wildlife_Specials Wildlife_Nature_Documentaries; do
    uv run wildlife_segmenter.py download $collection -d 30 --analyze &
done
wait

# Index everything and export for TCL
uv run wildlife_segmenter.py scan --analyze
uv run wildlife_segmenter.py export
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
- Check internet connection
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
- Check the command help: `uv run wildlife_segmenter.py --help`
- Review the database with: `uv run wildlife_segmenter.py query`
