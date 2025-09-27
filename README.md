# Wildlife Documentary Segmenter

A Python tool for downloading wildlife documentaries from Internet Archive and automatically segmenting them into research-ready clips with SQLite database management.

## Features

- **🎬 Automated Downloads**: Download wildlife documentaries from Internet Archive collections
- **⚡ Parallel Segmentation**: Multi-core video processing using FFmpeg
- **🗄️ Database Integration**: SQLite database for clip metadata and research organization
- **🔧 Flexible Configuration**: Customizable clip durations and output directories
- **📊 TCL Integration**: Direct SQLite access for experimental trial configuration
- **🔍 Smart Discovery**: Multiple fallback methods for finding wildlife content

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

### Install Wildlife Segmenter

```bash
# Clone the repository
git clone https://github.com/yourusername/wildlife-segmenter.git
cd wildlife-segmenter

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
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

# Custom clip duration (15 seconds for Instagram Reels)
uv run wildlife_segmenter.py download Wildlife_Specials --clip-duration 15

# Download specific video from collection
uv run wildlife_segmenter.py download Wildlife_Specials --file Eagle
```

### 3. Database Management

```bash
# Scan existing clips into database
uv run wildlife_segmenter.py scan

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

## Usage Examples

### Research Workflows

**TikTok-length clips (30 seconds):**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 30
```

**Instagram Reels (15 seconds):**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 15
```

**YouTube Shorts (60 seconds):**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 60
```

**Micro-moment analysis (5 seconds):**
```bash
uv run wildlife_segmenter.py download Wildlife_Specials -d 5
```

### Batch Processing

```bash
# Download multiple collections
uv run wildlife_segmenter.py download WildlifeDocumentaries -d 30
uv run wildlife_segmenter.py download Wildlife_Nature_Documentaries -d 30
uv run wildlife_segmenter.py download time-life-nature-video-library -d 30

# Index everything
uv run wildlife_segmenter.py scan
```

## Database Schema

The tool creates a SQLite database (`clips_database.db`) with comprehensive metadata:

### Clips Table
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
    contains_animals BOOLEAN,  -- For manual annotation
    scene_type TEXT,          -- For manual annotation
    notes TEXT,               -- For manual annotation
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

## TCL Integration

The SQLite database can be directly accessed from TCL for experimental configuration:

```tcl
package require sqlite3
sqlite3 db "wildlife_clips/clips_database.db"

# Get random clips for trials
set trial_clips [db eval {
    SELECT filename, filepath, start_time, duration, contains_animals
    FROM clips 
    WHERE duration = 30 AND contains_animals IS NOT NULL
    ORDER BY RANDOM() 
    LIMIT 50
}]

# Configure experimental trials
foreach {filename filepath start_time duration has_animals} $trial_clips {
    configure_trial $filepath $start_time $has_animals
}

db close
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

### Cloud Computing
Optimized for spot compute instances:
- **24-48 cores**: Ideal for batch processing
- **Cloud bandwidth**: Faster Internet Archive downloads
- **Cost efficient**: Process dozens of documentaries in minutes

### Recommended Spot Instance Setup
```bash
# Lightning-fast setup on cloud instances
git clone your-repo && cd wildlife-segmenter && uv sync

# Batch download multiple collections
for collection in WildlifeDocumentaries Wildlife_Specials Wildlife_Nature_Documentaries; do
    uv run wildlife_segmenter.py download $collection -d 30 &
done
wait

# Index everything
uv run wildlife_segmenter.py scan
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

**Performance issues:**
- Use `--no-parallel` for debugging
- Check available disk space
- Monitor CPU/memory usage during processing

### Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check the command help: `uv run wildlife_segmenter.py --help`
- Review the database with: `uv run wildlife_segmenter.py query`
