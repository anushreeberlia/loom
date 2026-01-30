# AI Outfit Styler

Upload a clothing item image, get 3 styled outfit recommendations.

## Status

🚧 **Work in Progress**

- [x] Image upload & storage
- [x] PostgreSQL integration
- [ ] Vision AI (describe image)
- [ ] Tag parsing (category, color, style)
- [ ] Catalog with embeddings
- [ ] Outfit retrieval & assembly

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/outfit-styler.git
cd outfit-styler

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Database
createdb outfit_styler
psql outfit_styler -f schema.sql

# Create uploads folder
mkdir uploads

# Run
python app.py
```

### Test

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/v1/outfits:generate
```

Or visit http://localhost:8000/docs

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/outfits:generate` | POST | Generate outfits from image |

## Tech Stack

- **Backend**: FastAPI
- **Database**: PostgreSQL
- **AI**: Gemini (planned)

