# Loom - Outfit Builder

Your personal AI-powered wardrobe assistant. Upload your closet, get daily outfit recommendations based on weather, occasion, and your style preferences.

## Features

- **Personal Closet** - Upload and manage your wardrobe items with automatic tagging
- **Daily Outfits** - Get 3 curated outfit suggestions each day
- **Weather-Aware** - Recommendations adapt to local weather conditions
- **Occasion Detection** - Auto-detects work hours, evenings, weekends for appropriate styling
- **Custom Moods** - Type any mood/occasion for personalized suggestions
- **Style Learning** - Like/dislike feedback improves recommendations over time
- **Top Rotation** - FIFO queue ensures variety in outfit suggestions
- **Save & Track** - Bookmark outfits and track what you've worn
- **Background Removal** - Client-side AI removes backgrounds from item photos

## Tech Stack

- **Backend**: FastAPI + Python
- **Database**: PostgreSQL with pgvector for embeddings
- **AI**: OpenAI (GPT-4 Vision for tagging, text-embedding-3-small for similarity)
- **Images**: Cloudinary for storage and transformations
- **Weather**: OpenWeatherMap API
- **Auth**: Google OAuth + Email/Password with JWT
- **Hosting**: Railway

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with pgvector extension
- Cloudinary account
- OpenAI API key
- OpenWeatherMap API key (free tier works)

### Setup

```bash
# Clone
git clone https://github.com/anushreeberlia/loom.git
cd loom

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Environment variables
cp env.example .env
# Edit .env with your API keys

# Database
createdb loom
psql loom -f schema.sql

# Run
uvicorn app:app --reload --port 8080
```

Visit http://localhost:8080

## Environment Variables

```
DATABASE_URL=postgresql://user:pass@localhost/loom
OPENAI_API_KEY=sk-...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
OPENWEATHERMAP_API_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
JWT_SECRET=...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/closet` | GET | Daily outfits page |
| `/inventory` | GET | Manage closet items |
| `/v1/closet/items` | GET/POST | List/add closet items |
| `/v1/closet/daily` | GET | Get daily outfit recommendations |
| `/v1/closet/outfits:generate` | POST | Generate outfits from a specific item |
| `/v1/closet/feedback` | POST | Submit like/dislike feedback |
| `/v1/closet/outfits/save` | POST | Save outfit for later |
| `/v1/closet/outfits/saved` | GET | List saved outfits |
| `/v1/closet/outfits/worn` | GET | List worn outfit history |
| `/auth/google` | GET | Google OAuth login |
| `/auth/register` | POST | Email/password registration |
| `/auth/login` | POST | Email/password login |

## Project Structure

```
├── app.py              # Main FastAPI application
├── schema.sql          # Database schema
├── requirements.txt    # Python dependencies
├── services/
│   ├── retrieval.py    # Outfit retrieval & assembly
│   ├── collage.py      # Outfit image generation
│   ├── weather.py      # Weather API integration
│   └── auth.py         # Authentication
└── static/
    ├── closet.html     # Daily outfits UI
    ├── inventory.html  # Closet management UI
    ├── index.html      # Demo page
    └── login.html      # Auth page
```

## License

MIT
