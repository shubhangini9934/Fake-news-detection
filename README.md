# Smart News Verifier

Smart News Verifier is a full-stack fake-news verification app with a React + Tailwind frontend, an Express + Socket.io backend, and a Flask NLP microservice for text classification.

## Folder structure

```text
smart-news-verifier/
├── client/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── .env.example
│   ├── package.json
│   └── tailwind.config.js
├── server/
│   ├── src/
│   │   ├── config/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── services/
│   │   ├── index.js
│   │   └── socket.js
│   ├── .env.example
│   └── package.json
└── ml-model/
    ├── app.py
    ├── train_model.py
    ├── model.pkl
    ├── vectorizer.pkl
    └── requirements.txt
```

## Setup

### Backend

1. Go to [server](C:/Users/sahan/OneDrive/Desktop/Fake news detection/server)
2. Run `npm install`
3. Copy `.env.example` to `.env`
4. Start with `npm run dev`

Example `.env`:

```env
PORT=4000
CLIENT_URL=http://localhost:5173
MONGODB_URI=mongodb://127.0.0.1:27017/smart-news-verifier
NEWS_API_PROVIDER=newsapi
NEWS_API_KEY=your_news_api_key
NEWS_COUNTRY=us
NEWS_LANGUAGE=en
POLL_INTERVAL_MS=30000
ML_SERVICE_URL=http://127.0.0.1:5001
```

### ML service

1. Go to [ml-model](C:/Users/sahan/OneDrive/Desktop/Fake news detection/ml-model)
2. Create a virtual environment with `python -m venv .venv`
3. Activate it with `.venv\Scripts\activate`
4. Run `pip install -r requirements.txt`
5. Optionally retrain with `python train_model.py`
6. Start with `python app.py`

### Frontend

1. Go to [client](C:/Users/sahan/OneDrive/Desktop/Fake news detection/client)
2. Run `npm install`
3. Copy `.env.example` to `.env`
4. Start with `npm run dev`

Example `.env`:

```env
VITE_API_URL=http://localhost:4000
```

## Deployment

- Frontend: Vercel or Netlify
- Backend: Render or Railway
- ML service: Render or Railway as a separate Flask service

## API summary

- `GET /news`
- `POST /verify`
- `GET /history`
- `PATCH /news/:id/bookmark`

## Features included

- React + Tailwind dashboard with animated modern UI
- Real-time scrolling headline ticker
- Socket.io breaking-news popup alerts
- News cards with headline, source, timestamp, and verify button
- Custom text verification search bar
- Dark and light mode toggle
- TRUE / FAKE / saved filters
- Bookmark support
- MongoDB persistence for news and verification history
- Flask NLP classifier with TF-IDF + Logistic Regression
- Verification distribution chart
- Loading and error states
