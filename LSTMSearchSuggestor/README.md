## Data Sources
* billboard
* Kaggle-GeniusSongLyrics
* MillionSongSubset
* lastfm

## Running the app
  python src/train.py
  # after training completes:
  python src/inference.py --prefix "love "

  BEST RESULTS
  python src/inference.py --prefix "love " --top_k 5 --temperature 1.3 --top_p 0.95