## Data instructions

This project expects you to download the **5 Million Song Lyrics** dataset from Kaggle and place it at:

`data/lyrics.csv`

The original Kaggle dataset page:
https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset

The scripts in `src/` assume the CSV contains at least the following columns:
- `song` or `title` (title of the song)
- `artist` (artist name)
- `lyrics` (the song lyrics)

If your file uses different column names, pass `--lyrics_col` / `--title_col` / `--artist_col` flags to the scripts.
