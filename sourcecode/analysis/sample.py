def sample(df, notes, filename):
  results = df.merge(right=notes, on=["noteId"], how="left")
  results_condensed = notes.loc[notes["noteId"].isin(df["noteId"])]
  results_condensed.sample(n=100).to_csv(f"samples/{filename}.tsv", sep="\t")