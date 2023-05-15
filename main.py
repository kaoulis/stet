import pandas as pd

df = pd.read_csv('tags/tags_bel_sep_a.csv', dtype={'Tag Name': str}, sep=';')
tags = df['Tag Name']
quoted_tags = ','.join("'" + tags.astype(str) + "'")
bracket_tags = ','.join("[" + tags.astype(str) + "]")

