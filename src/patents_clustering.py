# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR

archive_name = '08_patents.zip'
kwargs = {
    'filepath_or_buffer': DATA_DIR.joinpath('external').joinpath(archive_name),
    'index_col': 0,
    'parse_dates': [0]
}
df = pd.read_csv(**kwargs)
X = df.copy()
le = LabelEncoder()
le.fit(X.iloc[:, 1])
X['category'] = le.transform(X.iloc[:, 1])

print(X.info())
clustering = AgglomerativeClustering().fit(
    np.array(X.iloc[:, -1]).reshape(-1, 1)
)
print(clustering.labels_)


dbscan = DBSCAN(eps=3, min_samples=2)
dbscan.fit(X)
