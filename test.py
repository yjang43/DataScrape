import bow
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

dataset = bow.create_dataset('articles')
pca_data = bow.pca_transform(dataset)
bow.visualize_classification(pca_data, 'article_classification.json')

# bow.visualize(pca_data)
# hac_result = bow.hac(pca_data)
# fig = plt.figure(figsize=(25, 10))
# print(hac_result[0: -1, :].shape)
# dn = dendrogram(hac_result[0: -1, :])
# plt.show()


