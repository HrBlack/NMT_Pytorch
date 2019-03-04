import pickle
import numpy as np
import matplotlib.pyplot as plt


src_file = '/Users/liushihao/Desktop/NMT_Pytorch/prepared_data/train.jp'
tgt_file = '/Users/liushihao/Desktop/NMT_Pytorch/prepared_data/train.en'

with open(src_file, 'rb') as f:
    src_dataset = pickle.load(f)
    src_sizes = np.array([len(tokens) for tokens in src_dataset])
    type_src = np.array([i for t in src_dataset for i in t ])
    type_src = np.unique(type_src)
    print(type_src)
    print(len(type_src))

with open(tgt_file, 'rb') as f:
    tgt_dataset = pickle.load(f)
    tgt_sizes = np.array([len(tokens) for tokens in tgt_dataset])
    type_tgt = np.array([i for t in tgt_dataset for i in t ])
    type_tgt = np.unique(type_tgt)
    print(type_tgt)
    print(len(type_tgt))
fig, ax = plt.subplots()
n_src, bin_src, _ = ax.hist(src_sizes, bins=100, label='Japanese', alpha=0.5, color='b')
n_tgt, bin_tgt, _ = ax.hist(tgt_sizes, bins=100, label='English', alpha=0.5, color='r')
# ax.plot(bin_src[:-1:2], n_src[::2], 'g-', linewidth=1.5)
plt.xlim((0, 45))
ax.legend(loc='right')
ax.set_title('Distribution of sentence length in English and Japanese')
ax.set_xlabel('Sentence length')
ax.set_ylabel('Number of sentences')
plt.savefig('q2.pdf')
plt.show()
# tokens_src = sum(src_sizes)
# tokens_tgt = sum(tgt_sizes)
# print(tokens_src, tokens_tgt)



# plot the cumulative histogram
# n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
#                            cumulative=True, label='Empirical')
#
# # Add a line showing the expected distribution.
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# y = y.cumsum()
# y /= y[-1]
#
# ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
#
# # Overlay a reversed cumulative histogram.
# ax.hist(x, bins=bins, density=True, histtype='step', cumulative=-1,
#         label='Reversed emp.')
#
# # tidy up the figure
# ax.grid(True)
