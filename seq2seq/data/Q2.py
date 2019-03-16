import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

src_file = '/Users/liushihao/Desktop/NMT_Pytorch/raw_data/train.jp'
tgt_file = '/Users/liushihao/Desktop/NMT_Pytorch/raw_data/train.en'
SPACE_NORMALIZER = re.compile("\s+")
tgt_sizes = []
src_sizes = []
src_dataset = []
tgt_dataset = []

with open(tgt_file, 'r') as f:
    for line in f:
        line = SPACE_NORMALIZER.sub(" ", line).strip().split()
        # print(line)
        np.array([tgt_sizes.append(len(line))])
        tgt_dataset.extend(line)
    print(np.mean(tgt_sizes))

with open(src_file, 'r') as f:
    for line in f:
        line = SPACE_NORMALIZER.sub(" ", line).strip().split()
        # print(line)
        np.array([src_sizes.append(len(line))])
        src_dataset.extend(line)
    print(np.mean(src_sizes))



fig, ax = plt.subplots()
n_src, bin_src, _ = ax.hist(src_sizes, bins=100, label='Japanese', alpha=0.5, color='b')
n_tgt, bin_tgt, _ = ax.hist(tgt_sizes, bins=100, label='English', alpha=0.5, color='r')
# ax.plot(bin_src[:-1:2], n_src[::2], 'g-', linewidth=1.5)
plt.xlim((0, 45))
ax.legend(loc='right')
ax.set_title('Distribution of sentence length in English and Japanese')
ax.set_xlabel('Sentence length')
ax.set_ylabel('Number of sentences')
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
