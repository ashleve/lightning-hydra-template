from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, ShardingFilter, Shuffler, Mapper, RandomSplitter

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import torch

def test_func(x):
    return x*2

data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))

datapipe = IterableWrapper(zip(data,labels)) \
        .shuffle() \
        .sharding_filter()##

rs = MultiProcessingReadingService(num_workers=2)
# for x in datapipe:
#     print(x)
dataloader = DataLoader2(datapipe,reading_service=rs)
for x in dataloader:
    # print(x)
    # print()
    data_batch, label_batch = x
    print(data_batch.shape, label_batch)

dataloader.shutdown()