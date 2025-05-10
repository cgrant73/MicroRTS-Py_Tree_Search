from kingdomtreeworking import bigBatch
from kingdomtreeworking import generateForBatch
from multiprocessing import Pool, cpu_count, freeze_support

if __name__ == "__main__":
    # on Windows freezing into an exe you’d need this:
    freeze_support()

    batch = generateForBatch(batch_num=100)
    print("i am child")
    results = bigBatch(batch)
    print(results.shape)
    # for thing in results:
    #     print(thing)

