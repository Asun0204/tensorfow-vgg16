import tflearn.datasets.oxflower17 as oxflower17
import numpy as np

class BatchDatset:

    def __init__(self):
        print("Initializing Batch Dataset Reader...")
        self._read_images()
        self.batch_offset = 0
        self.epochs_completed = 0

    def _read_images(self):
        self.images, self.annotations = oxflower17.load_data()
        self.image_mean = np.mean(self.images, axis=(1,2), keepdims=True)
        self.images -= np.mean(self.images, axis=(1,2), keepdims=True)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        # return self.images[start:end], self.annotations[start:end]
        data = self.images[start:end]
        labels = self.annotations[start:end]
        return data, labels
        
    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, int(self.images.shape[0]), size=[batch_size]).tolist()
        data = self.images[indexes]
        labels = self.annotations[indexes]
        return data, labels