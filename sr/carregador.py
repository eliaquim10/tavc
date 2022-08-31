from .utils import *
from tensorflow.python.data.experimental import AUTOTUNE

size = 50

class Loader:
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='S2TLD/JPEGImages',
                 caches_dir='S2TLD/caches',
                 percent=0.5):

        _scales = [2, 3, 4, 8]
        _files = sorted(os.listdir(images_dir))[:size]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        idxs = int(size*percent) 
        if subset == 'train':
            self.image_ids = _files[:idxs]
        elif subset == 'valid':
            self.image_ids = _files[idxs:]
        else:
            raise ValueError("subset must be 'train' or 'valid'")
            
        self.subset = subset
        self.downgrade = downgrade
        self.images_dir = images_dir
        self.caches_dir = caches_dir
        # self.caches_dir = f'{caches_dir}/{downgrade}' 

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        # ds = tf.data.Dataset.zip((self.noise_dataset(), self.denoise_dataset()))
        ds = self.noise_dataset()
        
        # ds = tf.data.Dataset.list_files(f'{self.images_dir}/*.jpg')
        # ds = ds.map(lambda filename: load(filename), num_parallel_calls=AUTOTUNE)

        # return ds
        ds = ds.map(lambda original: noiser(original, std=(20)/(255)), num_parallel_calls=AUTOTUNE)
        
        if random_transform:
            ds = ds.map(lambda noise, original: random_crop(noise, original, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def denoise_dataset(self):
        if not os.path.exists(self._denoise_images_dir()):
            raise ValueError("subset not found ")


        ds = self._images_dataset(self._denoise_image_files()).cache(self._denoise_cache_file())
        
        if not os.path.exists(self._denoise_cache_index()):
            self._populate_cache(ds, self._denoise_cache_file())

        return ds

    def noise_dataset(self):
        if not os.path.exists(self._noise_images_dir()):
            raise ValueError("subset not found ")
            
        ds = self._images_dataset(self._noise_image_files()).cache(self._noise_cache_file())
        
        if not os.path.exists(self._noise_cache_index()):
            self._populate_cache(ds, self._noise_cache_file())
        
        return ds

    def _denoise_cache_file(self):
        return os.path.join(self.caches_dir, f'color_{self.subset}_denoise.cache')

    def _noise_cache_file(self):
        return os.path.join(self.caches_dir, f'color_{self.subset}_noise_{self.downgrade}_X{self.scale}.cache')

    def _denoise_cache_index(self):
        return f'{self._denoise_cache_file()}.index'

    def _noise_cache_index(self):
        return f'{self._noise_cache_file()}.index'

    def _denoise_image_files(self):
        images_dir = self._denoise_images_dir()
        return [os.path.join(images_dir, f'{image_id}') for image_id in self.image_ids]

    def _noise_image_files(self):
        images_dir = self._noise_images_dir()
        return [os.path.join(images_dir, f'{image_id}') for image_id in self.image_ids]

    def _noise_image_file(self, image_id):
        return f'{image_id}.png'

    def _denoise_images_dir(self):
        return os.path.join(self.images_dir)

    def _noise_images_dir(self):
        return os.path.join(self.images_dir)

    @staticmethod
    def _images_dataset(image_files):
        ds = []
        for filename in image_files:
            with tf.device("/GPU:0"):
                image = load(filename)
                if image is not None:
                    ds.append(image)
                else:
                    os.remove(filename)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        
        # with tf.device("/CPU:0"):
        # ds = tf.data.Dataset.from_tensor_slices(image_files[:2])
        # ds = ds.map(tf.io.read_file)
        # ds = ds.map(lambda x: tf.image.decode_jpeg(x, channels=3), num_parallel_calls=AUTOTUNE)
        
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')

# data_loader_train = Loader(scale=4, 
#                 subset='train', 
#                 downgrade='bicubic', 
#                 images_dir="dataset/S2TLD/JPEGImages",
#                 caches_dir="dataset/S2TLD/caches",
#                 percent=0.7)

# print("="*100)
# train = data_loader_train.dataset(4, random_transform=True)
# print(train)
# print(len(os.listdir("dataset/S2TLD/JPEGImages")))
# load_image("dataset/S2TLD/JPEGImages/2020-03-30 11_30_03.690871079.jpg")
def __main__():
    pass
if __name__ == "__main__":
    pass