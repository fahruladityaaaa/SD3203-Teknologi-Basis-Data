TUGAS TBD
Nama   : Muhammad Fahrul Aditya
NIM    : 121450156
kelas  : 121450156

# Three Ways of Storing and Accessing Lots of Images in Python
>https://realpython.com/storing-images-in-python/

Artikel "Three Ways of Storing and Accessing Lots of Images in Python" oleh Rebecca Stone membahas tiga strategi untuk menyimpan dan mengakses sejumlah besar gambar menggunakan Python. Semakin banyak gambar yang perlu ditangani, semakin kompleks prosesnya. Algoritma machine learning seperti convolutional neural networks (CNNs) dapat membantu mengelola volume data yang besar dan bahkan belajar dari data itu sendiri. Artikel ini juga mengenalkan ImageNet, sebuah database gambar publik yang terkenal digunakan untuk melatih model pada berbagai tugas seperti klasifikasi objek, deteksi, dan segmentasi, yang terdiri dari lebih dari 14 juta gambar. Bayangkan waktu yang dibutuhkan untuk memuat semua gambar tersebut ke dalam memori untuk pelatihan, sering kali dalam batch yang mencapai ratusan atau ribuan kali.

Dalam artikel ini, metode penyimpanan gambar dalam format file .png, menggunakan lightning memory-mapped databases (LMDB), dan penyimpanan dalam format data hierarki (HDF5) akan dibahas.


```
import numpy as np
import pickle
from pathlib import Path
# Path to the unzipped CIFAR data
data_dir = Path("data/cifar-10-batches-py/")
# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])
print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

### Setup for Storing Images on Disk
pastikan bahwa kita telah menginstall python 3.X. selanjutnya kita akan melakukan manipulasi gambar dengan `pillow` :
```
$ pip install Pillow
```
dan kita juga dapat menginstall menggunakan `anaconda` :
```
$ conda install -c conda-forge pillow
```
terdapat juga catatan dari artikel ini yaitu : PIL adalah versi asli dari Python Imaging Library, yang tidak lagi dipertahankan dan tidak kompatibel dengan Python 3.x. Jika kita sudah menginstal sebelumnya PIL, pastikan untuk menghapus instalannya sebelum menginstal Pillow, karena keduanya tidak dapat ada bersamaan.

### Getting Started With LMDB
LMDB (Lightning Memory-Mapped Database), yang sering disebut sebagai Lightning Database, menonjol dengan kecepatan dan penggunaan file yang dapat diandalkan dalam memori. Berbeda dengan database relasional, LMDB menyimpan kunci-nilai dalam struktur pohon B+. Struktur ini menyerupai pohon dan disimpan di memori, memungkinkan penjelajahan antar simpul dengan efisiensi tinggi. Kunci-kunci pada pohon B+ disesuaikan dengan ukuran halaman sistem operasi, yang mengoptimalkan efisiensi akses data. LMDB juga memanfaatkan pemetaan memori, mengembalikan penunjuk langsung ke alamat memori kunci dan nilai tanpa perlu menyalin data. Dengan memanfaatkan sistem file dan implementasi yang mendasarinya, LMDB memaksimalkan kinerjanya.

Bagi yang ingin mendalami topik ini lebih lanjut, terdapat artikel yang membahas pohon B+ dan visualisasi penyisipan simpul yang dapat menjadi sumber belajar yang bermanfaat. Selanjutnya, kita dapat menggunakan python binding untuk perpustakaan LMDB C, yang dapat diinstal melalui pip.

```
$ pip install lmdb
```
Opsi dengan menggunakan `Anaconda`:
```
$ conda install -c conda-forge python-lmdb
```
### Getting Started With HDF5
HDF5, singkatan dari Hierarchical Data Format, adalah format file yang disebut juga sebagai HDF4 atau HDF5. Meskipun HDF4 masih ada, kita lebih fokus pada HDF5 karena ini adalah versi yang sedang dipertahankan.

Format ini berasal dari National Center for Supercomputing Applications, diciptakan sebagai format data ilmiah yang ringkas dan mudah dipindahkan. Jika Anda ingin melihat seberapa luas penggunaannya, NASA menjelaskan penggunaan HDF5 dalam proyek Data Bumi mereka.

File HDF terdiri dari dua jenis objek utama:

Dataset: Ini adalah array multidimensi, yang berisi data aktual.
Group: Ini adalah wadah yang dapat berisi dataset atau grup lainnya. Dengan adanya grup, Anda bisa memiliki struktur yang lebih terorganisir dalam file HDF.
Dataset dapat berisi array multidimensi dengan berbagai ukuran dan tipe data, namun setiap dataset harus memiliki dimensi dan tipe data yang seragam. Meskipun begitu, karena grup dan dataset dapat bersarang, Anda masih dapat menciptakan struktur data yang heterogen jika diperlukan.

```
$ pip install h5py
```
Dengan menggunakan `Anaconda`.
```
$ conda install -c conda-forge h5py
```
Jika kita sudah bisa mengimportkan h5py dari shell Python, maka sudah dikatakan berhasil dan telah diatur dengan benar.

### Storing a Single Image
Sekarang, setelah kita memiliki pemahaman umum tentang metode-metode tersebut, mari kita langsung masuk ke perbandingan kuantitatif dari tugas-tugas dasar yang kita perhatikan: berapa lama waktu yang diperlukan untuk membaca dan menulis file, serta berapa banyak memori disk yang digunakan. Ini juga akan berfungsi sebagai pengantar dasar tentang cara kerja metode-metode tersebut, dengan contoh kode tentang cara menggunakannya.

Ketika saya menyebut "file", saya mengacu pada banyak file. Namun, penting untuk membedakan karena beberapa metode dapat dioptimalkan untuk operasi dan jumlah file yang berbeda.

Untuk tujuan eksperimen, kita dapat membandingkan kinerja antara berbagai jumlah file, dengan faktor 10 dari satu gambar hingga 100.000 gambar. Misalnya, karena kami memiliki lima kelompok CIFAR-10 yang masing-masing berisi 10.000 gambar, kita dapat menggunakan setiap kelompok gambar dua kali untuk mencapai total 100.000 gambar.

Untuk mempersiapkan eksperimen, Anda perlu membuat sebuah folder untuk setiap metode yang berisi semua file database atau gambar, dan menyimpan jalur ke direktori-direktori tersebut dalam variabel.

```
from pathlib import Path
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```
`Path` tidak secara otomatis membuat folder untuk kita kecuali kita secara khusus memintanya untuk melakukannya.
```
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```

### Storing to Disk
Masukan kita untuk percobaan ini adalah sebuah gambar gambar tunggal, yang saat ini berada di memori sebagai larik NumPy. Anda ingin menyimpannya terlebih dahulu ke dalam disk sebagai gambar .png, dan menamainya dengan menggunakan ID gambar yang unik image_id. Hal ini dapat dilakukan dengan menggunakan paket Pillow yang telah Anda instal sebelumnya:
```
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```
### Storing to LMDB
Pertama, LMDB merupakan sistem penyimpanan nilai-kunci di mana setiap entri disimpan sebagai larik byte. Dalam konteks kami, kunci akan menjadi pengenal unik untuk setiap gambar, dan nilainya adalah representasi byte dari gambar itu sendiri. Baik kunci maupun nilai diharapkan berupa string, sehingga umumnya nilai akan diserialisasi sebagai string sebelum disimpan, dan kemudian diunserialisasi saat dibaca kembali.

Anda dapat menggunakan pustaka seperti pickle untuk melakukan serialisasi. Objek Python apa pun dapat diserialisasi, jadi Anda mungkin juga ingin menyertakan data meta gambar dalam basis data untuk menghindari masalah dalam melampirkan data meta kembali ke data gambar saat memuat dataset dari disk.

Anda bisa membuat kelas Python dasar untuk representasi gambar dan metadata, seperti contoh di bawah ini:
```
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary 
        # for this dataset, but some datasets may include images of 
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```

### Storing With HDF5
Ingatlah bahwa file HDF5 dapat berisi lebih dari satu kumpulan data. Dalam kasus yang agak sepele ini, Anda dapat membuat dua set data, satu untuk gambar, dan satu lagi untuk meta datanya:

```
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```

