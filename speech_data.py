import os
import re
import sys
import wave


from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav_go

import numpy
import numpy as np
import skimage.io  # scikit-image

try:
  import librosa
except:
  print("pip install librosa ; if you want mfcc_batch_generator")
# import extensions as xx
from random import shuffle
try:
  from six.moves import urllib
  from six.moves import xrange  # pylint: disable=redefined-builtin
except:
  pass # fuck 2to3


DATA_DIR = 'data/'
pcm_path = "data/amazing_test2/" # 8 bit
mano_path = "data/amazing_test2/"
path = pcm_path
CHUNK = 1924
test_fraction=0.1 # 10% of data for test / verification

class Source:  # labels
  DIGIT_WAVES = 'spoken_numbers_pcm.tar'
  DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
  NUMBER_WAVES = 'spoken_numbers_wav.tar'
  NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
  WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width,height=512# todo: sliding window!
  WORD_WAVES = 'spoken_words_wav.tar'
  TEST_INDEX = 'test_index.txt'
  TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3
  word_phonemes=4
  word = 5  # int vector as opposed to binary hotword
  sentence=6
  sentiment=7
  first_letter=8
  hotword = 9


num_characters = 32

max_word_length = 20
terminal_symbol = 0

def pad(vec, pad_to=max_word_length, one_hot=False,paddy=terminal_symbol):
  for i in range(0, pad_to - len(vec)):
    if one_hot:
      vec.append([paddy] * num_characters)
    else:
      vec.append(paddy)
  return vec

def char_to_class(c):
  return (ord(c) - offset) % num_characters

def string_to_int_word(word, pad_to):
  z = map(char_to_class, word)
  z = list(z)
  z = pad(z)
  return z

class SparseLabels:
  def __init__(labels):
    labels.indices = {}
    labels.values = []

  def shape(self):
    return (len(self.indices),len(self.values))

def sparse_labels(vec):
  labels = SparseLabels()
  b=0
  for lab in vec:
    t=0
    for c in lab:
      labels.indices[b, t] = len(labels.values)
      labels.values.append(char_to_class(c))
      # labels.values[i] = char_to_class(c)
      t += 1
    b += 1
  return labels



def progresshook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def spectro_batch(batch_size=10):
  return spectro_batch_generator(batch_size)

def speaker(filename):  # vom Dateinamen
  # if not "_" in file:
  #   return "Unknown"
  out = filename.split("_")[1]

  return out[0:5]

def get_speakers(path=mano_path):
  # maybe_download(Source.DIGIT_SPECTROS)
  # maybe_download(Source.DIGIT_WAVES)
  files = os.listdir("data/amazing_test2")
  def nobad(name):
    return "_" in name and not "." in name.split("_")[1]
  speakers=list(set(map(speaker,files)))
  print(len(speakers)," speakers: ",speakers)
  return speakers

def wave_mfcc(bufffer):
  mfcc_feat = mfcc(numpy.array(bufffer),44100)
  chunk = mfcc_feat.flatten()
  chunk = numpy.append(chunk, numpy.zeros(CHUNK * 2 - len(chunk)))
  return chunk

def load_wav_file(name):
  # f = wave.open(name, "rb")

  (rate,sig) = wav_go.read(name)

  mfcc_feat = mfcc(sig,rate)
  # fbank_feat = logfbank(sig,rate)
  chunk = mfcc_feat.flatten()
  # chunk = numpy.append(chunk, fbank_feat.flatten())
  chunk = numpy.append(chunk, numpy.zeros(CHUNK * 2 - len(chunk)))
  return chunk


# If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.
# only apply to a subset of all images at one time
def wave_batch_generator(batch_size=10,target=Target.speaker): #speaker
  # maybe_download(source, DATA_DIR)
  if target == Target.speaker: speakers=get_speakers()
  batch_waves = []
  labels = []
  # input_width=CHUNK*6 # wow, big!!
  # files = os.listdir(path)
  files = os.listdir("data/amazing_test2/")
  while True:
    shuffle(files)
    print("loaded batch of %d files" % len(files))
    for wav in files:
      if not wav.endswith(".wav"):continue
      if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
      elif target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
      elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
      else: raise Exception("todo : Target.word label!")

      chunk = load_wav_file("data/amazing_test2/" + wav)

      batch_waves.append(chunk)
      # batch_waves.append(chunks[input_width])
      if len(batch_waves) >= batch_size:
        yield batch_waves, labels
        batch_waves = []  # Reset for next batch
        labels = []

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False, load=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      num = len(images)
      assert num == len(labels), ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      print("len(images) %d" % num)
      self._num_examples = num
    self.cache={}
    self._image_names = numpy.array(images)
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._images=[]
    if load: # Otherwise loaded on demand
      self._images=self.load(self._image_names)

  @property
  def images(self):
    return self._images

  @property
  def image_names(self):
    return self._image_names

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  # only apply to a subset of all images at one time
  def load(self,image_names):
    print("loading %d images"%len(image_names))
    return list(map(self.load_image,image_names)) # python3 map object WTF

  def load_image(self,image_name):
    if image_name in self.cache:
        return self.cache[image_name]
    else:
      image = skimage.io.imread(DATA_DIR+ image_name).astype(numpy.float32)
      # images = numpy.multiply(images, 1.0 / 255.0)
      self.cache[image_name]=image
      return image


  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * width * height
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      # self._images = self._images[perm]
      self._image_names = self._image_names[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self.load(self._image_names[start:end]), self._labels[start:end]


# multi-label
def dense_to_some_hot(labels_dense, num_classes=140):
  """Convert class labels from int vectors to many-hot vectors!"""
  raise "TODO dense_to_some_hot"


def one_hot_to_item(hot, items):
  i=np.argmax(hot)
  item=items[i]
  return item

def one_hot_from_item(item, items):
  # items=set(items) # assure uniqueness
  x=[0]*len(items)# numpy.zeros(len(items))
  i=items.index(item)
  x[i]=1
  return x


def one_hot_word(word,pad_to=max_word_length):
  vec=[]
  for c in word:#.upper():
    x = [0] * num_characters
    x[(ord(c) - offset)%num_characters]=1
    vec.append(x)
  if pad_to:vec=pad(vec, pad_to, one_hot=True)
  return vec

def many_hot_to_word(word):
  s=""
  for c in word:
    x=np.argmax(c)
    s+=chr(x+offset)
    # s += chr(x + 48) # numbers
  return s


def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
  output_shape = tf.reshape(concat, [2])
  sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
  return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
  output_shape = tf.reshape(concat, [2])
  sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
  return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  return numpy.eye(num_classes)[labels_dense]

def extract_labels(names_file,train, one_hot):
  labels=[]
  for line in open(names_file).readlines():
    image_file,image_label = line.split("\t")
    labels.append(image_label)
  if one_hot:
      return dense_to_one_hot(labels)
  return labels

def extract_images(names_file,train):
  image_files=[]
  for line in open(names_file).readlines():
    image_file,image_label = line.split("\t")
    image_files.append(image_file)
  return image_files


def read_data_sets(train_dir,source_data=Source.NUMBER_IMAGES, fake_data=False, one_hot=True):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
    return data_sets
  VALIDATION_SIZE = 2000
  local_file = maybe_download(source_data, train_dir)
  train_images = extract_images(TRAIN_INDEX,train=True)
  train_labels = extract_labels(TRAIN_INDEX,train=True, one_hot=one_hot)
  test_images = extract_images(TEST_INDEX,train=False)
  test_labels = extract_labels(TEST_INDEX,train=False, one_hot=one_hot)
  data_sets.train = DataSet(train_images, train_labels , load=False)
  data_sets.test = DataSet(test_images, test_labels, load=True)
  # data_sets.validation = DataSet(validation_images, validation_labels, load=True)
  return data_sets
