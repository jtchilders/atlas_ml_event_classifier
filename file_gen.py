import numpy,os,json,random,time
import keras,numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


class FileSequencer(keras.utils.Sequence):
   """
    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.
    # Notes
    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.
    # Examples
    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np
        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        class CIFAR10Sequence(Sequence):
            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size
            def __len__(self):
                return len(self.x) // self.batch_size
            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
   """

   def __init__(self,filelist,num_classes,batch_size=20,evt_per_file=1000):
      self.filelist        = filelist
      self.evt_per_file    = evt_per_file
      self.batch_size      = batch_size
      self.nevts           = len(filelist)*self.evt_per_file
      self.nbatches        = int(self.nevts*1./self.batch_size)
      self.num_classes     = num_classes
      
   

   def __getitem__(self, index):
      """Gets batch at position `index`.
      # Arguments
         index: position of the batch in the Sequence.
      # Returns
         A batch
      """
      start = time.clock()

      file_index,event_index = self.get_start_index(index)
      #end_file_index,end_event_index = self.get_end_index(start_file_index,start_event_index)
      
      images = []
      classes = []
      
      # loop until we have enough images
      while len(images) < self.batch_size:
         #logger.debug(' n images: %-10s file_index: %-10s event_index: %-20s',len(images),file_index,event_index)
         # open file
         npfile = numpy.load(self.filelist[file_index])
         image_list = npfile['event_images']
         class_list = npfile['output_labels']
         
         
         while event_index < self.evt_per_file:
            images.append(image_list[event_index,...])
            classes.append(class_list[event_index])
            event_index += 1
         
         event_index = 0
         file_index += 1
      
      #logger.debug(' n images: %-10s n classes: %-10s',len(images),len(classes))

      # convert to numpy array
      np_images = numpy.array(images)
      np_classes = numpy.array(classes)
      
      # convert to categorical
      np_classes = keras.utils.to_categorical(np_classes,self.num_classes)
      #logger.debug(' classes shape: %s',np_classes.shape)
      #tf.logger.info('served %d images in %08.2f seconds',self.batch_size,time.clock() - start)
      return np_images,np_classes
   
   def get_start_index(self,batch_index):
      
      event_index = batch_index*self.batch_size
      file_index = int(event_index / self.evt_per_file)
      return file_index, ( event_index - file_index * self.evt_per_file )
     
   def __len__(self):
      """Number of batch in the Sequence.
      # Returns
          The number of batches in the Sequence.
      """
      return self.nbatches

   def on_epoch_end(self):
      """Method called at the end of every epoch.
      """
      #logger.debug('end of epoch') 
      # shuffle the images
      random.shuffle(self.filelist)

class FileGenerator:
   def __init__(self,filelist,num_classes,batch_size=20,evt_per_file=1000):
      self.filelist        = filelist
      self.evt_per_file    = evt_per_file
      self.batch_size      = batch_size
      self.nevts           = len(filelist)*self.evt_per_file
      self.nbatches        = int(self.nevts*1./self.batch_size)
      self.num_classes     = num_classes
      
      
      self.file_index               = 0
      self.global_event_index       = 0
      self.file_event_index         = 0
      self.npfile                   = None
   
   def __iter__(self):
      return self

   def next(self):
      
      start = time.clock()

      images = []
      classes = []
      
      # loop until we have enough images
      while len(images) < self.batch_size:
         #logger.debug(' n images: %-10s file_index: %-10s global_event_index: %-20s',len(images),self.file_index,self.global_event_index)
         
         # if npfile is set to None, need to open next file_index
         if self.npfile is None:
            # increment file index
            self.file_index += 1
            # check that we are not exceeding the number of files 
            if self.file_index >= len(self.filelist):
               tf.logging.info('end epoch, resetting data generator')
               # shuffle the files
               random.shuffle(self.filelist)
               # reset file index counter
               self.file_index = 0
            # load file
            self.npfile = numpy.load(self.filelist[self.file_index])
            # exctract images and class
            self.image_list = self.npfile['event_images']
            self.class_list = self.npfile['output_labels']
            # ensure they are the same length
            assert len(self.image_list) == len(self.class_list)
            # reset file event index
            self.file_event_index = 0


         # append images and classes for output
         images.append(self.image_list[self.file_event_index])
         classes.append(self.class_list[self.file_event_index])
         # increment counters
         self.file_event_index += 1
         self.global_event_index += 1

         # if there are no events left in file, remove file
         if self.file_event_index >= len(self.image_list):
            self.npfile = None
            self.image_list = None
            self.class_list = None


      # convert to numpy array
      np_images = numpy.array(images)
      np_classes = numpy.array(classes)

      # convert to categorical
      #np_classes = keras.utils.to_categorical(np_classes,self.num_classes)
      #logger.debug(' classes shape: %s',np_classes.shape)
      #tf.logging.info('served %d images in %08.2f seconds',self.batch_size,time.clock() - start)
      return np.rollaxis(np_images,1,4),np_classes
   
     
   def __len__(self):
      """Number of batch in the Sequence.
      # Returns
          The number of batches in the Sequence.
      """
      return self.nbatches

   def on_epoch_end(self):
      """Method called at the end of every epoch.
      """
      logger.debug('end of epoch') 
      # shuffle the images
      random.shuffle(self.filelist)



def load_file(filename):
   ''' loads the numpy compressed file  and returns the stored array '''
   if os.path.exists(filename):
      npfile = numpy.load(filename)
      images = npfile['event_images']
      truth = npfile['output_truth']
      labels = npfile['output_labels']
      return images,truth,labels
   return None,None,None


if __name__ == '__main__':


   import glob
   fl = glob.glob('/Users/jchilders/workdir/ml/output/*.npz')

   fg = FileGenerator(fl,3)

   print len(fg)

   x = 0
   for i in fg:
      a,b = i
      print a.shape,b.shape

      x+=1
      if x > 10: break



