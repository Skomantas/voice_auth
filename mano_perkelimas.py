for subdir, dirs, files in os.walk("data/TRAIN"):
     for file in files:
         if file.endswith('.WAV') & (file.startswith("SA1") | file.startswith("SA2")):
             os.rename(os.path.abspath(subdir)+'/' + file,"/home/skomantas/workspace/speech/tensorflow-speech-recognition/data/amazing/data_" + os.path.basename(subdir) + file)
             print os.path.basename(subdir) + file
             print os.path.join(subdir, file)

             print "data/TRAIN/" + file
             print "data/amazing/data" + os.path.basename(subdir) + file