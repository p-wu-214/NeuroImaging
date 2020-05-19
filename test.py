import gcsfs
if __name__ == '__main__':
    fs = gcsfs.GCSFileSystem(project='Tensorflow')
    fs.ls('trends_dataset')