import os
import tensorflow as tf


def process_tfr_to_tfds(directory_path,
                        process,
                        filter_imgs=False,
                        train_split=0.8,
                        val_split=0.1,
                        test_split=0.1,
                        random_seed=None,
                        group_function=lambda le, re, m, c, z: z):
    """
    Creates a parsed tensorflow dataset from a directory of tfrecords
    files.

    :param directory_path: path of directory containing tfrecords files. 
    Make sure to include / at end.
    :param process: process function that corresponds to shape of data
    :param filter_imgs: True if images should all be of shape=(640,480,3), False otherwise
    :return: parsed tensorflow dataset
    """
    assert (train_split + val_split + test_split) == 1

    files_arr = os.listdir(directory_path)
    file_paths = [directory_path + file_name for file_name in files_arr]

    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(process)
    if filter_imgs:
        dataset = dataset.filter(lambda *args: tf.math.reduce_all(tf.math.equal(tf.shape(args[0]), (640, 480, 3))))
        dataset = dataset.map(lambda *args: (tf.reshape(args[0], (640, 480, 3)),) + args[1:])

    dataset_grouped = dataset.group_by_window(
            key_func= group_function,
            reduce_func= lambda key, dataset: dataset.batch(144),
            window_size= 144
        )

    n_groups = 0
    for _ in dataset_grouped.as_numpy_iterator():
        n_groups += 1

    train_size = int(train_split * n_groups)
    val_size = int(val_split * n_groups)

    shuffled_dataset = dataset_grouped.shuffle(n_groups, seed=random_seed, reshuffle_each_iteration=False)

    train_grouped_ds = shuffled_dataset.take(train_size)
    val_grouped_ds = shuffled_dataset.skip(train_size).take(val_size)
    test_grouped_ds = shuffled_dataset.skip(train_size).skip(val_size)

    # Ungroup the datasets
    train_ds = train_grouped_ds.unbatch()
    val_ds = val_grouped_ds.unbatch()
    test_ds = test_grouped_ds.unbatch()

    return train_ds, val_ds, test_ds


def get_subject_id(tfdata):
    """Helper function for process_tfr_to_tfds"""
    subject_id = tfdata.map(lambda *args: args[-1])
    id_int = list(subject_id.as_numpy_iterator())[0]
    return id_int


def parse_single_eye_tfrecord(element):
    """Process function that parses a tfr element in a raw dataset for process_tfr_to_tfds function.
    Gets mediapipe landmarks, raw image, image width, image height, subject id, and xy labels.
    Use for data generated with eye image TFRecords (single_eye_tfrecords)

    :param element: tfr element in raw dataset
    :return: image, landmarks, label(x,y), subject_id
    """

    data_structure = {
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'eye_img': tf.io.FixedLenFeature([], tf.string),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data_structure)

    landmarks = content['landmarks']
    raw_image = content['eye_img']
    width = content['img_width']
    height = content['img_height']
    label = [content['x'], content['y']]
    subject_id = content['subject_id']

    landmarks = tf.io.parse_tensor(landmarks, out_type=tf.float32)
    landmarks = tf.reshape(landmarks, (478, 3))

    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)

    return image, landmarks, label, subject_id


def rescale_coords_map(eyes, mesh, coords, id):
    """Rescale coordinates from 0-100 range to 0-1 range"""
    return eyes, mesh, coords / 100.0, id
