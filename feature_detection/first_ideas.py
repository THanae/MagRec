from data_handler.data_importer.data_import import get_probe_data
from feature_detection.training_events import all_events

from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import pandas as pd


# TODO add more training data
random.shuffle(all_events)


def generator_data():
    """
    Generates data around the event or non event, used in the model below
    :return:
    """
    while True:
        data_set = []
        target_set = []
        for loop in range(100):

            number = np.int(np.random.uniform(0, len(all_events)))
            _event, _probe, _number_of_events = all_events[number]
            new_data = pd.read_pickle(
                f'{_event.year}_{_event.month}_{_event.day}_{_event.hour}_{_event.minute}_{_event.second}_{_probe}.pkl')

            # probe_data = get_probe_data(_probe, _event.strftime('%d/%m/%Y'),
            #                             start_hour=(_event - timedelta(hours=2)).hour, duration=4)
            #
            # probe_data.create_processed_column('b_magnitude')
            # probe_data.create_processed_column('vp_magnitude')
            #
            # # probe_data.data.dropna(inplace=True)
            # new_data = probe_data.data.resample('40S').bfill()
            # new_data.fillna(method='ffill', inplace=True)
            # new_data.dropna(inplace=True)
            time_before = np.random.uniform(240, 6960)  # 6*40, 7200-240
            data = new_data[_event - timedelta(seconds=time_before): _event + timedelta(seconds=7200 - time_before)]
            vec_b, vec_v = data['b_magnitude'], data['vp_magnitude']
            vec_b_x, vec_v_x = data['Bx'], data['vp_x']
            vec_b_y, vec_v_y = data['By'], data['vp_y']
            vec_b_z, vec_v_z = data['Bz'], data['vp_z']
            _b_v_array = [vec_b / np.max(vec_b),
                          vec_v / np.max(vec_v),
                          vec_b_x / np.max(vec_b_x),
                          vec_v_x / np.max(vec_v_x),
                          vec_b_y / np.max(vec_b_y),
                          vec_v_y / np.max(vec_v_y),
                          vec_b_z / np.max(vec_b_z),
                          vec_v_z / np.max(vec_v_z)]
            if len(_b_v_array[0]) != 178:
                change = 178 - len(_b_v_array[0])
                if np.sign(change) > 0:
                    for i in range(len(_b_v_array)):
                        _b_v_array[i] = list(_b_v_array[i]) + [0 for _ in range(change)]
                else:
                    for i in range(len(_b_v_array)):
                        _b_v_array[i] = list(_b_v_array[i])[:change]
            _b_v_array = np.array(_b_v_array).transpose((1, 0))
            target = 1 if _number_of_events else 0
            data_set.append(_b_v_array)
            target_set.append(target)
        yield (np.array(data_set), np.array(target_set))


def save_data_to_npy():
    """
    Creates a validation set, a training set and a test set from all_events
    :return:
    """
    reduction = np.int(np.floor(len(all_events) / 8))
    events_list = all_events[:6 * reduction]
    validation_set = all_events[6 * reduction: -10]

    test_set = all_events[-10:]

    def padding1(data_input: list):
        max_list = 0
        for element in data_input:
            if np.array(element).shape[1] > max_list:
                max_list = np.array(element).shape[1]
        global input_length
        input_length = max_list
        for element in data_input:
            if np.array(element).shape[1] != max_list:
                _padding = max_list - np.array(element).shape[1]
                left_padding, right_padding = np.int(np.floor(_padding / 2)), np.int(np.ceil(_padding / 2))
                for i in range(len(element)):
                    element[i] = [0 for _ in range(left_padding)] + list(element[i]) + [0 for _ in range(right_padding)]
        return data_input

    def padding2(data_input: list):
        for element in data_input:
            if np.array(element).shape[1] < input_length:
                _padding = input_length - np.array(element).shape[1]
                left_padding, right_padding = np.int(np.floor(_padding / 2)), np.int(np.ceil(_padding / 2))
                for i in range(len(element)):
                    element[i] = [0 for _ in range(left_padding)] + list(element[i]) + [0 for _ in range(right_padding)]
            elif np.array(element).shape[1] > input_length:
                _padding = np.array(element).shape[1] - input_length
                for i in range(len(element)):
                    element[i] = element[i][:input_length]
        return data_input

    def return_samples_labels(list_of_events: list, pad: int = 1):
        pics, _labels = [], []

        def make_array(_event, _probe, _number_of_events, time_before: float = 1):
            probe_data = get_probe_data(_probe, _event.strftime('%d/%m/%Y'),
                                        start_hour=(_event - timedelta(hours=time_before)).hour, duration=2)

            probe_data.data.dropna(inplace=True)
            probe_data.create_processed_column('b_magnitude')
            probe_data.create_processed_column('vp_magnitude')

            vec_b, vec_v = probe_data.data['b_magnitude'], probe_data.data['vp_magnitude']
            vec_b_x, vec_v_x = probe_data.data['Bx'], probe_data.data['vp_x']
            vec_b_y, vec_v_y = probe_data.data['By'], probe_data.data['vp_y']
            vec_b_z, vec_v_z = probe_data.data['Bz'], probe_data.data['vp_z']
            _b_v_array = [vec_b / np.max(vec_b),
                          vec_v / np.max(vec_v),
                          vec_b_x / np.max(vec_b_x),
                          vec_v_x / np.max(vec_v_x),
                          vec_b_y / np.max(vec_b_y),
                          vec_v_y / np.max(vec_v_y),
                          vec_b_z / np.max(vec_b_z),
                          vec_v_z / np.max(vec_v_z)]
            print(np.array(_b_v_array).shape)
            return _b_v_array

        counter = 0
        for event, probe, number_of_events in list_of_events:
            counter += 1
            test_times = [1]
            if counter % 3 == 0:
                test_times = [0.5, 1, 1.5]

            for time in test_times:
                event_happened = 1 if number_of_events else 0
                b_v_array = make_array(event, probe, number_of_events, time_before=time)
                pics.append(b_v_array)
                _labels.append(event_happened)
        if pad == 1:
            _data = np.array(padding1(pics))
        else:
            _data = np.array(padding2(pics))
        _labels = np.array(_labels)
        return _data, _labels

    data, labels = return_samples_labels(events_list, 1)
    val_data, val_labels = return_samples_labels(validation_set, 2)
    test_data, test_labels = return_samples_labels(test_set, 2)

    np.save('data.npy', data)
    np.save('labels.npy', labels)
    np.save('val_data.npy', val_data)
    np.save('val_labels.npy', val_labels)
    np.save('test_data.npy', test_data)
    np.save('test_labels.npy', test_labels)


# data = np.load('data.npy')
# data = data.transpose((0, 2, 1))
# labels = np.load('labels.npy')
# val_data = np.load('val_data.npy')
# val_data = val_data.transpose((0, 2, 1))
# val_labels = np.load('val_labels.npy')
test_data = np.load('test_data.npy')
test_data = test_data.transpose((0, 2, 1))
test_labels = np.load('test_labels.npy')


def save_data_to_pickle():
    """
    Saves test data to pickle file for faster recollection
    :return:
    """
    for _event, _probe, _happened in all_events:
        probe_data = get_probe_data(_probe, _event.strftime('%d/%m/%Y'),
                                    start_hour=(_event - timedelta(hours=2)).hour, duration=4)
        probe_data.create_processed_column('b_magnitude')
        probe_data.create_processed_column('vp_magnitude')
        new_data = probe_data.data.resample('40S').bfill()
        new_data.fillna(method='ffill', inplace=True)
        new_data.dropna(inplace=True)
        new_data.to_pickle(
            f'{_event.year}_{_event.month}_{_event.day}_{_event.hour}_{_event.minute}_{_event.second}_{_probe}.pkl')


if __name__ == '__main__':
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(128, kernel_size=24, activation='relu', input_shape=(178, 8), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(data, labels, epochs=200, batch_size=48, validation_data=(val_data, val_labels))
    # model.fit_generator(generator_data(), steps_per_epoch=48, epochs=200, validation_data=(val_data, val_labels))
    model.fit_generator(generator_data(), steps_per_epoch=48, epochs=100)

    tf.keras.models.save_model(model, 'test_model.h5')
    # new_model = tf.keras.models.load_model('test_model.h5')

    result = model.predict(test_data, batch_size=48)

    print(test_labels)
    print(result)
