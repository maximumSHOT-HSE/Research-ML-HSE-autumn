import numpy as np
import argparse
import time
from multiprocessing import Process, Queue, Lock

import matplotlib.pyplot as plt
import torch

from StreamBuffer import StreamBuffer
from model import VoiceActivityDetector
from utils import load_labeled_audio
from threading import Thread


def producer2(lock, stream_id, signal, block_size_f, sleep_time, queues):
    signal_size_f = len(signal)

    for l in range(0, signal_size_f, block_size_f):
        r = min(signal_size_f, l + block_size_f)
        block = signal[l: r]
        time.sleep(sleep_time)
        start_time = time.perf_counter()
        queues[stream_id].put((stream_id, block, start_time))


def producer(stream_signals, block_size_f, sleep_time, queues):
    n_streams = len(stream_signals)
    ptrs = np.zeros(n_streams, dtype=int)
    prev_time = np.ones(n_streams) * time.perf_counter()

    while True:
        candidates = [i for i in range(n_streams) if ptrs[i] < len(stream_signals[i])]
        if len(candidates) == 0:
            break
        id = candidates[np.random.randint(0, len(candidates))]

        l = ptrs[id]
        r = min(len(stream_signals[id]), l + block_size_f)
        ptrs[id] = r

        block = stream_signals[id][l: r]

        t0 = max(0, sleep_time - (time.perf_counter() - prev_time[id]))
        time.sleep(t0)

        sending_time = time.perf_counter()
        prev_time[id] = sending_time
        queues[id].put((block, sending_time))


def worker(lock, worker_id, queues, buffers, detector, buffer_locks):
    max_delay = 0
    min_delay = np.inf
    sum_delay = 0
    cnt_requests = 0

    _start = time.perf_counter()

    while True:
        for stream_id, queue in enumerate(queues):
            if queue.empty():
                continue

            block, sending_time = queue.get()

            # START
            with buffer_locks[stream_id]:
                detector.append(block, buffers[stream_id])
                pred = detector.query(buffers[stream_id])
            # END

            finish_time = time.perf_counter()
            delay = finish_time - sending_time

            sum_delay += delay
            cnt_requests += 1

            if cnt_requests > 500:
                cnt_requests = 1
                min_delay = np.inf
                max_delay = 0
                sum_delay = delay

            if delay > max_delay:
                max_delay = delay

            if delay < min_delay:
                min_delay = delay

            with lock:
                # print(f'pred = {np.sum(pred)}')
                print(f'worker {worker_id}, new max delay = {max_delay}, new min delay = {min_delay}'
                      f' new mean = {sum_delay / cnt_requests} | on stream {stream_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Parallel real time voice activity detector"""
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model is stored'
    )
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to the raw audio file in .wav format'
    )
    parser.add_argument(
        '--streams',
        type=int,
        default=20,
        help='The number of streams (audio sources)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='The number of workers for parallelism'
    )
    parser.add_argument(
        '--block-size',
        type=float,
        default=0.02,
        help='Size of signal blocks in seconds'
    )
    parser.add_argument(
        '--graph-runtime',
        type=float,
        default=None
    )
    args = parser.parse_args()

    VoiceActivityDetector.DEVICE = torch.device('cpu')

    rate, signal, labels = load_labeled_audio(args.audio_path)

    # X = int(rate * 10)
    # signal = signal[: X]
    # labels = labels[: X]

    signal_size_f = len(signal)

    n_streams = args.streams
    n_workers = args.workers
    block_size_s = args.block_size
    block_size_f = int(block_size_s * rate)

    stream_signal_size_f = signal_size_f // n_streams
    stream_signals = []

    for i in range(n_streams):
        l = i * stream_signal_size_f
        r = l + stream_signal_size_f
        stream_signals.append(signal[l: r])

    queues = []
    for i in range(n_streams):
        q = Queue()
        queues.append(q)
    lock = Lock()

    producer = Process(target=producer, args=(stream_signals, block_size_f, block_size_s, queues))

    # producers = []
    buffers = []
    buffer_locks = []
    for i, stream in enumerate(stream_signals):
        # p = Process(target=producer, args=(io_lock, i, s, block_size_f, block_size_s, queues))
        # p = Thread(target=producer2, args=(lock, i, stream, block_size_f, block_size_s, queues))
        # producers.append(p)

        stream_buffer = StreamBuffer(rate)
        buffers.append(stream_buffer)

        buffer_lock = Lock()
        buffer_locks.append(buffer_lock)

    workers = []
    for i in range(n_workers):
        detector = VoiceActivityDetector()
        detector.load(args.model_path)
        detector.setup(rate)

        if i == 0:
            print(f'MODEL\n{detector}')

        w = Process(target=worker, args=(lock, i, queues, buffers, detector, buffer_locks))
        w.daemon = True
        # w = Thread(target=worker, args=(io_lock, i, queues, buffers, detector, buffer_locks))
        # w.setDaemon(True)
        workers.append(w)

    for w in workers:
        w.start()

    producer.start()
    producer.join()

    # for p in producers:
    #     p.start()

    # for p in producers:
    #     p.join()

    # for w in workers:
    #     w.join()
