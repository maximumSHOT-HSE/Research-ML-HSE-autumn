import argparse

import numpy as np

from model import VoiceActivityDetector
from utils import build_spectrogram, load_labeled_audio, save_images

import time
import os
import random
from multiprocessing import Process, Queue, Lock
import datetime
import timeit
from StreamBuffer import StreamBuffer


def producer(io_lock, stream_id, signal, block_size_f, sleep_time, queues):
    signal_size_f = len(signal)
    time.sleep(1)
    for l in range(0, signal_size_f, block_size_f):
        r = min(signal_size_f, l + block_size_f)
        block = signal[l: r]
        time.sleep(0.05)
        start_time = time.perf_counter()
        queues[stream_id].put((stream_id, block, start_time))


def worker(io_lock, worker_id, queues, buffers, detector, buffer_locks):
    with io_lock:
        print(f'worker = {worker_id}')

    while True:
        for queue in queues:
            if queue.empty():
                continue
            stream_id, block, send_time = queue.get()

            # START

            proc_time = 0

            with buffer_locks[stream_id]:
                st = time.perf_counter()
                detector.append(block, buffers[stream_id])
                pred = detector.query(stream_buffer)
                fn = time.perf_counter()
                proc_time = fn - st

            # END

            finish_time = time.perf_counter()
            delay = finish_time - send_time
            with io_lock:
                print(f'receive block from stream = {stream_id} with dalay = {delay} seconds, proc time = {proc_time}')


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
    args = parser.parse_args()

    rate, signal, labels = load_labeled_audio(args.audio_path)
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
    io_lock = Lock()

    producers = []
    buffers = []
    buffer_locks = []
    for i, s in enumerate(stream_signals):
        p = Process(target=producer, args=(io_lock, i, s, block_size_f, block_size_s, queues))
        producers.append(p)

        stream_buffer = StreamBuffer(rate)
        buffers.append(stream_buffer)

        buffer_lock = Lock()
        buffer_locks.append(buffer_lock)

    workers = []
    for i in range(n_workers):
        detector = VoiceActivityDetector()
        detector.load(args.model_path)
        detector.setup(rate)
        w = Process(target=worker, args=(io_lock, i, queues, buffers, detector, buffer_locks))
        w.daemon = True
        workers.append(w)

    for w in workers:
        w.start()

    for p in producers:
        p.start()

    for p in producers:
        p.join()
